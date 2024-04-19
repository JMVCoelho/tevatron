from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor

from transformers import PreTrainedModel, AutoModel
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from transformers.file_utils import ModelOutput
from tevatron.retriever.arguments import ModelArguments, TevatronTrainingArguments as TrainingArguments

import logging
logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    bow_loss: Optional[Tensor] = None


class EncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 matrioshka_reg: bool = False,
                 lm_head: nn.Linear = None,
                 ld_head: nn.Linear = None,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature

        # for matrioshka
        self.matrioshka_reg = matrioshka_reg
        self.lm_head = lm_head

        # for landmark
        self.ld_head = ld_head
        

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_reps = self.encode_query(query) if query else None
        p_reps = self.encode_passage(passage) if passage else None

        # for inference
        if q_reps is None or p_reps is None:
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        # for training
        if self.training:
            if self.is_ddp:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            bow_loss = None
            if self.matrioshka_reg:
                do_sub_vectors=True
                full_weights = torch.zeros([p_reps.shape[0], 50304], device=p_reps.device)
                for idx, _input_ids in enumerate(passage['input_ids']):
                    unique_tokens = torch.unique(_input_ids)
                    unique_tokens = unique_tokens[unique_tokens != 0] # remove special token 0
                    num_tokens = unique_tokens.shape[0]
                    if num_tokens > 0:
                        full_weights[idx, unique_tokens] = 1.0 / num_tokens

                projections = self.lm_head(p_reps)
                bow_loss = torch.mean(-torch.sum(full_weights * torch.nn.functional.log_softmax(projections, dim=-1), dim=1))

                if do_sub_vectors:
                    N = 4
                    sub_embedding_weights = [torch.zeros([p_reps.shape[0], 50304], device=p_reps.device) for _ in range(N)]
                    for idx, _input_ids in enumerate(passage['input_ids']):
                        quarter_size =  (passage['attention_mask'][idx].sum()//N).item()
                        for quarter_idx in range(4):
                            start_idx = quarter_idx * quarter_size
                            end_idx = (quarter_idx + 1) * quarter_size

                            quarter_tokens = _input_ids[start_idx:end_idx]
                            unique_tokens = torch.unique(quarter_tokens)
                            unique_tokens = unique_tokens[unique_tokens != 0]  # remove special token 0
                            num_tokens = unique_tokens.shape[0]

                            if num_tokens > 0:
                                sub_embedding_weights[quarter_idx][idx, unique_tokens] = 1.0 / num_tokens

                    sub_embedding_projections = ()
                    start_idx = 0
                    num_sub_embeddings = p_reps.shape[1] // N
                    sub_embedding_list = [num_sub_embeddings] * N
                    for num_feat in sub_embedding_list:
                        end_idx = start_idx + num_feat
                        if self.lm_head.bias is None:
                            sub_embedding_projections += (torch.matmul(p_reps[:, start_idx:end_idx], (self.lm_head.weight[:, start_idx:end_idx]).t()),)
                        else:
                            sub_embedding_projections += (torch.matmul(p_reps[:, start_idx:end_idx], (self.lm_head.weight[:, start_idx:end_idx]).t()) + self.lm_head.bias,)
                        start_idx = end_idx

                    subvector_bow_loss = 0
                    for w, proj in zip(sub_embedding_weights, sub_embedding_projections):
                        subvector_bow_loss += torch.mean(-torch.sum(w * torch.nn.functional.log_softmax(proj, dim=-1), dim=1))

                bow_loss = (bow_loss + subvector_bow_loss)/5


            scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)

            target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
            target = target * (p_reps.size(0) // q_reps.size(0))

            loss = self.compute_loss(scores / self.temperature, target)

            if bow_loss is not None:
                loss = 0.7*loss + 0.3*bow_loss

            if self.is_ddp:
                loss = loss * self.world_size  # counter average weight reduction
        # for eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
            bow_loss=bow_loss
        )

    def encode_passage(self, psg):
        raise NotImplementedError('EncoderModel is an abstract class')

    def encode_query(self, qry):
        raise NotImplementedError('EncoderModel is an abstract class')

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)
    
    def gradient_checkpointing_enable(self, **kwargs):
        try:
            self.encoder.model.gradient_checkpointing_enable()
        except Exception:
            self.encoder.gradient_checkpointing_enable()

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):  
        if not model_args.local:
            if "jamba" in model_args.model_name_or_path.lower():
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    device_map="auto",
                    llm_int8_skip_modules=["mamba"]
                )
                base_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, 
                                                                trust_remote_code=True,
                                                                torch_dtype=torch.bfloat16,
                                                                attn_implementation="flash_attention_2",
                                                                quantization_config=quantization_config,
                                                                **hf_kwargs)
                print("loaded jamba!")
            else:
                base_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            print("loaded hf model")
        else:
            state_dict = torch.load(f"{model_args.model_name_or_path}/pytorch_model.bin")
            base_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, local_files_only=True, state_dict=state_dict, attn_implementation="flash_attention_2", **hf_kwargs)
            print("loaded local model")

        print(f"Model class: {base_model.__class__}")

        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0

        lm_head = None
        if model_args.matrioshka_reg:
            lm_head = nn.Linear(base_model.config.hidden_size, base_model.config.vocab_size, bias=False)
            state_dict = torch.load(f"{model_args.model_name_or_path}/pytorch_model.bin")
            lm_head.weight.data.copy_(state_dict['embed_out.weight'])
        
        ld_head = None
        #if model_args.pooling == "landmark":
        #    ld_head = nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size//4, bias=False)
        #    state_dict = torch.load(f"{model_args.model_name_or_path}/pytorch_model.bin")
        #    ld_head.weight.data.copy_(state_dict['ld_head.weight'])


        if model_args.lora or model_args.lora_name_or_path:
            if train_args.gradient_checkpointing:
                base_model.enable_input_require_grads()
            if model_args.lora_name_or_path:
                lora_config = LoraConfig.from_pretrained(model_args.lora_name_or_path, **hf_kwargs)
                lora_model = PeftModel.from_pretrained(base_model, model_args.lora_name_or_path)
            else:
                lora_config = LoraConfig(
                    base_model_name_or_path=model_args.model_name_or_path,
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules=model_args.lora_target_modules.split(','),
                    inference_mode=False
                )
                lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
            
        else:

            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                matrioshka_reg=model_args.matrioshka_reg,
                lm_head=lm_head,
                ld_head=ld_head,
            )

        return model

    @classmethod
    def load(cls,
            model_name_or_path: str,
            local: bool = False,
            pooling: str = 'cls',
            normalize: bool = False,
            lora_name_or_path: str = None,
            **hf_kwargs):
        
        if not local:
            base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
            print("loaded hf model")
        else:
            state_dict = torch.load(f"{model_name_or_path}/pytorch_model.bin")
            base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, local_files_only=True, state_dict=state_dict, attn_implementation="flash_attention_2", **hf_kwargs)
            print("loaded local model")

        print(f"Model class: {base_model.__class__}")
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if lora_name_or_path:
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=pooling,
                normalize=normalize
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=pooling,
                normalize=normalize
            )
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)
