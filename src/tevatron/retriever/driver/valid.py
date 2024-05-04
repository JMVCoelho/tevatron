import logging
import os
import sys

from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from contextlib import nullcontext

from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.retriever.dataset import ValidDatasetPreprocessed
from tevatron.retriever.collator import TrainCollatorPreprocessed
from tevatron.retriever.modeling import DenseModel
from tevatron.retriever.trainer import TevatronTrainer as Trainer
from tevatron.retriever.gc_trainer import GradCacheTrainer as GCTrainer

from torch.utils.data import DataLoader

import torch
import numpy as np
from tqdm import tqdm

import pickle

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = 'right'
    
    model = DenseModel.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
    )

    model.to("cuda")

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    valid_dataset = ValidDatasetPreprocessed(data_args)
    collator = TrainCollatorPreprocessed(data_args, tokenizer)
    valid_dataset.tokenizer = tokenizer

    dtype = None
    if training_args.fp16:
        print("Set encoding precision: fp16")
        dtype = torch.float16
    elif training_args.bf16:
        print("Set encoding precision: bf16")
        dtype = torch.bfloat16

    BS=20

    total_gradients = []
    dataloader = DataLoader(valid_dataset, batch_size=BS, collate_fn=collator)

    losses = []
    with torch.cuda.amp.autocast(dtype=dtype) if dtype is not None else nullcontext():
        for batch in tqdm(dataloader):

            q, d = batch

            q = {k:v.to("cuda") for k, v in q.items()}
            d = {k:v.to("cuda") for k, v in d.items()}

            loss = model(q, d).loss

            losses.append(loss)

            loss.backward()

            gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.detach().view(-1))
            
            gradient_vector = torch.cat(gradients).cpu().numpy()
            total_gradients.append(gradient_vector)
            model.zero_grad()

    print(sum(losses)/len(losses))
    exit()      
    average_gradient = np.mean(total_gradients, axis=0)

    norm = np.linalg.norm(average_gradient)

    print("Norm of the average gradient vector:", norm)

    with open(f"{data_args.save_gradient_path}/validation_grad_bs{BS}.pkl", 'wb') as h:
        pickle.dump(average_gradient, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
