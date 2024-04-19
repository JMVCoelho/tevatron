import logging
from typing import List, Tuple
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from tevatron.retriever.arguments import DataArguments


logger = logging.getLogger(__name__)


@dataclass
class TrainCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: List[Tuple[str, List[str]]]):
        """
        Collate function for training.
        :param features: list of (query, passages) tuples
        :return: tokenized query_ids, passage_ids
        """
        all_queries = [f[0] for f in features]
        all_passages = []
        for f in features:
            all_passages.extend(f[1])
        q_collated = self.tokenizer(
            all_queries,
            padding=False, 
            truncation=True,
            max_length=self.data_args.query_max_len-1 if self.data_args.append_eos_token else self.data_args.query_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        d_collated = self.tokenizer(
            all_passages,
            padding=False, 
            truncation=True,
            max_length=self.data_args.passage_max_len-1 if self.data_args.append_eos_token else self.data_args.passage_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        if self.data_args.append_eos_token:
            q_collated['input_ids'] = [q + [self.tokenizer.eos_token_id] for q in q_collated['input_ids']]
            d_collated['input_ids'] = [d + [self.tokenizer.eos_token_id] for d in d_collated['input_ids']]
            
        q_collated = self.tokenizer.pad(
            q_collated,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        d_collated = self.tokenizer.pad(
            d_collated,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return q_collated, d_collated
    
@dataclass
class TrainCollatorPreprocessed:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer
    landmarks: bool = False

    def __call__(self, features):
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        if self.data_args.append_eos_token:
            for q in qq:
                q['input_ids'] = q['input_ids'] + [self.tokenizer.eos_token_id]
                
            for d in dd:
                d['input_ids'] = d['input_ids'] + [self.tokenizer.eos_token_id]
                if self.landmarks:
                    print("SHOULD NOT BE PRITTING THIS!!!!!!!!")
                    num_tokens = len(d['input_ids'])
                    landmarks = [(num_tokens - ((num_tokens//4) * i)) - 1 for i in range(4)]
                    for ldmk in landmarks:
                        d['input_ids'][ldmk] = self.tokenizer.eos_token_id
                
        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.data_args.query_max_len,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_tensors="pt",
        )

        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.data_args.passage_max_len,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        return q_collated, d_collated


@dataclass
class EncodeCollator:
    data_args: DataArguments
    tokenizer: PreTrainedTokenizer
    landmarks: bool = False

    def __call__(self, features: List[Tuple[str, str]]):
        """
        Collate function for encoding.
        :param features: list of (id, text) tuples
        """
        text_ids = [x[0] for x in features]
        texts = [x[1] for x in features]
        max_length = self.data_args.query_max_len if self.data_args.encode_is_query else self.data_args.passage_max_len
        collated_texts = self.tokenizer(
            texts,
            padding=False, 
            truncation=True,
            max_length=max_length-1 if self.data_args.append_eos_token else max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        if self.data_args.append_eos_token:
            collated_texts['input_ids'] = [x + [self.tokenizer.eos_token_id] for x in collated_texts['input_ids']]
            
            if self.landmarks and not self.data_args.encode_is_query:
                print("SHOULD NOT !!!!")
                for i in range(len(collated_texts['input_ids'])):
                    num_tokens = len(collated_texts['input_ids'][i])
                    landmarks = [(num_tokens - ((num_tokens//4) * i)) - 1 for i in range(4)]
                    for ldmk in landmarks:
                        collated_texts['input_ids'][i][ldmk] = self.tokenizer.eos_token_id

                
        collated_texts = self.tokenizer.pad(
            collated_texts,
            padding=True, 
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return text_ids, collated_texts