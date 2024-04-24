import random
from typing import List, Tuple

import numpy as np

from datasets import load_dataset
from torch.utils.data import Dataset


from tevatron.retriever.arguments import DataArguments

from transformers import BatchEncoding

import logging
logger = logging.getLogger(__name__)


def format_query(query: str, prefix: str = '') -> str:
    return f'{prefix} {query.strip()}'.strip()

def format_passage(text: str, title: str = '', prefix: str = '', add_markers: bool = False) -> str:
    if add_markers:
        return f'{prefix} Title: {title.strip()} Text: {text.strip()}'.strip()
    else:
        return f'{prefix} {title.strip()} {text.strip()}'.strip()

class DistilDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None, embeddings = None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer
        self.embeddings = embeddings

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        query = group['query']
        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        formated_query = format_query(query, self.data_args.query_prefix)
        formated_passages = []

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
        formated_passages.append(format_passage(pos_psg['text'], pos_psg['title'], self.data_args.passage_prefix))

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            formated_passages.append(format_passage(neg_psg['text'], neg_psg['title'], self.data_args.passage_prefix))


        query_id = group['query_id']
        negative_ids = [neg['docid'] for neg in negs]
        positive_ids = [pos_psg['docid']]

        query_embeddings = self.embeddings['query'][query_id]
        negative_embeddings = np.array([self.embeddings['passages'][neg_id] for neg_id in negative_ids])
        positive_embeddings = np.array([self.embeddings['passages'][pos_id] for pos_id in positive_ids])

        passage_embeddings = np.concatenate((positive_embeddings, negative_embeddings), axis=0)

        # negative_scores = np.dot(query_embeddings, negative_embeddings.T).tolist()
        # positive_scores = np.dot(query_embeddings, positive_embeddings.T).tolist()

        # scores = positive_scores + negative_scores

        # scores = [0.6679622, 0.52343357, 0.50806546, 0.57454115, 0.59866923, 0.547013,   0.5771849,
        #         0.565547,   0.5919082,  0.5800996,  0.55530787, 0.5427108, 0.49957478,
        #         0.52329457, 0.50641984, 0.59075034]
             
        #return formated_query, formated_passages, scores
        return formated_query, formated_passages, query_embeddings, passage_embeddings

class TrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        query = group['query']
        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        formated_query = format_query(query, self.data_args.query_prefix)
        formated_passages = []

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        
        formated_passages.append(format_passage(pos_psg['text'], pos_psg['title'], self.data_args.passage_prefix))

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            formated_passages.append(format_passage(neg_psg['text'], neg_psg['title'], self.data_args.passage_prefix))

        return formated_query, formated_passages
    
class TrainDatasetPreprocessed(Dataset):
    def __init__(self, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.trainer = trainer

    def create_one_example(self, text_encoding: List[int], is_query=False):
        max_length = self.data_args.query_max_len if is_query else self.data_args.passage_max_len

        if self.data_args.append_eos_token:
            max_length -= 1

        item = self.tokenizer.prepare_for_model(
            text_encoding,
            truncation='only_first',
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group['query']
        encoded_query = self.create_one_example(qry, is_query=True)

        encoded_passages = []
        group_positives = group['positives']
        group_negatives = group['negatives']

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        encoded_passages.append(self.create_one_example(pos_psg))

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            encoded_passages.append(self.create_one_example(neg_psg))

        return encoded_query, encoded_passages

class EncodeDataset(Dataset):

    def __init__(self, data_args: DataArguments):
        self.data_args = data_args
        self.encode_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, str]:
        text = self.encode_data[item]
        if self.data_args.encode_is_query:
            text_id = text['query_id']
            formated_text = format_query(text['query'], self.data_args.query_prefix)
        else:
            text_id = text['docid'] if 'docid' in text else text['text_id']
            formated_text = format_passage(text['text'], text['title'], self.data_args.passage_prefix, self.data_args.add_markers)
        
        return text_id, formated_text
