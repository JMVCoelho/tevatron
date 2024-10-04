from abc import ABC, abstractmethod
from tqdm import tqdm
#from sklearn.cluster import KMeans
from contextlib import nullcontext
import math
import json
from collections import defaultdict
import torch.nn.functional as F
import copy


#from opacus.grad_sample import GradSampleModule

import numpy as np
import torch
import random
import glob
import pickle
import os
from torch import nn, optim, Tensor


from tevatron.retriever.dataset import TrainDatasetPreprocessed
from tevatron.retriever.collator import TrainCollatorPreprocessed
from tevatron.retriever.modeling import DenseModel, DenseModelLESS

import operator

from transformers import AutoTokenizer

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
class HardNegatives(ABC):

    def __init__(self, qrels_path, run_path, embeddings_path = None):

        self.load_qrels(qrels_path)
        self.load_run(run_path)
        if embeddings_path is not None:
            self.load_embeddings(embeddings_path)
    
    @abstractmethod
    def get_sample_loss(self, query: int, n: int) -> list[str]:
        # each subclass implements the way to sample the negatives for the query
        pass


    def load_qrels(self, path):
        logger.info(f"Loading qrels: {path}")
        self.qid2pos = {}

        with open(path, 'r') as h:

            for line in h:
                qid, q0, did, rel = line.strip().split("\t")

                if qid not in self.qid2pos:
                    self.qid2pos[qid] = []
                
                self.qid2pos[qid].append(did)

    def load_run(self, path):
        logger.info(f"Loading run: {path}")
        self.qid2negs = {}

        with open(path, 'r') as h:

            for line in h:
                qid, did, score = line.strip().split()

                if qid not in self.qid2negs:
                    self.qid2negs[qid] = []

                if did not in self.qid2pos[qid]:
                    self.qid2negs[qid].append(did) 

    
    def sample_hard_negatives(self, n, outpath):
        logger.info(f"Sampling started.")
        with open(outpath, 'w') as h:
            for query in tqdm(self.qid2negs):
                
                chosen_negatives = self.choose_negatives(query, n)
                
                line_to_write = f"{query}\t{','.join(chosen_negatives)}\n"
                h.write(line_to_write)

class MATESQueryAttribution(HardNegatives):
    def __init__(self, qrels_path, run_path, valid_path, model_args, data_args, training_args):
        
        super().__init__(qrels_path, run_path, embeddings_path=None) # no need to store embeddings

        self.corpus_path = "/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.tsv" #TODO argument
        self.corpus_path_extra = "/data/group_data/cx_group/query_generation_data/synth_corpus.tsv" #TODO argument
        self.queries_path = "/data/group_data/cx_group/query_generation_data/gen4.query.tsv" #TODO argument
        self.valid_samples_path = valid_path 
        print(self.valid_samples_path)

        self.valid_group_initial_loss = {}

        self.parse_queries()
        self.parse_corpus()
        print(len(self.did2text))
        self.parse_extra_corpus()
        print(len(self.did2text))

        self.parse_jsonl(group_size=10)

        self.data_args = data_args

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        self.tokenizer.padding_side = 'right'


        self.model = DenseModelLESS.build(
            model_args,
            training_args,
            cache_dir=model_args.cache_dir,
            )
        self.model.to("cuda")

        self.model_copy = copy.deepcopy(self.model)
    
    def set_seed(self, seed):
        random.seed(seed)
    
    def parse_jsonl(self, group_size):
        self.valid_samples = []
        with open(self.valid_samples_path, 'r') as file:
            for line in file:
                # Strip newline character and parse JSON
                json_data = json.loads(line.strip())
                self.valid_samples.append(json_data)

        self.valid_samples = [self.valid_samples[i:i + group_size] for i in range(0, len(self.valid_samples), group_size)]

        print(len(self.valid_samples))
        
    def parse_corpus(self):
        logger.info(f"Loading corpus text")
        self.did2text = {}

        with open(self.corpus_path, 'r') as h:
            for line in h:
                did, title, text = line.strip().split("\t")
                self.did2text[did] = f' Title: {title.strip()} Text: {text.strip()}'.strip()

    def parse_extra_corpus(self):
        logger.info(f"Loading EXTRA corpus text")
        self.did2text_extra = {}

        with open(self.corpus_path_extra, 'r') as h:
            for line in h:
                try:
                    did, text = line.strip().split("\t")
                except Exception as e:
                    did = line.strip()
                    text = "ERROR"
                    print("err...")
                self.did2text_extra[did] = f'{text.strip()}'.strip()

        self.did2text.update(self.did2text_extra)

    def parse_queries(self):
        logger.info(f"Loading queries text")
        self.qid2text = {}
        with open(self.queries_path, 'r') as h:
            for line in h:
                qid, text = line.strip().split("\t")
                self.qid2text[qid] = text

    def tokenize(self, q, d):

            q_collated = self.tokenizer(
                q,
                padding=False, 
                truncation=True,
                max_length=self.data_args.query_max_len-1 if self.data_args.append_eos_token else self.data_args.query_max_len,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=True,
            )
            d_collated = self.tokenizer(
                d,
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

    def get_valid_loss(self, q, d, qv, dv):

        # if valid_instances_group in self.valid_group_initial_loss:
        #     initial = self.valid_group_initial_loss[valid_instances_group]

        # else:
        #     with torch.no_grad():
        #         with torch.cuda.amp.autocast(dtype=torch.bfloat16): #HACK hardcoded to bf16
        #             qv = {k:v.to("cuda") for k, v in qv.items()}
        #             dv = {k:v.to("cuda") for k, v in dv.items()}

        #             loss = self.model(qv, dv).loss

        #         initial = loss.item()
        #         self.valid_group_initial_loss[valid_instances_group] = initial
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16): #HACK hardcoded to bf16
            #TRAIN
            q = {k:v.to("cuda") for k, v in q.items()}
            d = {k:v.to("cuda") for k, v in d.items()}

            loss = self.model(q, d).loss

            loss.backward(gradient=torch.ones_like(loss))

            gradients = []
            for param in self.model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.detach().view(-1))
            
            gradient_vector = torch.cat(gradients)
            gradient_norm = torch.norm(gradient_vector).item()
            
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16): #HACK hardcoded to bf16
                #VALID
                qv = {k:v.to("cuda") for k, v in qv.items()}
                dv = {k:v.to("cuda") for k, v in dv.items()}

                loss = self.model(qv, dv).loss

        final = loss.item()

        self.model.load_state_dict(self.model_copy.state_dict())

        #delta = initial - final # if negative its bad!

        return final, gradient_norm
    
    def get_sample_loss(self, query: int, n: int) -> list[str]:
        positive_id = self.qid2pos[query][0]
        negative_ids = self.qid2negs[query]

        #print(len(negative_ids))
        
        sample = random.sample(negative_ids, n)

        # random_docs = random.sample(list(self.did2text.keys()), 9)
        # sample = sample + random_docs

        valid_instances_group = random.randint(0, len(self.valid_samples)-1)
        valid_instances = self.valid_samples[valid_instances_group]
        
        v_queries = []
        v_documents = []
        for valid_instance in valid_instances:
            v_queries.append(self.tokenizer.decode(valid_instance["query"]))
            v_documents.append(self.tokenizer.decode(valid_instance["positives"][0]))
            v_documents += self.tokenizer.batch_decode(valid_instance["negatives"][:9])

        qv, dv =  self.tokenize(v_queries, v_documents)

        positive_text = self.did2text[positive_id]
        negative_texts = [self.did2text[did] for did in sample]

        query_text = self.qid2text[query]

        queries = [query_text]
        documents = [positive_text] + negative_texts

        q, d = self.tokenize(queries, documents)

        valid_loss, gradient_norm = self.get_valid_loss(q, d, qv, dv)

        return sample, valid_loss, valid_instances_group, gradient_norm
    
    def sample_hard_negatives(self, n, outpath):
        logger.info(f"Sampling started.")
        
        with open(outpath+"_queries_with_valid_loss", 'w') as h1:
                for query in tqdm(self.qid2negs):
                    
                    sample, loss, valid_instances_group, gradient_norm = self.get_sample_loss(query, n)
                    
                    h1.write(f"{query}\t{','.join(sample)}\t{valid_instances_group}\t{loss}\t{gradient_norm}\n")


    def get_initial_valid_loss(self, outpath):
        logger.info(f"Computing initial valid losses")

        valid_losses = []
        for valid_instances in tqdm(self.valid_samples):
            v_queries = []
            v_documents = []
            for valid_instance in valid_instances:
                v_queries.append(self.tokenizer.decode(valid_instance["query"]))
                v_documents.append(self.tokenizer.decode(valid_instance["positives"][0]))
                v_documents += self.tokenizer.batch_decode(valid_instance["negatives"][:9])

            qv, dv =  self.tokenize(v_queries, v_documents)
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16): #HACK hardcoded to bf16
                    #VALID
                    qv = {k:v.to("cuda") for k, v in qv.items()}
                    dv = {k:v.to("cuda") for k, v in dv.items()}

                    loss = self.model(qv, dv).loss
        
                    valid_losses.append(loss)

        with open(outpath+"_initial_valid_loss", 'w') as h1:         
            for idx, num in enumerate(valid_losses):
                h1.write(f"{idx}\t{num}\n")



class MATESQueryAttributionPAIRS(HardNegatives):
    def __init__(self, qrels_path, run_path, valid_path, model_args, data_args, training_args):
        
        super().__init__(qrels_path, run_path, embeddings_path=None) # no need to store embeddings

        self.corpus_path = "/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.tsv" #TODO argument
        self.queries_path = "/data/user_data/jmcoelho/datasets/marco/documents/gen6.query.tsv" #TODO argument
        self.valid_samples_path = valid_path 
        print(self.valid_samples_path)

        self.valid_group_initial_loss = {}

        self.parse_queries()
        self.parse_corpus()
        self.parse_jsonl(group_size=50)

        self.data_args = data_args

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        self.tokenizer.padding_side = 'right'


        self.model = DenseModelLESS.build(
            model_args,
            training_args,
            cache_dir=model_args.cache_dir,
            )
        self.model.to("cuda")

        self.model_copy = copy.deepcopy(self.model)
    
    def set_seed(self, seed):
        random.seed(seed)
    
    def parse_jsonl(self, group_size):
        self.valid_samples = []
        with open(self.valid_samples_path, 'r') as file:
            for line in file:
                # Strip newline character and parse JSON
                json_data = json.loads(line.strip())
                self.valid_samples.append(json_data)

        self.valid_samples = [self.valid_samples[i:i + group_size] for i in range(0, len(self.valid_samples), group_size)]

        print(len(self.valid_samples))
        
    def parse_corpus(self):
        logger.info(f"Loading corpus text")
        self.did2text = {}

        with open(self.corpus_path, 'r') as h:
            for line in h:
                did, title, text = line.strip().split("\t")
                self.did2text[did] = f' Title: {title.strip()} Text: {text.strip()}'.strip()

    def parse_queries(self):
        logger.info(f"Loading queries text")
        self.qid2text = {}
        with open(self.queries_path, 'r') as h:
            for line in h:
                qid, text = line.strip().split("\t")
                self.qid2text[qid] = text

    def tokenize(self, q, d):

            q_collated = self.tokenizer(
                q,
                padding=False, 
                truncation=True,
                max_length=self.data_args.query_max_len-1 if self.data_args.append_eos_token else self.data_args.query_max_len,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=True,
            )
            d_collated = self.tokenizer(
                d,
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

    def get_valid_loss(self, q, d, qv, dv):

        # if valid_instances_group in self.valid_group_initial_loss:
        #     initial = self.valid_group_initial_loss[valid_instances_group]

        # else:
        #     with torch.no_grad():
        #         with torch.cuda.amp.autocast(dtype=torch.bfloat16): #HACK hardcoded to bf16
        #             qv = {k:v.to("cuda") for k, v in qv.items()}
        #             dv = {k:v.to("cuda") for k, v in dv.items()}

        #             loss = self.model(qv, dv).loss

        #         initial = loss.item()
        #         self.valid_group_initial_loss[valid_instances_group] = initial
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16): #HACK hardcoded to bf16
            #TRAIN
            q = {k:v.to("cuda") for k, v in q.items()}
            d = {k:v.to("cuda") for k, v in d.items()}

            loss = self.model(q, d).loss

            loss.backward(gradient=torch.ones_like(loss))

            gradients = []
            for param in self.model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.detach().view(-1))
            
            gradient_vector = torch.cat(gradients)
            gradient_norm = torch.norm(gradient_vector).item()
            
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16): #HACK hardcoded to bf16
                #VALID
                qv = {k:v.to("cuda") for k, v in qv.items()}
                dv = {k:v.to("cuda") for k, v in dv.items()}

                loss = self.model(qv, dv).loss

        final = loss.item()

        self.model.load_state_dict(self.model_copy.state_dict())

        #delta = initial - final # if negative its bad!

        return final, gradient_norm
    
    def get_sample_loss(self, query: int, n: int, valid_instances_group: int) -> list[str]:
        positive_id = self.qid2pos[query][0]
        negative_ids = self.qid2negs[query]

        #print(len(negative_ids))
        
        sample = random.sample(negative_ids, n)

        # random_docs = random.sample(list(self.did2text.keys()), 9)
        # sample = sample + random_docs

        #valid_instances_group = random.randint(0, len(self.valid_samples)-1)
        valid_instances = self.valid_samples[valid_instances_group]
        
        v_queries = []
        v_documents = []
        for valid_instance in valid_instances:
            v_queries.append(self.tokenizer.decode(valid_instance["query"]))
            v_documents.append(self.tokenizer.decode(valid_instance["positives"][0]))
            v_documents += self.tokenizer.batch_decode(valid_instance["negatives"][:9])

        qv, dv =  self.tokenize(v_queries, v_documents)

        positive_text = self.did2text[positive_id]
        negative_texts = [self.did2text[did] for did in sample]

        query_text = self.qid2text[query]

        queries = [query_text]
        documents = [positive_text] + negative_texts

        q, d = self.tokenize(queries, documents)

        valid_loss, gradient_norm = self.get_valid_loss(q, d, qv, dv)

        return sample, valid_loss, valid_instances_group, gradient_norm
    
    def sample_hard_negatives(self, n, outpath):
        logger.info(f"Sampling started.")
        
        keys = list(self.qid2negs.keys())

        # with open(outpath+"_queries_with_valid_loss_higher", 'w') as higher_loss, \
        #     open(outpath+"_queries_with_valid_loss_lower", 'w') as lower_loss:
                # assert len(keys[::2]) == len(keys[1::2])
                # for q1, q2 in tqdm(zip(keys[::2], keys[1::2]), total=len(keys[::2])):
                    
                #     valid_instances_group = random.randint(0, len(self.valid_samples)-1)

                #     sample1, loss1, valid_instances_group, norm1 = self.get_sample_loss(q1, n, valid_instances_group)
                #     sample2, loss2, valid_instances_group, norm2 = self.get_sample_loss(q2, n, valid_instances_group)

                #     if loss1 < loss2:
                #         lower_loss.write(f"{q1}\t{','.join(sample1)}\t{valid_instances_group}\t{loss1}\t{norm1}\n")
                #         higher_loss.write(f"{q2}\t{','.join(sample2)}\t{valid_instances_group}\t{loss2}\t{norm2}\n")
                #     else:
                #         higher_loss.write(f"{q1}\t{','.join(sample1)}\t{valid_instances_group}\t{loss1}\t{norm1}\n")
                #         lower_loss.write(f"{q2}\t{','.join(sample2)}\t{valid_instances_group}\t{loss2}\t{norm2}\n")

        with open(outpath+"_queries_with_valid_loss", 'w') as out:
                assert len(keys[::4]) == len(keys[1::4]) == len(keys[2::4]) == len(keys[3::4])

                for q1, q2, q3, q4 in tqdm(zip(keys[::4], keys[1::4], keys[2::4], keys[3::4]), total=len(keys[::4])):
                    
                    valid_instances_group = random.randint(0, len(self.valid_samples)-1)

                    sample1, loss1, valid_instances_group, norm1 = self.get_sample_loss(q1, n, valid_instances_group)
                    sample2, loss2, valid_instances_group, norm2 = self.get_sample_loss(q2, n, valid_instances_group)
                    sample3, loss3, valid_instances_group, norm3 = self.get_sample_loss(q3, n, valid_instances_group)
                    sample4, loss4, valid_instances_group, norm4 = self.get_sample_loss(q4, n, valid_instances_group)

                    out.write(f"{q1}\t{','.join(sample1)}\t{valid_instances_group}\t{loss1}\t{norm1}\n")
                    out.write(f"{q2}\t{','.join(sample2)}\t{valid_instances_group}\t{loss2}\t{norm2}\n")
                    out.write(f"{q3}\t{','.join(sample3)}\t{valid_instances_group}\t{loss3}\t{norm3}\n")
                    out.write(f"{q4}\t{','.join(sample4)}\t{valid_instances_group}\t{loss4}\t{norm4}\n")


    def get_initial_valid_loss(self, outpath):
        logger.info(f"Computing initial valid losses")

        valid_losses = []
        for valid_instances in tqdm(self.valid_samples):
            v_queries = []
            v_documents = []
            for valid_instance in valid_instances:
                v_queries.append(self.tokenizer.decode(valid_instance["query"]))
                v_documents.append(self.tokenizer.decode(valid_instance["positives"][0]))
                v_documents += self.tokenizer.batch_decode(valid_instance["negatives"][:9])

            qv, dv =  self.tokenize(v_queries, v_documents)
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16): #HACK hardcoded to bf16
                    #VALID
                    qv = {k:v.to("cuda") for k, v in qv.items()}
                    dv = {k:v.to("cuda") for k, v in dv.items()}

                    loss = self.model(qv, dv).loss
        
                    valid_losses.append(loss)

        with open(outpath+"_initial_valid_loss", 'w') as h1:         
            for idx, num in enumerate(valid_losses):
                h1.write(f"{idx}\t{num}\n")


class LESSQueryAttribution(HardNegatives):
    def __init__(self, qrels_path, run_path, valid_path, model_args, data_args, training_args, embeddings_path):
        
        super().__init__(qrels_path, run_path, embeddings_path=None) # no need to store embeddings

        self.corpus_path = "/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.tsv" #TODO argument
        self.queries_path = "/data/user_data/jmcoelho/datasets/marco/documents/gen10.query.tsv" #TODO argument

        self.parse_queries()
        self.parse_corpus()

        self.data_args = data_args

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        self.tokenizer.padding_side = 'right'

        self.validation_gradient = torch.tensor(self.get_validation_gradient(embeddings_path)).cuda()

        filename="/data/user_data/jmcoelho/models/fine-tuned/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision-1gpu/checkpoint-100/global_step100/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
        optimizer_states = torch.load(filename, map_location='cpu')

        self.avg = optimizer_states['optimizer_state_dict']['optimizer_state_dict']['state'][0]['exp_avg'].cuda()
        self.avg_sq = optimizer_states['optimizer_state_dict']['optimizer_state_dict']['state'][0]['exp_avg_sq'].cuda()
        
        del optimizer_states

        self.model = DenseModelLESS.build(
            model_args,
            training_args,
            cache_dir=model_args.cache_dir,
            )

        self.model.to("cuda")

        self.model_copy = copy.deepcopy(self.model)
    
    def set_seed(self, seed):
        random.seed(seed)
        
    def parse_corpus(self):
        logger.info(f"Loading corpus text")
        self.did2text = {}

        with open(self.corpus_path, 'r') as h:
            for line in h:
                did, title, text = line.strip().split("\t")
                self.did2text[did] = f' Title: {title.strip()} Text: {text.strip()}'.strip()

    def parse_queries(self):
        logger.info(f"Loading queries text")
        self.qid2text = {}
        with open(self.queries_path, 'r') as h:
            for line in h:
                qid, text = line.strip().split("\t")
                self.qid2text[qid] = text

    def get_validation_gradient(self, embedding_path):
        with open(f"{embedding_path}/normalized_average_gradients.pkl", 'rb') as h:
            grad = pickle.load(h)
        
        print(f"Loaded valid gradient")
        return grad

    def tokenize(self, q, d):

            q_collated = self.tokenizer(
                q,
                padding=False, 
                truncation=True,
                max_length=self.data_args.query_max_len-1 if self.data_args.append_eos_token else self.data_args.query_max_len,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=True,
            )
            d_collated = self.tokenizer(
                d,
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

    def get_grad(self, q, d):

        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-08
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16): #HACK hardcoded to bf16

            q = {k:v.view(1, -1).to("cuda") for k, v in q.items()}
            d = {k:v.to("cuda") for k, v in d.items()}

            loss = self.model(q, d).loss

            loss.backward()

            gradients = []
            for param in self.model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.detach().view(-1))
            

            gradient_vector = torch.cat(gradients) 
            updated_avg = beta1 * self.avg + (1 - beta1) * gradient_vector
            updated_avg_sq = beta2 * self.avg_sq + (1 - beta2) * gradient_vector ** 2
            gradient_vector = updated_avg / torch.sqrt(updated_avg_sq + eps)

            gradient_vector = F.normalize(gradient_vector, p=2, dim=0)

            dot_product = torch.dot(gradient_vector, self.validation_gradient)
            
            self.model.zero_grad()

        return dot_product
    
    def get_sample_loss(self, query: int, n: int) -> list[str]:
        positive_id = self.qid2pos[query][0]
        negative_ids = self.qid2negs[query]
        sample = random.sample(negative_ids, n) # sample the negatives
        

        positive_text = self.did2text[positive_id]
        negative_texts = [self.did2text[did] for did in sample]

        query_text = self.qid2text[query]

        queries = [query_text]
        documents = [positive_text] + negative_texts

        q, d = self.tokenize(queries, documents)

        dot_prod = self.get_grad(q, d)
        #grad = self.get_grad(q, d)
        #dot_prod = np.dot(grad, self.validation_gradient)

        return sample, dot_prod
    
    def sample_hard_negatives(self, n, outpath):
        logger.info(f"Sampling started.")
        
        with open(outpath+"_queries_with_valid_grad_dot_p", 'w') as h1:
                for query in tqdm(self.qid2negs):
                    
                    sample, dot_product = self.get_sample_loss(query, n)
                    
                    h1.write(f"{query}\t{','.join(sample)}\t{dot_product}\n")





class GranNormQueryAttribution(HardNegatives):
    def __init__(self, qrels_path, run_path, valid_path, model_args, data_args, training_args):
        
        super().__init__(qrels_path, run_path, embeddings_path=None) # no need to store embeddings

        self.corpus_path = "/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.tsv" #TODO argument
        self.queries_path = "/data/user_data/jmcoelho/datasets/marco/documents/gen.query.tsv" #TODO argument

        self.parse_queries()
        self.parse_corpus()

        self.data_args = data_args

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        self.tokenizer.padding_side = 'right'

        self.model = DenseModelLESS.build(
            model_args,
            training_args,
            cache_dir=model_args.cache_dir,
            )
        self.model.to("cuda")

        self.model_copy = copy.deepcopy(self.model)
    
    def set_seed(self, seed):
        random.seed(seed)
        
    def parse_corpus(self):
        logger.info(f"Loading corpus text")
        self.did2text = {}

        with open(self.corpus_path, 'r') as h:
            for line in h:
                did, title, text = line.strip().split("\t")
                self.did2text[did] = f' Title: {title.strip()} Text: {text.strip()}'.strip()

    def parse_queries(self):
        logger.info(f"Loading queries text")
        self.qid2text = {}
        with open(self.queries_path, 'r') as h:
            for line in h:
                qid, text = line.strip().split("\t")
                self.qid2text[qid] = text

    def get_validation_gradient(self, embedding_path):
        with open(f"{embedding_path}/average_gradient.pkl", 'rb') as h:
            grad = pickle.load(h)
        
        print(f"Loaded valid gradient")
        return grad

    def tokenize(self, q, d):

            q_collated = self.tokenizer(
                q,
                padding=False, 
                truncation=True,
                max_length=self.data_args.query_max_len-1 if self.data_args.append_eos_token else self.data_args.query_max_len,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=True,
            )
            d_collated = self.tokenizer(
                d,
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

    def get_grad_norm(self, q, d):
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16): #HACK hardcoded to bf16

            q = {k:v.view(1, -1).to("cuda") for k, v in q.items()}
            d = {k:v.to("cuda") for k, v in d.items()}

            loss = self.model(q, d).loss

            loss.backward()

            gradients = []
            for param in self.model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.detach().view(-1))
            
            gradient_vector = torch.cat(gradients)
            gradient_norm = torch.norm(gradient_vector).item()
            self.model.zero_grad()

        return gradient_norm
    
    def get_sample_loss(self, query: int, n: int) -> list[str]:
        positive_id = self.qid2pos[query][0]
        negative_ids = self.qid2negs[query]
        sample = random.sample(negative_ids, n) # sample the negatives
        

        positive_text = self.did2text[positive_id]
        negative_texts = [self.did2text[did] for did in sample]

        query_text = self.qid2text[query]

        queries = [query_text]
        documents = [positive_text] + negative_texts

        q, d = self.tokenize(queries, documents)

        grad_norm = self.get_grad_norm(q, d)

        return sample, grad_norm
    
    def sample_hard_negatives(self, n, outpath):
        logger.info(f"Sampling started.")
        
        with open(outpath+"_queries_with_valid_grad_norm", 'w') as h1:
                for query in tqdm(self.qid2negs):
                    
                    sample, dot_product = self.get_sample_loss(query, n)
                    
                    h1.write(f"{query}\t{','.join(sample)}\t{dot_product}\n")


class GranVarianceQueryAttribution(HardNegatives):
    def __init__(self, qrels_path, run_path, valid_path, model_args, data_args, training_args):
        
        super().__init__(qrels_path, run_path, embeddings_path=None) # no need to store embeddings

        self.corpus_path = "/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.tsv" #TODO argument
        self.queries_path = "/data/user_data/jmcoelho/datasets/marco/documents/gen.query.tsv" #TODO argument

        self.parse_queries()
        self.parse_corpus()

        self.data_args = data_args

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        self.tokenizer.padding_side = 'right'

        self.model = DenseModelLESS.build(
            model_args,
            training_args,
            cache_dir=model_args.cache_dir,
            )
        self.model.to("cuda")

        self.model_copy = copy.deepcopy(self.model)
    
    def set_seed(self, seed):
        random.seed(seed)
        
    def parse_corpus(self):
        logger.info(f"Loading corpus text")
        self.did2text = {}

        with open(self.corpus_path, 'r') as h:
            for line in h:
                did, title, text = line.strip().split("\t")
                self.did2text[did] = f' Title: {title.strip()} Text: {text.strip()}'.strip()

    def parse_queries(self):
        logger.info(f"Loading queries text")
        self.qid2text = {}
        with open(self.queries_path, 'r') as h:
            for line in h:
                qid, text = line.strip().split("\t")
                self.qid2text[qid] = text

    def tokenize(self, q, d):

            q_collated = self.tokenizer(
                q,
                padding=False, 
                truncation=True,
                max_length=self.data_args.query_max_len-1 if self.data_args.append_eos_token else self.data_args.query_max_len,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=True,
            )
            d_collated = self.tokenizer(
                d,
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

    def get_grad_var(self, q, d):
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16): #HACK hardcoded to bf16

            q = {k:v.view(1, -1).to("cuda") for k, v in q.items()}
            d = {k:v.to("cuda") for k, v in d.items()}

            loss = self.model(q, d).loss

            loss.backward()

            gradients = []
            for param in self.model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.detach().view(-1))
            
            gradient_vector = torch.cat(gradients)
            gradient_var = torch.var(gradient_vector).item()
            self.model.zero_grad()

        return gradient_var
    
    def get_sample_loss(self, query: int, n: int) -> list[str]:
        positive_id = self.qid2pos[query][0]
        negative_ids = self.qid2negs[query]
        sample = random.sample(negative_ids, n) # sample the negatives
        

        positive_text = self.did2text[positive_id]
        negative_texts = [self.did2text[did] for did in sample]

        query_text = self.qid2text[query]

        queries = [query_text]
        documents = [positive_text] + negative_texts

        q, d = self.tokenize(queries, documents)

        grad_var = self.get_grad_var(q, d)

        return sample, grad_var
    
    def sample_hard_negatives(self, n, outpath):
        logger.info(f"Sampling started.")
        
        with open(outpath+"_queries_with_valid_grad_var", 'w') as h1:
                for query in tqdm(self.qid2negs):
                    
                    sample, dot_product = self.get_sample_loss(query, n)
                    
                    h1.write(f"{query}\t{','.join(sample)}\t{dot_product}\n")
