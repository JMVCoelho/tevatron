from abc import ABC, abstractmethod
from tqdm import tqdm
from sklearn.cluster import KMeans
from contextlib import nullcontext
import math
import json
from collections import defaultdict
import torch.nn.functional as F


from opacus.grad_sample import GradSampleModule

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
    def choose_negatives(self, query: int, n: int) -> list[str]:
        # each subclass implements the way to sample the negatives for the query
        pass


    def load_qrels(self, path):
        logger.info(f"Loading qrels: {path}")
        self.qid2pos = {}

        with open(path, 'r') as h:

            for line in h:
                qid, q0, did, rel = line.strip().split("\t")

                if qid not in self.qid2pos:
                    self.qid2pos[qid] = did


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


    def load_embeddings(self, path):
        logger.info(f"Loading embeddings: {path}")

        assert glob.glob(os.path.join(path, "query-train.pkl")) \
                and glob.glob(os.path.join(path, "corpus.*.pkl")), \
                "Path does not contain query-train.pkl or corpus.*.pkl files"

        self.all_embeddings = {'query':{}, 'passages':{}}

        for file_name in os.listdir(path):
            if file_name.startswith("corpus") and file_name.endswith(".pkl"):
                file_path = os.path.join(path, file_name)
                with open(file_path, "rb") as f:
                    corpus_data = pickle.load(f)
                    embeddings, ids = corpus_data
                    self.all_embeddings['passages'].update(zip(ids, embeddings))

        with open(f"{path}/query-train.pkl","rb") as f:
            query_data = pickle.load(f)
            embeddings, ids = query_data
            self.all_embeddings['query'].update(zip(ids, embeddings))

    
    def sample_hard_negatives(self, n, outpath):
        logger.info(f"Sampling started.")
        with open(outpath, 'w') as h:
            for query in tqdm(self.qid2negs):
                
                chosen_negatives = self.choose_negatives(query, n)
                
                line_to_write = f"{query}\t{','.join(chosen_negatives)}\n"
                h.write(line_to_write)


class SimANSHardNegatives(HardNegatives):

    def set_seed(self, seed):
        random.seed(seed)

    def load_run(self, path):
        logger.info(f"Loading run: {path}")
        self.qid2negs = {}

        with open(path, 'r') as h:

            for line in h:
                qid, did, score = line.strip().split()

                if qid not in self.qid2negs:
                    self.qid2negs[qid] = []

                if did not in self.qid2pos[qid]:
                    self.qid2negs[qid].append((did, score)) 

    def choose_negatives(self, query: int, n: int) -> list[str]:
        possible_negatives = self.qid2negs[query]
        return random.sample(possible_negatives, n)


class RandomHardNegatives(HardNegatives):

    def set_seed(self, seed):
        random.seed(seed)

    def choose_negatives(self, query: int, n: int) -> list[str]:
        possible_negatives = self.qid2negs[query]
        return random.sample(possible_negatives, n)

class InDiHardNegatives(HardNegatives):
    # k means gradient vectors, pick metoids.

    def batch_doc_independent_grad_embeddings(self, queries, pos, docs):
        queries.requires_grad_(False)
        pos.requires_grad_(False)

        neg_sim = queries @ docs.T
        pos_sim = (queries * pos).sum(axis=1).repeat(len(docs), 1).T

        # if negatives are very hard, random sample:
        if pos_sim.mean().item() < neg_sim.mean().item() + 0.01:
            return None
                
        loss_per_doc = -(1 / (1 + (neg_sim - pos_sim).exp())).log()
        loss_per_doc.sum().backward()

        return docs.grad

    def choose_negatives(self, query: int, n: int) -> list[str]:
        q_embed = torch.tensor(self.all_embeddings['query'][query]).view(1, -1)
        pos_embed = torch.tensor(self.all_embeddings['passages'][self.qid2pos[query]]).view(1, -1)
        neg_embeds = np.array([self.all_embeddings['passages'][negative] for negative in self.qid2negs[query]])
        neg_embeds = torch.tensor(neg_embeds, requires_grad=True)

        gradients = self.batch_doc_independent_grad_embeddings(q_embed, pos_embed, neg_embeds)

        if gradients is not None:
            data_np = gradients.numpy()

            kmeans = KMeans(n_clusters=n)
            kmeans.fit(data_np)

            centroids = kmeans.cluster_centers_
            centroid_indices = list(set([np.argmin(np.linalg.norm(data_np - centroid, axis=1)) for centroid in centroids]))
        
        else:
            centroid_indices = []

        if len(centroid_indices) != n: # kmeans converged to < n clusters. randomly sample remaining ones.
            while len(centroid_indices) != n:
                rand_index = random.randint(0, len(self.qid2negs[query])-1)
                centroid_indices.append(rand_index)
                centroid_indices = list(set(centroid_indices))

        chosen_negatives = []
        for index in list(set(centroid_indices)):
            negative_id = self.qid2negs[query][index]
            chosen_negatives.append(negative_id)

        return chosen_negatives
    



class LESSHardNegatives(HardNegatives):
    def __init__(self, qrels_path, run_path, embeddings_path, model_args, data_args, training_args):
        
        super().__init__(qrels_path, run_path, embeddings_path=None) # no need to store embeddings

        self.validation_gradients = self.get_validation_gradient(embeddings_path)

        self.corpus_path = "/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.tsv" #TODO argument
        self.queries_path = "/data/user_data/jmcoelho/datasets/marco/documents/train.query.filtered.txt" #TODO argument

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


        self.model = DenseModel.build(
            model_args,
            training_args,
            cache_dir=model_args.cache_dir,
            )

        self.model.to("cuda")

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

        
    def get_validation_gradient(self, embeddings_path):
        validation_gradients = []
        for file_name in os.listdir(embeddings_path):
            if file_name.endswith(".pkl"):
                with open(f"{embeddings_path}/{file_name}", 'rb') as h:
                    grad = pickle.load(h)
                    validation_gradients.append(grad)
        
        print(f"Loaded {len(validation_gradients)} valid gradients")
        return validation_gradients
        
    def tokenize(self, q, pos, neg):
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
                [pos, neg],
                padding=False, 
                truncation=True,
                max_length=self.data_args.passage_max_len-1 if self.data_args.append_eos_token else self.data_args.passage_max_len,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=True,
            )

            if self.data_args.append_eos_token:
                q_collated['input_ids'] += [self.tokenizer.eos_token_id]
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
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16): #HACK hardcoded to bf16

            q = {k:v.view(1, -1).to("cuda") for k, v in q.items()}
            d = {k:v.to("cuda") for k, v in d.items()}

            loss = self.model(q, d).loss

            loss.backward()

            gradients = []
            for param in self.model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.detach().view(-1))
            
            gradient_vector = torch.cat(gradients).cpu().numpy()
            self.model.zero_grad()

        return gradient_vector
    
    def choose_negatives(self, query: int, n: int) -> list[str]:
        positive_id = self.qid2pos[query]
        negative_ids = self.qid2negs[query]

        positive_text = self.did2text[positive_id]
        negative_texts = [self.did2text[did] for did in negative_ids]
        query_text = self.qid2text[query]

        dot_prods = []

        for neg in negative_texts:
            q, d = self.tokenize(query_text, positive_text, neg)
            grad = self.get_grad(q, d)

            dot_prods.append(np.dot(grad, random.choice(self.validation_gradients)))
        
        _, top_indices = torch.topk(torch.tensor(dot_prods), k=n)

        chosen_negatives = []
        for index in top_indices.tolist():
            negative_id = negative_ids[index]
            chosen_negatives.append(negative_id)


        return chosen_negatives



class LESSHardNegativesOpacus(HardNegatives):
    def __init__(self, qrels_path, run_path, embeddings_path, model_args, data_args, training_args):
        
        super().__init__(qrels_path, run_path, embeddings_path=None) # no need to store embeddings

        self.validation_gradients = self.get_validation_gradient(embeddings_path)

        self.corpus_path = "/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.tsv" #TODO argument
        self.queries_path = "/data/user_data/jmcoelho/datasets/marco/documents/train.query.filtered.txt" #TODO argument

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
        
        self.model = GradSampleModule(self.model)
        self.model.train() #hmmmm

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

        
    def get_validation_gradient(self, embeddings_path):
        validation_gradients = []
        for file_name in os.listdir(embeddings_path):
            if file_name.endswith(".pkl"):
                with open(f"{embeddings_path}/{file_name}", 'rb') as h:
                    grad = pickle.load(h)
                    validation_gradients.append(grad)
        
        print(f"Loaded {len(validation_gradients)} valid gradients")
        return validation_gradients
        
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

    def get_grad(self, q, d, vector):
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16): #HACK hardcoded to bf16
            vector = torch.tensor(vector).to("cuda")

            q = {k:v.to("cuda") for k, v in q.items()}
            d = {k:v.to("cuda") for k, v in d.items()}

            loss = self.model(q, d).loss

            loss.backward(gradient=torch.ones_like(loss))

            rolling_dot_prods = None # more memory efficient that shaping to [batchsize, n_params]
            for param in self.model.parameters(): # Opacus is computing for all documents?? ermmmmmmm... model(**docs), this seems to be their samples
                if param.grad is not None:
                    current_vector = param.grad_sample.detach().view(param.grad_sample.size(0), -1)[1:,:]
                    dimensions = current_vector.size(-1)
                    current_dot = torch.matmul(current_vector, vector[:dimensions])
                    vector = vector[dimensions:]
                    if rolling_dot_prods is None:
                        rolling_dot_prods = current_dot
                    else:
                        rolling_dot_prods += current_dot

        self.model.zero_grad() 

        return rolling_dot_prods
    
    def choose_negatives(self, query: int, n: int) -> list[str]:
        positive_id = self.qid2pos[query]
        negative_ids = self.qid2negs[query][:50]

        positive_text = self.did2text[positive_id]
        negative_texts = [self.did2text[did] for did in negative_ids]
        query_text = self.qid2text[query]

        queries = [query_text]
        documents = [positive_text]
        for neg in negative_texts:
            documents.append(neg)

        q, d = self.tokenize(queries, documents)

        dot_prods = self.get_grad(q, d, random.choice(self.validation_gradients))

        _, top_indices = torch.topk(dot_prods, k=n)

        TEMPERATURE = 10

        probabilities = torch.nn.functional.softmax(dot_prods / TEMPERATURE, dim=0).detach().float().cpu().numpy()
        probabilities /= probabilities.sum() # make sure sums to 1... numpy cant handle 1.000000001
        
        chosen_negatives_topk = []

        for index in top_indices.tolist():
            self.counter_topk[index] += 1
            negative_id = negative_ids[index]
            chosen_negatives_topk.append(negative_id)

        try:
            chosen_negatives_sample = np.random.choice(negative_ids, size=n, replace=False, p=probabilities)
        except Exception as e:
            print("sampling randomly")
            chosen_negatives_sample = np.random.choice(negative_ids, size=n, replace=False)

        for neg in chosen_negatives_sample:
            self.counter_sample[negative_ids.index(neg)] += 1

        return [chosen_negatives_sample, chosen_negatives_topk]
    
    def sample_hard_negatives(self, n, outpath):
        self.counter_topk = defaultdict(int)
        self.counter_sample = defaultdict(int)
        logger.info(f"Sampling started.")
        with open(outpath+"_topk", 'w') as h1, \
            open(outpath+"_sample", 'w') as h2:

                for query in tqdm(self.qid2negs):
                    
                    chosen_negatives_sample, chosen_negatives_topk = self.choose_negatives(query, n)
                    
                    h1.write(f"{query}\t{','.join(chosen_negatives_topk)}\n") 
                    if chosen_negatives_sample is not None:
                        h2.write(f"{query}\t{','.join(chosen_negatives_sample)}\n")
        
        print(self.counter_topk)
        print(self.counter_sample)
                    



##############################
##############################
##############################

        
class LESSHardNegativesQueryLevelOpacus(HardNegatives):
    def __init__(self, qrels_path, run_path, embeddings_path, model_args, data_args, training_args):
        
        super().__init__(qrels_path, run_path, embeddings_path=None) # no need to store embeddings

        self.validation_gradients = self.get_validation_gradient(embeddings_path)

        self.corpus_path = "/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.tsv" #TODO argument
        self.queries_path = "/data/user_data/jmcoelho/datasets/marco/documents/train.query.filtered.txt" #TODO argument

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

        
    def get_validation_gradient(self, embeddings_path):
        validation_gradients = []
        for file_name in os.listdir(embeddings_path):
            if file_name.endswith(".pkl"):
                with open(f"{embeddings_path}/{file_name}", 'rb') as h:
                    grad = pickle.load(h)
                    validation_gradients.append(grad)
        
        print(f"Loaded {len(validation_gradients)} valid gradients")
        return validation_gradients
        
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

    def get_grad(self, q, d, vector):
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16): #HACK hardcoded to bf16
            #vector = torch.tensor(vector).to("cuda")

            q = {k:v.to("cuda") for k, v in q.items()}
            d = {k:v.to("cuda") for k, v in d.items()}

            loss = self.model(q, d).loss

            loss.backward(gradient=torch.ones_like(loss))

            gradients = []
            for param in self.model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.detach().view(-1))
            
            gradient_vector = torch.cat(gradients).cpu().numpy()
            self.model.zero_grad()

        dot_prod = np.dot(gradient_vector, vector)

        return dot_prod
    
    def choose_negatives(self, query: int, n: int) -> list[str]:
        positive_id = self.qid2pos[query]
        negative_ids = self.qid2negs[query]

        selected_negatives = random.sample(negative_ids, n)

        positive_text = self.did2text[positive_id]
        negative_texts = [self.did2text[did] for did in selected_negatives]


        query_text = self.qid2text[query]

        queries = [query_text]
        documents = [positive_text] + negative_texts

        q, d = self.tokenize(queries, documents)

        query_weight = self.get_grad(q, d, random.choice(self.validation_gradients))

        chosen_negatives_topk = (selected_negatives, query_weight)
        chosen_negatives_sample = None

        return [chosen_negatives_sample, chosen_negatives_topk]
    
    def sample_hard_negatives(self, n, outpath):
        self.counter = defaultdict(int)
        logger.info(f"Sampling started.")
        with open(outpath+"_topk", 'w') as h1, \
            open(outpath+"_sample", 'w') as h2:
                for query in tqdm(self.qid2negs):
                    
                    chosen_negatives_sample, chosen_negatives_topk = self.choose_negatives(query, n)
                    
                    h1.write(f"{query}\t{','.join(chosen_negatives_topk[0])}\t{chosen_negatives_topk[1]}\n") 
                    if chosen_negatives_sample is not None:
                        h2.write(f"{query}\t{','.join(chosen_negatives_sample)}\n")
                    
        print(self.counter)
    
##############################
##############################
##############################

class MetaHardNegatives(HardNegatives):
    def __init__(self, qrels_path, run_path, embeddings_path, model_args, data_args, training_args):
        
        super().__init__(qrels_path, run_path, embeddings_path=None) # no need to store embeddings

        self.corpus_path = "/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.tsv" #TODO argument
        self.queries_path = "/data/user_data/jmcoelho/datasets/marco/documents/train.query.filtered.txt" #TODO argument
        self.valid_samples_path = "/data/user_data/jmcoelho/datasets/marco/documents/processed_data/pythia-160m-marco-docs-bow-pretrain/random/val.jsonl"
        self.parse_queries()
        self.parse_corpus()
        self.parse_jsonl()

        self.data_args = data_args

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        self.tokenizer.padding_side = 'right'

        self.model_static = DenseModelLESS.build(
            model_args,
            training_args,
            cache_dir=model_args.cache_dir,
            )
        self.model_static.to("cuda")


        self.model_pseudo_updates = DenseModelLESS.build(
            model_args,
            training_args,
            cache_dir=model_args.cache_dir,
            )
        
        self.model_pseudo_updates.train()

        def replace_params_with_buffers(module):
            # Create a copy of named parameters to avoid RuntimeError
            named_params_copy = dict(module.named_parameters(recurse=False))

            # Iterate over the copied dictionary
            for name, param in named_params_copy.items():
                data = param.data
                req=False
                if param.requires_grad:
                    data.requires_grad = True
                    req=True
                delattr(module, name)
                module.register_buffer(name, nn.Parameter(data, requires_grad=req))
            
            # Iterate over named children and recursively apply the function
            for name, child in module.named_children():
                replace_params_with_buffers(child)

        replace_params_with_buffers(self.model_pseudo_updates)

        # self.model_pseudo_updates.meta = nn.Module()

        # for key, value in self.model_pseudo_updates.encoder.named_parameters():
        #     self.model_pseudo_updates.meta.register_buffer(key.replace(".", "_"), value.data)
        
        # for n, p in self.model_pseudo_updates.meta.named_buffers():
        #     p.requires_grad = True

        self.model_pseudo_updates.to("cuda")

    def set_seed(self, seed):
        random.seed(seed)

    def init_optimizer(self):
        # init optimizer
        self.no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model_static.named_parameters() if not any(nd in n for nd in self.no_decay)],
             "weight_decay": 0.0,
             "names": [n for n, p in self.model_static.named_parameters() if not any(nd in n for nd in self.no_decay)],
            },
            {"params": [p for n, p in self.model_static.named_parameters() if any(nd in n for nd in self.no_decay)], 
             "weight_decay": 0.0,
             "names": [n for n, p in self.model_static.named_parameters() if any(nd in n for nd in self.no_decay)], 
            },
        ]

        self.optimizer = optim.Adam(
            optimizer_grouped_parameters,
            lr=1e-5,
            eps=1e-8,
        )

    def parse_jsonl(self):
        self.valid_samples = []
        with open(self.valid_samples_path, 'r') as file:
            for line in file:
                # Strip newline character and parse JSON
                json_data = json.loads(line.strip())
                self.valid_samples.append(json_data)
        
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
    
    def get_adam_delta(self, grads, optimizer):
        deltas = {}
        for group in optimizer.param_groups:
            for n, p in zip(group['names'], group['params']):
                grad = grads[n]
                state = optimizer.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                step = state['step'] + 1
                
                if group['weight_decay'] != 0:
                    grad = grad + group['weight_decay'] * p.data

                bias_correction1 = 1. - beta1 ** step
                bias_correction2 = 1. - beta2 ** step

                step_size = group['lr'] / bias_correction1

                _exp_avg = exp_avg * beta1 + (1. - beta1) * grad
                _exp_avg_sq = exp_avg_sq * beta2 + (1. - beta2) * grad * grad

                denom = (torch.sqrt(_exp_avg_sq + group['eps']) / math.sqrt(bias_correction2)).add_(group['eps'])
                deltas[n] = -step_size * _exp_avg / denom
        return deltas   
    
    @staticmethod
    def get_name2grad(grads, named_buffers):
        j = 0
        name2grad = {}
        for n, p in named_buffers:
            if p.requires_grad:
                name2grad[n] = grads[j]
                j += 1
        return name2grad

    def pseudo_update(self, q, d, vq, vd):
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16): #HACK hardcoded to bf16
            q = {k:v.to("cuda") for k, v in q.items()}
            d = {k:v.to("cuda") for k, v in d.items()}


            # init optim
            self.model_static.train()
            cost = self.model_static(q, d).loss
            eps = torch.zeros(cost.size(), requires_grad=True).to(cost.device)
            loss = torch.sum(cost * eps)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


            # pseudo-update
            loss = self.model_pseudo_updates(q, d).loss
            eps = torch.ones(loss.size(), requires_grad=True).to(loss.device)

            l_f_meta = torch.sum(loss * eps) 
            #self.model_pseudo_updates.zero_grad()

            grads = torch.autograd.grad(l_f_meta, 
                                        [p for n, p in self.model_pseudo_updates.named_buffers() if p.requires_grad], 
                                        create_graph=True)
            
            grads = self.get_name2grad(grads, self.model_pseudo_updates.named_buffers())
        
            # # Manually update params. grad_fn is not maintained on leaf nodes (nn.Params), hence we replaced all by nn.Buffers
            # # https://github.com/chenwydj/learning-to-learn-by-gradient-descent-by-gradient-descent
            # # https://discuss.pytorch.org/t/assign-parameters-to-nn-module-and-have-grad-fn-track-it/62677
            # for p, g in zip([p for n, p in self.model_pseudo_updates.named_buffers() if p.requires_grad], grads):
            #     p = p - 2e-5 * g

            deltas = self.get_adam_delta(grads, self.optimizer)

            self.model_pseudo_updates.update_params(deltas)

            # meta-update
            vq = {k:v.to("cuda") for k, v in vq.items()}
            vd = {k:v.to("cuda") for k, v in vd.items()}
            l_g_meta = self.model_pseudo_updates(vq, vd, per_sample=False).loss

            grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

            print(grad_eps)

            #grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

            w_tilde = torch.clamp(-grad_eps, min=0)
            norm_c = torch.sum(w_tilde)

            if norm_c != 0:
                w = w_tilde / norm_c
            else:
                w = w_tilde

            self.optimizer.zero_grad()

            print(w)
            print(w.shape)
            exit()


            # grads = torch.autograd.grad(
            #     l_f_meta, 
            #     [p for n, p in self.model_pseudo_updates.named_buffers() if p.requires_grad], 
            #     create_graph=True
            # )

            # grads = self.get_name2grad(grads, self.model_pseudo_updates.named_buffers())
            # deltas = self.convert_grad2delta(grads, self.optimizer)
            # self.model_pseudo_updates.update_params(deltas)

            
    
    def choose_negatives(self, query: int, n: int) -> list[str]:
        positive_id = self.qid2pos[query]
        negative_ids = self.qid2negs[query]

        positive_text = self.did2text[positive_id]
        negative_texts = [self.did2text[did] for did in negative_ids][:50]
        query_text = self.qid2text[query]

        queries = [query_text]
        documents = [positive_text]
        for neg in negative_texts:
            documents.append(neg)

        q, d = self.tokenize(queries, documents)

        valid_instances = random.sample(self.valid_samples, 6)
        v_queries = []
        v_documents = []
        for valid_instance in valid_instances:
            v_queries.append(self.tokenizer.decode(valid_instance["query"]))
            v_documents.append(self.tokenizer.decode(valid_instance["positives"][0]))
            v_documents += self.tokenizer.batch_decode(valid_instance["negatives"][:9])

        vq, vd =  self.tokenize(v_queries, v_documents)


        self.init_optimizer()
        self.pseudo_update(q, d, vq, vd)

        print("did pseudo update")
        exit()

        _, top_indices = torch.topk(dot_prods, k=n)
        probabilities = torch.nn.functional.softmax(dot_prods, dim=0)

        chosen_negatives_topk = []

        for index in top_indices.tolist():
            negative_id = negative_ids[index]
            chosen_negatives_topk.append(negative_id)

        chosen_negatives_sample = random.choices(negative_ids, weights=probabilities, k=n)

        return [chosen_negatives_sample, chosen_negatives_topk]
    


    def sample_hard_negatives(self, n, outpath):
        logger.info(f"Sampling started.")
        with open(outpath+"_topk", 'w') as h1, \
            open(outpath+"_sample", 'w') as h2:
                for query in tqdm(self.qid2negs):
                    
                    chosen_negatives_sample, chosen_negatives_topk = self.choose_negatives(query, n)
                    
                    h1.write(f"{query}\t{','.join(chosen_negatives_topk)}\n") 
                    h2.write(f"{query}\t{','.join(chosen_negatives_sample)}\n")