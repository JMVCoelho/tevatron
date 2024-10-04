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
        pos_embed = torch.tensor(self.all_embeddings['passages'][self.qid2pos[query][0]]).view(1, -1)
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
        positive_id = self.qid2pos[query][0]
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

        self.temperatures = [1, 10, 20, 40, 80, 100, 1000, 10000, 100000]
        #self.temperatures = ["inf"]

        self.dot_prods = {}

        self.avg = {}
        self.var = {}
        self.norm = {}

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

            n_params = vector.shape[0]

            q = {k:v.to("cuda") for k, v in q.items()}
            d = {k:v.to("cuda") for k, v in d.items()}

            loss = self.model(q, d).loss

            loss.backward(gradient=torch.ones_like(loss))

            rolling_dot_prods = None # more memory efficient that shaping to [batchsize, n_params]
            rolling_sum = None
            rolling_squared_sum = None
            for param in self.model.parameters(): # Opacus is computing for all documents?? ermmmmmmm... model(**docs), this seems to be their samples
                if param.grad is not None:
                    current_vector = param.grad_sample.detach().view(param.grad_sample.size(0), -1)[1:,:]

                    sums = torch.sum(current_vector, dim=-1)
                    squared_sums = torch.sum(current_vector**2, dim=-1)

                    dimensions = current_vector.size(-1)
                    current_dot = torch.matmul(current_vector, vector[:dimensions])
                    vector = vector[dimensions:]

                    if rolling_dot_prods is None:
                        rolling_dot_prods = current_dot
                        rolling_sum = sums
                        rolling_squared_sum = squared_sums
                    else:
                        rolling_dot_prods += current_dot
                        rolling_sum += sums
                        rolling_squared_sum += squared_sums
            
            avg = rolling_sum/n_params
            norm = torch.sqrt(rolling_squared_sum)
            var = (rolling_squared_sum / n_params) - ((rolling_sum/n_params) ** 2)


        self.model.zero_grad() 

        return rolling_dot_prods, avg, norm, var
    
    def choose_negatives(self, query: int, n: int) -> list[str]:
        positive_id = self.qid2pos[query][0]
        negative_ids = self.qid2negs[query]

        positive_text = self.did2text[positive_id]
        negative_texts = [self.did2text[did] for did in negative_ids]
        query_text = self.qid2text[query]

        queries = [query_text]
        documents = [positive_text]
        for neg in negative_texts:
            documents.append(neg)

        q, d = self.tokenize(queries, documents)

        dot_prods, avg, norm, var = self.get_grad(q, d, random.choice(self.validation_gradients))

        _, top_indices = torch.topk(dot_prods, k=n)

        self.dot_prods[query] = dot_prods
        self.avg[query] = avg
        self.norm[query] = norm
        self.var[query] = var

        samples = []
        for temperature in self.temperatures:

            probabilities = torch.nn.functional.softmax(dot_prods / temperature, dim=0).detach().float().cpu().numpy()
            probabilities /= probabilities.sum() # make sure sums to 1... numpy cant handle 1.000000001
            
            try:
                chosen_negatives_sample = np.random.choice(negative_ids, size=n, replace=False, p=probabilities)
            except Exception as e:
                print("sampling randomly")
                chosen_negatives_sample = np.random.choice(negative_ids, size=n, replace=False)

            if temperature not in self.distributions:
                self.distributions[temperature] = probabilities[:99]
            else:
                self.distributions[temperature] += probabilities[:99]

            samples.append(chosen_negatives_sample)


        
        chosen_negatives_topk = []

        for index in top_indices.tolist():
            negative_id = negative_ids[index]
            chosen_negatives_topk.append(negative_id)

        return [samples, chosen_negatives_topk]
    
    def sample_hard_negatives(self, n, outpath):
        self.counter_topk = defaultdict(int)
        self.counter_sample = defaultdict(int)

        self.distributions = {}
        logger.info(f"Sampling started.")

        with open(outpath+"_topk", 'w') as h1:
                k = 0
                for query in tqdm(self.qid2negs):

                    samples, chosen_negatives_topk = self.choose_negatives(query, n)
                    
                    h1.write(f"{query}\t{','.join(chosen_negatives_topk)}\n") 
                    if samples is not None:
                        for sample, temperature in zip(samples, self.temperatures):
                            with open(outpath+f"_sample_t{temperature}", 'a') as h2:
                                h2.write(f"{query}\t{','.join(sample)}\n")
                    
                    k+=1
                    if k == 5000:
                        break


        with open(outpath+"_dotprods.pkl", 'wb') as h:
            pickle.dump(self.dot_prods, h, protocol=pickle.HIGHEST_PROTOCOL)
        with open(outpath+"_avg.pkl", 'wb') as h:
            pickle.dump(self.avg, h, protocol=pickle.HIGHEST_PROTOCOL)
        with open(outpath+"_norm.pkl", 'wb') as h:
            pickle.dump(self.norm, h, protocol=pickle.HIGHEST_PROTOCOL)
        with open(outpath+"_var.pkl", 'wb') as h:
            pickle.dump(self.var, h, protocol=pickle.HIGHEST_PROTOCOL)

                    
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
            vector = torch.tensor(vector).to("cuda")

            q = {k:v.to("cuda") for k, v in q.items()}
            d = {k:v.to("cuda") for k, v in d.items()}

            loss = self.model(q, d).loss

            loss.backward(gradient=torch.ones_like(loss))
            
            gradients = []
            for param in self.model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.detach().view(-1))
            
            gradient_vector = torch.cat(gradients)
            self.model.zero_grad()

            dot_prod = torch.dot(gradient_vector, vector)

        return dot_prod.item()
    
    def choose_negatives(self, query: int, n: int) -> list[str]:
        positive_id = self.qid2pos[query][0]
        negative_ids = self.qid2negs[query]

        samples = []

        # Taking 5 independent samples
        for _ in range(5):
            sample = random.sample(negative_ids, n)
            samples.append(sample)

        highest_score = float('-inf')
        lowest_score = float('inf')
        best_sample = None
        worst_sample = None

        valid_grad = random.choice(self.validation_gradients)
        for sample in samples:

            positive_text = self.did2text[positive_id]
            negative_texts = [self.did2text[did] for did in sample]


            query_text = self.qid2text[query]

            queries = [query_text]
            documents = [positive_text] + negative_texts

            q, d = self.tokenize(queries, documents)

            score = self.get_grad(q, d, valid_grad)

            if score > highest_score:
                highest_score = score
                best_sample = sample
            if score < lowest_score:
                lowest_score = score
                worst_sample = sample

        return best_sample, worst_sample
    
    def sample_hard_negatives(self, n, outpath):
        logger.info(f"Sampling started.")

        k = 0
        with open(outpath+"_group_level_best", 'w') as h1, \
            open(outpath+"_group_level_worst", 'w') as h2:
                for query in tqdm(self.qid2negs):
                    
                    best, worst = self.choose_negatives(query, n)
                    
                    h1.write(f"{query}\t{','.join(best)}\n") 
                    h2.write(f"{query}\t{','.join(worst)}\n")
                    k += 1

                    if k == 1:
                        break

    
##############################
##############################
##############################
class LESSHardNegativesQueryLevelValid(HardNegatives):
    def __init__(self, qrels_path, run_path, valid_path, model_args, data_args, training_args):
        
        super().__init__(qrels_path, run_path, embeddings_path=None) # no need to store embeddings

        self.corpus_path = "/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.tsv" #TODO argument
        self.queries_path = "/data/user_data/jmcoelho/datasets/marco/documents/train.query.filtered.txt" #TODO argument
        self.valid_samples_path = valid_path 
        print(self.valid_samples_path)

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


        self.model = DenseModelLESS.build(
            model_args,
            training_args,
            cache_dir=model_args.cache_dir,
            )
        self.model.to("cuda")

        self.model_copy = copy.deepcopy(self.model)
    
    def set_seed(self, seed):
        random.seed(seed)
    
    def parse_jsonl(self):
        self.valid_samples = []
        with open(self.valid_samples_path, 'r') as file:
            for line in file:
                # Strip newline character and parse JSON
                json_data = json.loads(line.strip())
                self.valid_samples.append(json_data)
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

    def get_grad(self, q, d, qv, dv):

        # with torch.cuda.amp.autocast(dtype=torch.bfloat16): #HACK hardcoded to bf16
        #     qv = {k:v.to("cuda") for k, v in qv.items()}
        #     dv = {k:v.to("cuda") for k, v in dv.items()}

        #     loss = self.model(qv, dv).loss

        # initial = loss.item()
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16): #HACK hardcoded to bf16
            #TRAIN
            q = {k:v.to("cuda") for k, v in q.items()}
            d = {k:v.to("cuda") for k, v in d.items()}

            loss = self.model(q, d).loss

            loss.backward(gradient=torch.ones_like(loss))
            
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        optimizer.step()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16): #HACK hardcoded to bf16
            #VALID
            qv = {k:v.to("cuda") for k, v in qv.items()}
            dv = {k:v.to("cuda") for k, v in dv.items()}

            loss = self.model(qv, dv).loss

        final = loss.item()

        self.model.load_state_dict(self.model_copy.state_dict())

        return final
    
    def choose_negatives(self, query: int, n: int) -> list[str]:
        positive_id = self.qid2pos[query][0]
        negative_ids = self.qid2negs[query]
        
        self.qid2samplescores[query] = []
        samples = []

        # Taking 5 independent samples
        for _ in range(5):
            sample = random.sample(negative_ids, n)
            samples.append(sample)

        highest_loss = float('-inf')
        lowest_loss = float('inf')
        best_sample = None
        worst_sample = None

        valid_instances = random.sample(self.valid_samples, 10)
        v_queries = []
        v_documents = []
        for valid_instance in valid_instances:
            v_queries.append(self.tokenizer.decode(valid_instance["query"]))
            v_documents.append(self.tokenizer.decode(valid_instance["positives"][0]))
            v_documents += self.tokenizer.batch_decode(valid_instance["negatives"][:9])

        qv, dv =  self.tokenize(v_queries, v_documents)

        for sample in samples:

            positive_text = self.did2text[positive_id]
            negative_texts = [self.did2text[did] for did in sample]


            query_text = self.qid2text[query]

            queries = [query_text]
            documents = [positive_text] + negative_texts

            q, d = self.tokenize(queries, documents)

            valid_loss = self.get_grad(q, d, qv, dv)

            info_to_log = [sample, valid_loss]
            self.qid2samplescores[query].append(info_to_log)

            if valid_loss > highest_loss:
                highest_loss = valid_loss
                worst_sample = sample
            if valid_loss < lowest_loss:
                lowest_loss = valid_loss
                best_sample = sample

        return best_sample, worst_sample
    
    def sample_hard_negatives(self, n, outpath):
        logger.info(f"Sampling started.")

        self.qid2samplescores = {}
        
        with open(outpath+"_group_level_best", 'w') as h1, \
            open(outpath+"_group_level_worst", 'w') as h2:
                for query in tqdm(self.qid2negs):
                    
                    best, worst = self.choose_negatives(query, n)
                    
                    h1.write(f"{query}\t{','.join(best)}\n") 
                    h2.write(f"{query}\t{','.join(worst)}\n")

        with open(outpath + "_log.pkl", 'wb') as h:
            pickle.dump(self.qid2samplescores, h, protocol=pickle.HIGHEST_PROTOCOL)