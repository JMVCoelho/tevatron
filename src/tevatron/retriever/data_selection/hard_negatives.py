from abc import ABC, abstractmethod
from tqdm import tqdm
from sklearn.cluster import KMeans
from contextlib import nullcontext
import numpy as np
import torch
import random
import glob
import pickle
import os

from tevatron.retriever.dataset import TrainDatasetPreprocessed
from tevatron.retriever.collator import TrainCollatorPreprocessed
from tevatron.retriever.modeling import DenseModel

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
        k = 0
        with open(outpath, 'w') as h:
            for query in tqdm(self.qid2negs):
                
                chosen_negatives = self.choose_negatives(query, n)
                
                line_to_write = f"{query}\t{','.join(chosen_negatives)}\n"
                h.write(line_to_write)
                k+=1
                if k==20:
                    exit()

    


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

        self.validation_gradient = self.get_validation_gradient(embeddings_path)

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
        if os.path.exists(f"{embeddings_path}/validation_grad_bs20.pkl"):
            logger.info(f"Loading pre-computed validation set gradient")

            with open(f"{embeddings_path}/validation_grad_bs20.pkl", 'rb') as h:
                grad = pickle.load(h)
                return grad
        
        else:
            raise ValueError("Missing validation gradient")
        
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

            dot_prods.append(np.dot(grad, self.validation_gradient))
        
        _, top_indices = torch.topk(torch.tensor(dot_prods), k=n)

        print(top_indices)

        chosen_negatives = []
        for index in top_indices.tolist():
            negative_id = negative_ids[index]
            chosen_negatives.append(negative_id)


        return chosen_negatives


