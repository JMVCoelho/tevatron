import os
import pickle
import torch
from tqdm import tqdm

BASE_MODEL = "pythia-160m-marco-docs-bow-pretrain"

path = f"/data/user_data/jmcoelho/embeddings/marco_docs/{BASE_MODEL}"
train_run = f"{path}/run.train.txt"
train_qrels = "/data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv"

def batch_doc_independent_grad_embeddings(queries, pos, docs, args):
    queries.requires_grad_(False)
    pos.requires_grad_(False)

    neg_sim = queries @ docs.T
    pos_sim = (queries * pos).sum(axis=1).repeat(len(docs), 1).T
    # if args.dist_query_pos_choose_random is not None:
    #     if pos_sim.mean().item() < neg_sim.mean().item() - args.dist_query_pos_choose_random:
    #         return None

    loss_per_doc = -(1 / (1 + (neg_sim - pos_sim).exp())).log()
    loss_per_doc.sum().backward()

    return docs.grad


all_embeddings = {'query':{}, 'passages':{}}

for file_name in os.listdir(path):
    if file_name.startswith("corpus") and file_name.endswith(".pkl"):
        file_path = os.path.join(path, file_name)
        with open(file_path, "rb") as f:
            corpus_data = pickle.load(f)
            embeddings, ids = corpus_data
            all_embeddings['passages'].update(zip(ids, embeddings))

with open(f"{path}/query-train.pkl","rb") as f:
    query_data = pickle.load(f)
    embeddings, ids = query_data
    all_embeddings['query'].update(zip(ids, embeddings))

qid2pos = {}
with open(train_qrels, 'r') as h:
     for line in h:
        qid, q0, did, rel = line.strip().split("\t")
        if qid not in qid2pos:
            qid2pos[qid] = did


qid2negs = {}
with open(train_run, 'r') as h:
     for line in h:
        qid, did, score = line.strip().split()
        if qid not in qid2negs:
            qid2negs[qid] = []
        if did not in qid2pos[qid]:
            qid2negs[qid].append(did) 



with open("hard_negs.txt", 'w') as h:
    for query in tqdm(qid2negs):
        q_embed = torch.tensor(all_embeddings['query'][query]).view(1, -1)
        pos_embed = torch.tensor(all_embeddings['passages'][qid2pos[query]]).view(1, -1)
        neg_embeds = torch.tensor([all_embeddings['passages'][negative] for negative in qid2negs[query]], requires_grad=True)

        gradient_norms = torch.norm(batch_doc_independent_grad_embeddings(q_embed, pos_embed, neg_embeds, None), dim=1)
        top_values, top_indices = torch.topk(gradient_norms, k=9)
        
        chosen_negatives = []
        for index in top_indices.tolist():
            negative_id = qid2negs[query][index]
            chosen_negatives.append(negative_id)
        
        line_to_write = f"{query}\t{','.join(chosen_negatives)}\n"
        h.write(line_to_write)
        


    