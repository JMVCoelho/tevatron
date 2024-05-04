from tqdm import tqdm

import random
random.seed(17121998)

qid2pos = {}

with open("/data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv", 'r') as h:

    for line in h:
        qid, q0, did, rel = line.strip().split("\t")

        if qid not in qid2pos:
            qid2pos[qid] = did


qid2negs = {}

qid2posscore={}

with open("/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-pretrain/run.train.txt", 'r') as h:

    for line in h:
        qid, did, score = line.strip().split()

        if qid not in qid2negs:
            qid2negs[qid] = []

        if did not in qid2pos[qid]:
            qid2negs[qid].append(float(score))
        else:
            qid2posscore[qid] = float(score)


print(len(qid2negs))
# qid2hardness = {}

# for qid in qid2negs:
#     if qid in qid2posscore:
#         pos_score = float(qid2posscore[qid])

#         hardness = 0
#         for neg_score in qid2negs[qid]:
#             if neg_score > pos_score:
#                 hardness += 1
#         qid2hardness[qid] = hardness

def top_n_keys_with_largest_values(d, n):
    return [key for key, _ in sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]]

def get_random_keys(dictionary, n):
    return random.sample(list(dictionary.keys()), n)

#hardest_queries = set(top_n_keys_with_largest_values(qid2hardness, 18350))
random_queries = set(get_random_keys(qid2negs, 18350))


with open("/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-pretrain/run.train.txt", 'r') as h, \
    open("/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-pretrain/run.train.random.txt", 'w') as out:

    for line in tqdm(h):
        qid, did, score = line.strip().split()

        if qid in random_queries:
            out.write(f"{qid} {did} {score}\n")



