# import transformers
# import torch

# state_dict = torch.load("/data/user_data/jmcoelho/models/pythia-160m-1024-marco-docs-hf/pytorch_model.bin")

# model = transformers.AutoModel.from_pretrained(
#     "/data/user_data/jmcoelho/models/pythia-160m-1024-marco-docs-hf/", local_files_only=True, state_dict=state_dict, attn_implementation="flash_attention_2"
# )

# print(model.__class__)

# import pickle

# with open('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs/corpus.1.pkl', 'rb') as h:
#     x = pickle.load(h)

# print(len(x))
# print(len(x[1]))

# with open('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs/corpus.2.pkl', 'rb') as h:
#     x = pickle.load(h)

# print(len(x))
# print(len(x[1]))

# with open('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs/corpus.3.pkl', 'rb') as h:
#     x = pickle.load(h)

# print(len(x))
# print(len(x[1]))

# with open('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs/corpus.0.pkl', 'rb') as h:
#     x = pickle.load(h)

# print(len(x))
# print(len(x[1]))



# import json
# import random

# def parse_jsonl_file(file_path):
#     data = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             # Load each line as a JSON object
#             json_object = json.loads(line)
#             data.append(json_object)
#     return data


# sample = parse_jsonl_file("/data/user_data/jmcoelho/datasets/marco/documents/10.percent.sample.train.query.filtered.jsonl")
# full = parse_jsonl_file("/data/user_data/jmcoelho/datasets/marco/documents/train.query.filtered.jsonl")


# sample_train_ids = set([s["query_id"] for s in sample])

# full = [s for s in full if s["query_id"] not in sample_train_ids]

# random_selection = random.sample(full, 1000)


# def write_to_jsonl(data, file_path):
#     with open(file_path, 'w') as file:
#         for item in data:
#             json.dump(item, file)
#             file.write('\n')

# write_to_jsonl(random_selection, "/data/user_data/jmcoelho/datasets/marco/documents/10.percent.sample.val.query.filtered.jsonl")

## BM25 SCRIPT

import json
import pickle

def parse_jsonl_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Load each line as a JSON object
            json_object = json.loads(line)
            data.append(json_object)
    return data

path = "/data/user_data/jmcoelho/datasets/marco/documents/"

with open(path + "id2seqint_mapper.pkl", 'rb') as h:
    mapper = pickle.load(h)

print(list(mapper.keys())[:10])


sample = parse_jsonl_file(path+"20.percent.sample.train.query.filtered.jsonl")
queries = set([s["query_id"] for s in sample])
print(len(queries))

with open(path+"msmarco-doctrain-top100", 'r') as h, \
    open(path+"20pc-msmarco-doctrain-top100-seqids.txt", 'w') as out:
    for line in h:
        q, q0, did, pos, score, method = line.strip().split()
        if q in queries:
            did = mapper[did[1:]]
            out.write(f"{q}\t{did}\t{score}\n")




# Sanity CHECK
# def load_qrels(path):
#     qid2pos = {}

#     with open(path, 'r') as h:

#         for line in h:
#             qid, q0, did, rel = line.strip().split("\t")

#             if qid not in qid2pos:
#                 qid2pos[qid] = []

#             qid2pos[qid].append(did)
    
#     return qid2pos

# qid2pos = load_qrels("/data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv")
# print(len(qid2pos))

# ZAU = []

# def load_run(path):
#     qid2negs = {}

#     with open(path, 'r') as h:
#         i=1
#         for line in h:  
#             qid, did, score = line.strip().split()
#             if qid not in qid2negs:
#                 qid2negs[qid] = []

#             if did not in qid2pos[qid]:
#                 qid2negs[qid].append(did)

#     return qid2negs 


# qid2negs = load_run("/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs64-10pc-sample-less-negs/less_train_run_splits/run.train.10pc.sample.0")

# for idx,qid in enumerate(qid2negs):
#     if len(qid2negs[qid]) not in [99, 100]:
#         print(qid, len(qid2negs[qid]))
