import pickle
import numpy as np
import sys
import os
model_to_remove_name = sys.argv[1]


print(f"MERGING INDEXES FOR {model_to_remove_name}")
model_to_remove_path = "/data/user_data/jmcoelho/embeddings/marco_docs/"
small_qrels = "/home/jmcoelho/tevatron/qrels/marco.docs.dev.move.passage.qrel.tsv"

out_path = f"/data/user_data/jmcoelho/embeddings/marco_docs/moving_passage/{model_to_remove_name}/reduced_index"
if not os.path.exists(out_path):
    os.makedirs(out_path)
    
dids = set()
with open(small_qrels, 'r') as h:
    for line in h:
        qid,_,did,_ = line.strip().split('\t')
        dids.add(did)

def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup

full_0 = f"{model_to_remove_path}{model_to_remove_name}/corpus.0.pkl"
full_1 = f"{model_to_remove_path}{model_to_remove_name}/corpus.1.pkl"
full_2 = f"{model_to_remove_path}{model_to_remove_name}/corpus.2.pkl"
full_3 = f"{model_to_remove_path}{model_to_remove_name}/corpus.3.pkl"

arr_0, lookup_0 = pickle_load(full_0)
arr_1, lookup_1 = pickle_load(full_1)
arr_2, lookup_2 = pickle_load(full_2)
arr_3, lookup_3 = pickle_load(full_3)

res = [[],[]]

for i in range(len(lookup_0)):
    if lookup_0[i] not in dids:
        res[0].append(arr_0[i])
        res[1].append(lookup_0[i])

for i in range(len(lookup_1)):
    if lookup_1[i] not in dids:
        res[0].append(arr_1[i])
        res[1].append(lookup_1[i])

for i in range(len(lookup_2)):
    if lookup_2[i] not in dids:
        res[0].append(arr_2[i])
        res[1].append(lookup_2[i])

for i in range(len(lookup_3)):
    if lookup_3[i] not in dids:
        res[0].append(arr_3[i])
        res[1].append(lookup_3[i])

print(len(res[1]))
print(len(set(res[1])))

with open(f"{out_path}/corpus.0.pkl", 'wb') as f:
        pickle.dump(res, f)
