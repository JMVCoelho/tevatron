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

qids = set()
with open(small_qrels, 'r') as h:
    for line in h:
        qid,_,did,_ = line.strip().split('\t')
        qids.add(qid)

def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup

qrys = f"{model_to_remove_path}{model_to_remove_name}/query-dev.pkl"
arr_0, lookup_0 = pickle_load(qrys)


res = [[],[]]

for i in range(len(lookup_0)):
    if lookup_0[i] in qids:
        res[0].append(arr_0[i])
        res[1].append(lookup_0[i])

print(len(res[1]))
print(len(set(res[1])))

with open(f"{out_path}/qry-dev-reduced.pkl", 'wb') as f:
        pickle.dump(res, f)
