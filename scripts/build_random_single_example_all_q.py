import sys

import random

query_idx = int(sys.argv[1])
model = sys.argv[2]
out_path = sys.argv[3]

# query="1185869"
# model="pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1"
# out_path = "temp__"

negs = []
queries = set()

with open(f"/data/user_data/jmcoelho/embeddings/marco_docs/{model}/run.gen.txt") as h:
    for line in h:
        qid, did, _ = line.strip().split()

        queries.add(qid)

queries = list(queries)
query = queries[query_idx]

with open(f"/data/user_data/jmcoelho/embeddings/marco_docs/{model}/run.gen.txt") as h:
    for line in h:
        qid, did, _ = line.strip().split()

        if qid == query:
            negs.append(did)

        if len(negs) == 100:
            break


pos = None
with open("/data/user_data/jmcoelho/datasets/marco/documents/qrels.gen.2.tsv", 'r') as h:

    for line in h:
        qid, q0, did, rel = line.strip().split("\t")

        if qid == query:
            pos = did
            break

negs = [n for n in negs if n != pos]
negs = random.sample(negs, 9)

with open(out_path, 'w') as f:
    f.write(f"{query}\t{','.join(negs)}\n")