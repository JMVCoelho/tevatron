import random
random.seed(17121998)
N_MOMENTUM = 3

import sys

run=sys.argv[1]
prev_iter=sys.argv[2]
out=sys.argv[3]


qid2negs = {}
with open(run, 'r') as h:
    for line in h:
        qid, negs = line.strip().split("\t")
        negs = negs.split(",")

        negs = random.sample(negs, 6)

        qid2negs[qid] = negs


qid2prevnegs = {}
with open(prev_iter, 'r') as h:
    for line in h:
        qid, negs = line.strip().split("\t")
        negs = negs.split(",")
        negs = random.sample(negs, 3)
        qid2prevnegs[qid] = negs

qid2fullnegs = {k: qid2prevnegs[k] + qid2negs[k] for k,_ in qid2negs.items()}

with open(out, 'w') as h:
    for query in qid2fullnegs:
        negs = qid2fullnegs[query]
        h.write(f"{query}\t{','.join(negs)}\n")

