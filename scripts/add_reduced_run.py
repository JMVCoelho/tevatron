import random
random.seed(17121998)
TOTAL_NEGS = 6

# this assumes run "qid\t<did>" where <did> is a comma separated list of negs, ordered by hardness.
run = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-20pc-sample-less-negs-triplet-topk/random_train_run_splits/random/20pc.tain+val.random.txt"
out = f"/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-20pc-sample-less-negs-triplet-topk/random_train_run_splits/random/20pc.tain+val.random.{TOTAL_NEGS}negs.txt" 

qid2negs = {}
with open(run, 'r') as h:
    for line in h:
        qid, negs = line.strip().split("\t")
        negs = negs.split(",")

        qid2negs[qid] = negs

for qid in qid2negs:
    negs = qid2negs[qid]
    reduced_negs = random.sample(negs, TOTAL_NEGS)
    qid2negs[qid] = reduced_negs

with open(out, 'w') as h:
    for qid in qid2negs:
        h.write(f"{qid}\t{','.join(qid2negs[qid])}\n")
