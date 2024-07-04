import random
random.seed(17121998)
N_MOMENTUM = 3


# full_run_current = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-all-queries-less-5-group-level-best/run.train.txt"
# full_run_prev = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-1024-marco-docs-bow-contrastive-pretrain/run.train.txt"
# out = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-all-queries-less-5-group-level-best/run.train.2.to.1.momentum.txt"


full_run_current = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1/run.train.txt"
full_run_prev = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-1024-marco-docs-bow-contrastive-pretrain/run.train.txt"
out = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1/run.train.2.to.1.momentum.txt"


qid2prev_negs = {}
with open(full_run_prev, 'r') as h:

    for line in h:
        qid, did, score = line.strip().split()

        if qid not in qid2prev_negs:
            qid2prev_negs[qid] = []

        qid2prev_negs[qid].append(did)

qid2curr_negs = {}
with open(full_run_current, 'r') as h:

    for line in h:
        qid, did, score = line.strip().split()

        if qid not in qid2curr_negs:
            qid2curr_negs[qid] = []

        qid2curr_negs[qid].append(did)



# 367012 * 150

with open(out, 'w') as fout:
    for qid in qid2curr_negs:
        curr_negs = qid2curr_negs[qid]
        prev_negs = qid2prev_negs[qid]
        momentum_sample = random.sample(prev_negs, 50)

        negs = curr_negs + momentum_sample

        for neg in negs:
            fout.write(f"{qid} {neg} 0\n")


