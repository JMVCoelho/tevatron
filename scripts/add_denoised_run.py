import random
random.seed(17121998)
N_DENOISE = 3

def load_qrels(path):
    qid2pos = {}
    with open(path, 'r') as h:
        for line in h:
            qid, q0, did, rel = line.strip().split("\t")
            
            if qid not in qid2pos:
                qid2pos[qid] = []
            
            qid2pos[qid].append(did)

    return qid2pos

qid2pos = load_qrels("/data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv")


# this assumes run "qid\t<did>" where <did> is a comma separated list of negs, ordered by hardness.
# run="/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-20pc-sample-less-negs-triplet-topk/20pc-sample-run-splits/less-opacus-triplet/hardnegs_less_opacus.20.pc.full.topk"
# out=f"/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-20pc-sample-less-negs-triplet-topk/20pc-sample-run-splits/less-opacus-triplet/hardnegs_less_opacus.20.pc.full.topk.denoised.{N_DENOISE}.random.added" 
# full_run = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-20pc-sample-less-negs-triplet-topk/run.train.20pc.sample.txt"

# add_random=False
# random_run = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-20pc-sample-less-negs-triplet-topk/random_train_run_splits/random/20pc.tain+val.random.txt"


run="/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-1024-marco-docs-bow-contrastive-pretrain/20pc-sample-run-splits/less-opacus/hardnegs_less_opacus.20.pc.full.topk"
out=f"delete" 
full_run = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-1024-marco-docs-bow-contrastive-pretrain/run.train.20pc.sample.txt"

add_random=False
random_run = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-20pc-sample-less-negs-triplet-topk/random_train_run_splits/random/20pc.tain+val.random.txt"


qid2negs = {}
with open(run, 'r') as h:
    for line in h:
        qid, negs = line.strip().split("\t")
        negs = negs.split(",")

        qid2negs[qid] = negs

qid2randnegs = {}
with open(random_run, 'r') as h:
    for line in h:
        qid, negs = line.strip().split("\t")
        negs = negs.split(",")

        qid2randnegs[qid] = negs

import collections

counter = collections.defaultdict(int)

qid2ordered_negs = {}
with open(full_run, 'r') as h:
    for line in h:
        qid, did, score = line.strip().split()

        if qid not in qid2ordered_negs:
            i = 0
            qid2ordered_negs[qid] = []

        if did in qid2negs[qid]:
            qid2ordered_negs[qid].append(did)
            counter[i] += 1

        if did not in qid2pos[qid]:
            i+=1

print(counter)
exit()

for qid in qid2ordered_negs:
    qid2ordered_negs[qid] = qid2ordered_negs[qid][N_DENOISE:]
    if add_random:
        rand_negs = [neg for neg in qid2randnegs[qid] if neg not in qid2ordered_negs[qid]]
        sampled_negs = random.sample(rand_negs, N_DENOISE)
        qid2ordered_negs[qid] += sampled_negs
        assert len(set(qid2ordered_negs[qid])) == 9


with open(out, 'w') as h:
    for qid in qid2ordered_negs:
        h.write(f"{qid}\t{','.join(qid2ordered_negs[qid])}\n")
