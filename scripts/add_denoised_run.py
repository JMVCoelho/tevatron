import random
random.seed(17121998)
N_DENOISE = 1

# this assumes run "qid\t<did>" where <did> is a comma separated list of negs, ordered by hardness.
run="/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-20pc-sample-less-negs-triplet-topk/20pc-sample-run-splits/less-opacus-triplet/hardnegs_less_opacus.20.pc.full.topk"
out=f"/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-20pc-sample-less-negs-triplet-topk/20pc-sample-run-splits/less-opacus-triplet/hardnegs_less_opacus.20.pc.full.topk.denoised.{N_DENOISE}" 
full_run = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-20pc-sample-less-negs-triplet-topk/run.train.20pc.sample.txt"

qid2negs = {}
with open(run, 'r') as h:
    for line in h:
        qid, negs = line.strip().split("\t")
        negs = negs.split(",")

        qid2negs[qid] = negs

qid2ordered_negs = {}
with open(full_run, 'r') as h:
    for line in h:
        qid, did, score = line.strip().split()

        if qid not in qid2ordered_negs:
            i = 0
            qid2ordered_negs[qid] = []

        if did in qid2negs[qid]:
            qid2ordered_negs[qid].append(did)
        i+=1


for qid in qid2ordered_negs:
    qid2ordered_negs[qid] = qid2ordered_negs[qid][N_DENOISE:]

with open(out, 'w') as h:
    for qid in qid2ordered_negs:
        h.write(f"{qid}\t{','.join(qid2ordered_negs[qid])}\n")
