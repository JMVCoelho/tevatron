import random
random.seed(17121998)
N_MOMENTUM = 3

# run = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs64-10pc-sample-less-negs/less_train_run_splits/less_grad_bs64_temperature_top100_triplet/hardnegs_less_opacus_10.pc.full.topk"
# out = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs64-10pc-sample-less-negs/less_train_run_splits/less_grad_bs64_temperature_top100_triplet/hardnegs_less_opacus_10.pc.full.topk.momentum"

# run="/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-all-queries-less-topk-negs-top50/full-queries-run-splits/less-opacus-triplet/hardnegs_less_opacus_full_qs.topk"
# out="/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-all-queries-less-topk-negs-top50/full-queries-run-splits/less-opacus-triplet/hardnegs_less_opacus_full_qs.topk.3.momentum.denoise" 

# run="/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-all-queries-less-topk-negs-top50/random_train_run_splits/random/full.queries.train+val.random.txt.348182"
# out="/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-all-queries-less-topk-negs-top50/random_train_run_splits/random/full.queries.train+val.random.3.momentum.denoise.348182" 
#prev_iter = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-1024-marco-docs-bow-contrastive-pretrain/run.train.txt"

run="/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-all-queries-random-negs-top100/random_train_run_splits/random/full.stain+val.random.top100.txt"
out="/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-all-queries-random-negs-top100/random_train_run_splits/random/full.stain+val.random.top100.txt.3momentum.denoise" 
prev_iter = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-1024-marco-docs-bow-contrastive-pretrain/run.train.txt"


#add
method = "replace" #needs full_run to be defined (qid did score)
full_run = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-all-queries-random-negs-top100/run.train.txt"


qid2negs = {}
with open(run, 'r') as h:
    for line in h:
        qid, negs = line.strip().split("\t")
        negs = negs.split(",")

        qid2negs[qid] = negs


qid2pos = {}

with open("/data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv", 'r') as h:

    for line in h:
        qid, q0, did, rel = line.strip().split("\t")

        if qid not in qid2pos:
            qid2pos[qid] = did


qid2prev_negs = {}
with open(prev_iter, 'r') as h:

    for line in h:
        qid, did, score = line.strip().split()

        if qid not in qid2prev_negs:
            qid2prev_negs[qid] = []

        if did not in qid2pos[qid]:
            qid2prev_negs[qid].append(did)

prevq = list(qid2prev_negs.keys())
currq = list(qid2negs.keys())

qid2prev_negs = {k:random.sample(v, N_MOMENTUM) for k,v in qid2prev_negs.items()}


if method == "add":
    qid2fullnegs = {k: qid2prev_negs[k] + qid2negs[k] for k,_ in qid2negs.items()}


elif method == "replace":
    missing = 0
    qid2ordered_negs = {}
    with open(full_run, 'r') as h:
        for line in h:
            qid, did, score = line.strip().split()
            if qid not in qid2negs:
                missing += 1
                continue

            if qid not in qid2ordered_negs:
                i = 0
                qid2ordered_negs[qid] = {}

            if did in qid2negs[qid]:
                qid2ordered_negs[qid][i] = did
            i+=1
            

    qid2fullnegs = {k: qid2prev_negs[k] + list(qid2ordered_negs[k].values())[N_MOMENTUM:] for k,_ in qid2negs.items()}

print("missing: ", missing)
with open(out, 'w') as h:
    for qid in qid2fullnegs:
        h.write(f"{qid}\t{','.join(qid2fullnegs[qid])}\n")
