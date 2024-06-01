PATH_LESS="/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs64-10pc-sample-less-negs/less_train_run_splits/less_grad_bs64_temperature_top100"
PATH_RANDOM="/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs64-10pc-sample-less-negs/random_train_run_splits/random/"


less_negs = f"{PATH_LESS}/hardnegs_less_opacus_10.pc.full.sample.t1.momentum-3headrep"
random_negs = f"{PATH_RANDOM}/10pc.train.random.txt"
out = f"{PATH_LESS}/joint/joint.10pc.random-denoisedless.txt"

def parse_negs(path):
    qid2negs = {}
    with open(path, 'r') as h:
        for line in h:
            qid, negs = line.strip().split("\t")
            negs = negs.split(",")

            qid2negs[qid] = negs
    
    return qid2negs

qid2lessnegs = parse_negs(less_negs)
qid2randomnegs = parse_negs(random_negs)

qid2jointnegs = {k:qid2lessnegs[k] + qid2randomnegs[k] for k in qid2randomnegs.keys()}

with open(out, 'w') as h:
    for qid in qid2jointnegs:
        h.write(f"{qid}\t{','.join(qid2jointnegs[qid])}\n")