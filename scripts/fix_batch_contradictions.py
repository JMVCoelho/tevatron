
qid2pos = {}

with open("/data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv", 'r') as h:

    for line in h:
        qid, q0, did, rel = line.strip().split("\t")

        if qid not in qid2pos:
            qid2pos[qid] = did

qid2negs = {}
with open("/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs64-10pc-sample-less-negs/less_train_run_splits/contrastive/hardnegs_less_opacus_10.pc.sample.full.txt", 'r') as h:
#with open("/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs64-10pc-sample-less-negs/random_train_run_splits/random/10pc.train.random.txt", 'r') as h:
    for line in h:
        qid, negs = line.strip().split("\t")
        negs = negs.split(",")

        qid2negs[qid] = negs

k = 0
batch = 1
counter = 0
batch_pos = []
batch_negs = []

total = 0

for qid in qid2negs:
    batch_pos.append(qid2pos[qid])
    for neg in qid2negs[qid]:
        batch_negs.append(neg)

    k += 1
    if k==64:
        
        batch_pos = list(set(batch_pos))

        for neg in batch_negs:
            if neg in batch_pos:
                counter += 1
                total += 1

        print(f"{batch} {counter}")

        batch_pos = []
        batch_negs = []
        k = 0
        counter = 0
        batch += 1

print(total)


