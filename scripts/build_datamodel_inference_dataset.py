import random
import pickle


random.seed(17121998)


run_path = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1/run.train.txt"
test_data_path =  f"/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1/group_level_10000_two_valid_orcale_momentum_600_query_2k/datamodel_test_independency.tsv"

QID2POS  = {}

with open("/data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv", 'r') as h:

    for line in h:
        qid, q0, did, rel = line.strip().split("\t")

        if qid not in QID2POS:
            QID2POS[qid] = []
        
        QID2POS[qid].append(did)


all_labels = []


with open(run_path, 'r') as h, \
    open(test_data_path, 'w') as out:
        
        for line in h:
            qid, did, _ = line.strip().split()
            pos = QID2POS[qid]

            if did not in pos:
                 out.write(f"{qid}\t{pos[0]}\t{did}\n")


        