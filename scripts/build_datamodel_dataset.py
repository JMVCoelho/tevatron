import random
import pickle


random.seed(17121998)


log_path = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1/group_level_10000_two_valid_orcale_momentum_600_query_2k"

def standardize_list(input_list):
    mu = sum(input_list) / len(input_list)
    sigma = (sum((x - mu) ** 2 for x in input_list) / len(input_list)) ** 0.5
    
    standardized_list = [(x - mu) / sigma for x in input_list]
    
    return standardized_list


QID2POS  = {}

with open("/data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv", 'r') as h:

    for line in h:
        qid, q0, did, rel = line.strip().split("\t")

        if qid not in QID2POS:
            QID2POS[qid] = []
        
        QID2POS[qid].append(did)


all_labels = []

nq = 0
            
for i in range(12):
    with open(f"{log_path}/group_hardnegs_{i}_log.pkl", 'rb') as h:
        data = pickle.load(h)
        #QLEVEL STD

        for query in data:
            with open(f"{log_path}/datamodel_data_q{nq}.tsv", 'w') as out1:
                samples = data[query]
                labels = standardize_list([float(s[1]) for s in samples])
                document_ids = [s[0] for s in samples] # negative docs

                for doc_ids, label in zip(document_ids, labels):
                    doc_ids.insert(0, QID2POS[query][0]) # add positive
                    out1.write(f"{query}\t{','.join(doc_ids)}\t{label}\n")

                nq += 1