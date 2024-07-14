import random
import pickle


random.seed(17121998)

TRAIN_SIZE = 1600
TEST_VAL_SIZE = 200

log_path = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1/group_level_10000_two_valid_orcale_momentum_single_query_2k"

def standardize_list(input_list):
    mu = sum(input_list) / len(input_list)
    sigma = (sum((x - mu) ** 2 for x in input_list) / len(input_list)) ** 0.5
    
    standardized_list = [(x - mu) / sigma for x in input_list]
    
    return standardized_list

def to_binary_labels(lst):
    smallest_values = sorted(lst)[:2]
    
    for i in range(len(lst)):
        if lst[i] in smallest_values:
            lst[i] = 1
        else:
            lst[i] = 0
            
    return lst



QID2POS  = {}

with open("/data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv", 'r') as h:

    for line in h:
        qid, q0, did, rel = line.strip().split("\t")

        if qid not in QID2POS:
            QID2POS[qid] = []
        
        QID2POS[qid].append(did)


all_labels = []
k = 0
with open(f"{log_path}/datamodel_train.tsv", 'w') as out1, \
      open(f"{log_path}/datamodel_val.tsv", 'w') as out2, \
        open(f"{log_path}/datamodel_test.tsv", 'w') as out3:
            
            with open(f"{log_path}/group_hardnegs_0_log.pkl", 'rb') as h:
                data = pickle.load(h)

                count1 = 0  # for out1 (train)
                count2 = 0  # for out2 (val)
                

                #GLOBAL STD
                # for query in data:
                #     samples = data[query]
                #     labels = [float(s[1]) for s in samples]
                #     for l in labels:
                #         all_labels.append(l)

                # all_labels = standardize_list(all_labels)
                # num_sublists = len(all_labels) // 5
                # list_of_lists = [all_labels[i*5:(i+1)*5] for i in range(num_sublists)]

                # for query, labels in zip(data, list_of_lists):
                #     samples = data[query]
                #     document_ids = [s[0] for s in samples] # negative docs

                #     assert len(document_ids) == len(labels)

                #     for doc_ids, label in zip(document_ids, labels):
                        
                #         doc_ids.insert(0, QID2POS[query][0]) # add positive

                #         if count1 < TRAIN_SIZE:
                #             out_file = out1
                #             count1 += 1
                #         elif count2 < TEST_VAL_SIZE:
                #             out_file = out2
                #             count2 += 1
                #         else:
                #             out_file = out3
                
                #         out_file.write(f"{query}\t{','.join(doc_ids)}\t{label}\n")


                #QLEVEL STD

                for query in data:
                    samples = data[query]
                    labels = standardize_list([float(s[1]) for s in samples])
                    document_ids = [s[0] for s in samples] # negative docs

                    for doc_ids, label in zip(document_ids, labels):
                        
                        doc_ids.insert(0, QID2POS[query][0]) # add positive

                        if count1 < TRAIN_SIZE:
                            out_file = out1
                            count1 += 1
                        elif count2 < TEST_VAL_SIZE:
                            out_file = out2
                            count2 += 1
                        else:
                            out_file = out3
                
                        out_file.write(f"{query}\t{','.join(doc_ids)}\t{label}\n")

            
                # Binary cls 

                # for query in data:
                #     samples = data[query]
                #     labels = to_binary_labels([float(s[1]) for s in samples])
                #     document_ids = [s[0] for s in samples] # negative docs

                #     for doc_ids, label in zip(document_ids, labels):
                        
                #         doc_ids.insert(0, QID2POS[query][0]) # add positive

                #         if count1 < TRAIN_SIZE:
                #             out_file = out1
                #             count1 += 1
                #         elif count2 < TEST_VAL_SIZE:
                #             out_file = out2
                #             count2 += 1
                #         else:
                #             out_file = out3
                
                #         out_file.write(f"{query}\t{','.join(doc_ids)}\t{label}\n")