import random
import pickle
import numpy as np


random.seed(17121998)

TEMPERATURES=[0.1, 1]

log_path = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T1/group_level_10000_two_valid_orcale"

def softmax(weights):
    exp_weights = np.exp(weights - np.max(weights))  
    return exp_weights / exp_weights.sum()

for temp in TEMPERATURES:
    with open(f"{log_path}/group_hardnegs_softmax_t{temp}", "w") as out:
        for i in range(16):
            with open(f"{log_path}/group_hardnegs_{i}_log.pkl", 'rb') as h:
                data = pickle.load(h)

                for query in data:
                    samples = data[query]
                    probabilities = softmax([-float(s[1])/temp for s in samples])

                    random_sample_index = np.random.choice(len(samples), p=probabilities)
                    
                    negs = samples[random_sample_index][0]

                    out.write(f"{query}\t{','.join(negs)}\n")
