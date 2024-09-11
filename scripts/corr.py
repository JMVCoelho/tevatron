import numpy as np
from scipy.stats import pearsonr, spearmanr

# Load the third column from each CSV file
floats1 = np.loadtxt('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/query_influence/run_all_grouped_with_valid', delimiter='\t', usecols=3)
floats2 = np.loadtxt('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/query_influence_grouped_gradnorm/run_all_queries_with_grad_norm', delimiter='\t', usecols=2)

# Compute the Pearson correlation coefficient
correlation, _ = pearsonr(floats1, floats2)

print("Spearmanr", correlation)