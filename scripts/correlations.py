import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Load the two TSV files
file1 = "/data/user_data/jmcoelho/datasets/marco/documents/qrels.gen10.rr.tsv"
file2 = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision-1gpu/gen10_less_influence_v2/run_all_queries_with_valid_grad_dot_p"

# Read the TSV files
df1 = pd.read_csv(file1, sep='\t')
df2 = pd.read_csv(file2, sep='\t')

# Extract the last column (scores)
scores1 = df1.iloc[:, -1]
scores2 = df2.iloc[:, -1]

# Compute Pearson and Spearman correlations
pearson_corr, _ = pearsonr(scores1, scores2)
spearman_corr, _ = spearmanr(scores1, scores2)

print(f"Pearson correlation: {pearson_corr}")
print(f"Spearman correlation: {spearman_corr}")