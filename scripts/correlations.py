import pandas as pd
from scipy.stats import pearsonr, spearmanr
import math

# Load the two TSV files
file1 = "/data/user_data/jmcoelho/datasets/marco/documents/qrels.gen14_0.rr.tsv"
file2 = "/data/user_data/jmcoelho/datasets/marco/documents/qrels.gen14_1.rr.tsv"
file3 = "/data/user_data/jmcoelho/datasets/marco/documents/qrels.gen14_0.lp.tsv"
file4 = "/data/user_data/jmcoelho/datasets/marco/documents/qrels.gen14_1.lp.tsv"

# Read the TSV files
df1 = pd.read_csv(file1, sep='\t')
df2 = pd.read_csv(file2, sep='\t')
df3 = pd.read_csv(file3, sep='\t')
df4 = pd.read_csv(file4, sep='\t')

# Extract the last column (scores)
scores1 = [x for x in df1.iloc[:, -1].to_list() + df2.iloc[:, -1].to_list()]
scores2 = [x for x in df3.iloc[:, -1].to_list() + df4.iloc[:, -1].to_list()]


# Compute Pearson and Spearman correlations
pearson_corr, p1 = pearsonr(scores1, scores2)
spearman_corr, p2 = spearmanr(scores1, scores2)

print(f"Pearson correlation: {pearson_corr}")
print(f"Spearman correlation: {spearman_corr}")


import matplotlib.pyplot as plt

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(scores1, scores2, alpha=0.5)
plt.title('Episode 2')
plt.xlabel('Ranker(q, d)')
plt.ylabel('log P(q | d)')

# Save the plot to a file
plt.savefig('scatter_plot.png')

# Close the plot to avoid displaying
plt.close()