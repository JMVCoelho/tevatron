path = "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-minicpmembed-random-20k-8gpu-RANDOM-80K/negatives.train.txt"
scores = "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-minicpmembed-random-20k-8gpu/marco_all_queries_loss/mates_valid_loss.tsv"
save = "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-minicpmembed-random-20k-8gpu-RANDOM-80K/negatives.train.mates.80k.best.gumbel.txt"
save_inter = "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-minicpmembed-random-20k-8gpu-RANDOM-80K/negatives.train.mates.80k.best.gumbel.test.txt"
import pandas as pd
import numpy as np

# File paths for the TSV files
x_file = path
y_file = scores
output_file = save
intermediate_file = save_inter

# Read the TSV files
x_data = pd.read_csv(x_file, sep="\t", header=None)  # Assuming no header
y_data = pd.read_csv(y_file, sep="\t", header=None)  # Assuming no header

# Ensure the columns are correctly indexed
x_lines = len(x_data)
y_lines = len(y_data)

if x_lines != y_lines:
    raise ValueError(
        f"X and Y must have the same number of lines. X: {x_lines}, Y: {y_lines}"
    )

# Get the indices of the top 80,000 scores
losses = y_data[1]
rng = np.random.default_rng()
gumbel_noise = rng.gumbel(size=len(losses))
noisy_losses = losses + gumbel_noise
top_indices = noisy_losses.nsmallest(80000).index  # Column 1 is the scores

# Select the corresponding lines from X and their scores from Y
x_prime = x_data.iloc[top_indices]
scores = y_data.iloc[top_indices]

# Combine X' with the scores for inspection
x_prime_with_scores = pd.concat(
    [x_prime.reset_index(drop=True), scores[1].reset_index(drop=True)], axis=1
)

# Save an intermediate version with a smaller subset for inspection
intermediate_subset = x_prime_with_scores.head(10)  # Inspect the top 10 rows
intermediate_subset.to_csv(intermediate_file, sep="\t", index=False, header=False)
print(f"Intermediate subset saved to {intermediate_file}")

# Save the full subset to a new TSV file (without the scores)
x_prime.to_csv(output_file, sep="\t", index=False, header=False)
print(f"Full subset saved to {output_file}")
