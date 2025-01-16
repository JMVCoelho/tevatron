import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress
import matplotlib.pyplot as plt


with open(
    "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-random-20k-1gpu/marco_queries_loss/mates_valid_loss.tsv",
    "r",
) as h:
    losses = []
    for line in h:
        sample = []
        q, full, all_groups = line.strip().split("\t")
        sample.append(float(full))
        all_groups = [float(x) for x in all_groups.split(",")]
        sample.extend(all_groups)

        losses.append(sample)

    x = np.array(losses).T

    a = x[0]
    b = x[15]
    pearson_corr, _ = pearsonr(a, b)
    spearman_corr, _ = spearmanr(a, b)

    print(pearson_corr, spearman_corr)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        b,
        a,
        color="blue",
        alpha=0.5,
        label=f"Pearson: {pearson_corr:.2f}, Spearman: {spearman_corr:.2f}",
    )

    # Linear regression line
    slope, intercept, _, _, _ = linregress(b, a)
    plt.plot(b, slope * b + intercept, color="red", label="Regression line")

    # Label axes
    plt.xlabel("Subset valid loss")
    plt.ylabel("Full valid loss")
    plt.legend()
    plt.title("Scatter plot of Full vs. Subset valid loss - MARCO queries")

    # Save to PDF
    plt.savefig("full_vs_subset_valid_loss_scatter2.pdf", format="pdf")
    plt.close()
