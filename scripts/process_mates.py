import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import seaborn as sns


def process_file(input_path, output_path, total_ones=360000):
    # Read the TSV file without header
    df = pd.read_csv(input_path, sep="\t", header=None, names=["text", "loss", "group"])

    # Calculate how many 1s each group should have
    unique_groups = df["group"].unique()
    ones_per_group = math.ceil(total_ones / len(unique_groups))

    # Initialize the binary column with zeros
    df["binary"] = 0

    # For each group, set the lowest losses to 1 while maintaining original order
    for group in tqdm(unique_groups):
        # Create boolean mask for current group
        group_mask = df["group"] == group

        # Get loss values for current group with their original indices
        group_losses = df.loc[group_mask, "loss"]

        # Find indices of lowest losses without changing their order
        # We use Series.nsmallest which returns indices in order of appearance
        lowest_losses_idx = group_losses.nsmallest(ones_per_group).index

        # Set binary values to 1 for lowest losses
        df.loc[lowest_losses_idx, "binary"] = 1

    # Verify the total number of 1s
    total_assigned = df["binary"].sum()
    print(f"Total 1s assigned: {total_assigned}")
    print(f"1s per group: {ones_per_group}")
    print(f"Number of groups: {len(unique_groups)}")

    # Save the result, maintaining original order
    df.to_csv(output_path, sep="\t", header=False, index=False)

    for group in unique_groups[:5]:
        group_data = df[df["group"] == group].copy()

        group_data = group_data.sort_values("loss", ascending=True)

        print(group_data.head(5))
        print(group_data.tail(5))
        print("======")

    return df


losses = []


def plot_kde(infile):
    losses = []

    # Read losses from the input file
    k = 0
    with open(infile, "r") as h:
        for line in h:
            q, loss, x = line.strip().split("\t")

            # other_losses = x.strip().split(",")
            # losses.append(float(other_losses[14]))
            if float(loss) < 5:
                losses.append(float(loss))
            if float(loss) < 2.5:
                k += 1

    # Plot the KDE
    plt.figure(figsize=(8, 6))
    sns.kdeplot(losses, shade=True, color="blue"),  # bw_adjust=1.1)
    plt.title("KDE Plot of Losses")
    plt.xlabel("Loss")
    plt.ylabel("Density")

    # Save the plot to a PDF file
    plt.savefig("loss_kde2345_marco.pdf", format="pdf")

    print(k)


def plot_box(infile):
    with open(infile, "r") as h:
        for line in h:
            q, loss, _ = line.strip().split("\t")
            losses.append(float(loss))

        # Plot the box plot
        plt.figure(figsize=(8, 6))
        plt.boxplot(losses)
        plt.title("Box Plot of Losses")
        plt.ylabel("Loss")

        # Save the plot to a PDF file
        plt.savefig("loss_boxplot.pdf", format="pdf")


# Example usage
if __name__ == "__main__":
    input_path = "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-random-20k-1gpu//mates_neg_cache/mates_loss_neg_cache.tsv"
    # output_path = "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-random-20k-1gpu//mates_neg_cache/mates_loss_neg_cache_processed.tsv"
    # result_df = process_file(input_path, output_path)

    plot_kde(
        "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-minicpmembed-random-20k-8gpu/marco_all_queries_loss_14/mates_valid_loss.tsv"
    )

    # plot_kde(
    #     "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-minicpmembed-random-20k-8gpu/mates_neg_cache_loss_100_6_subset_corrected/mates_valid_loss.tsv"
    # )
