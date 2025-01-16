import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt


def plot_extreme_clusters(file_path, output_pdf):
    """
    Create boxplots for clusters with highest and lowest mean loss

    Parameters:
    file_path (str): Path to TSV file with three columns (query, loss, cluster)
    output_pdf (str): Path for output PDF file
    """
    # Read TSV file without headers
    df = pd.read_csv(
        file_path, sep="\t", header=None, names=["query", "loss", "cluster"]
    )

    # Calculate mean loss for each cluster
    cluster_means = df.groupby("cluster")["loss"].mean().sort_values()

    # Get top and bottom 3 clusters
    bottom_clusters = cluster_means.head(3).index
    top_clusters = cluster_means.tail(3).index
    clusters_to_plot = list(bottom_clusters) + list(top_clusters)

    # Prepare data for plotting
    plot_data = [
        df[df["cluster"] == cluster]["loss"].values for cluster in clusters_to_plot
    ]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create box plot
    bp = ax.boxplot(plot_data, patch_artist=True)

    # Customize box colors
    for box in bp["boxes"]:
        box.set(facecolor="lightblue", alpha=0.7)

    # Add cluster labels
    ax.set_xticklabels(clusters_to_plot, rotation=45)

    # Add titles and labels
    ax.set_title("Loss Distribution for Extreme Clusters")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Loss")

    # Add a horizontal line at the median of all data
    ax.axhline(y=df["loss"].median(), color="r", linestyle="--", alpha=0.3)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save to PDF
    plt.savefig(output_pdf)
    plt.close()

    # Print summary statistics
    print("\nCluster Statistics:")
    print("-" * 50)
    for cluster in clusters_to_plot:
        cluster_data = df[df["cluster"] == cluster]
        print(f"\nCluster {cluster}:")
        print(f"Mean loss: {cluster_data['loss'].mean():.3f}")
        print(f"Sample size: {len(cluster_data)}")


def print_cluster_samples(file_path):
    """
    Read TSV file and print first 5 rows of each cluster

    Parameters:
    file_path (str): Path to TSV file with columns: query, loss, cluster
    """
    # Read TSV file
    df = pd.read_csv(
        file_path, sep="\t", header=None, names=["query", "loss", "cluster"]
    )

    # Group by cluster
    grouped = df.groupby("cluster")

    # Iterate through each group and print first 5 rows
    k = 0
    for cluster_name, group in grouped:
        print(f"\nCluster: {cluster_name}")
        print("-" * 50)
        print(group.head())
        print("\n")
        k += 1
        if k == 10:
            break


# print_cluster_samples(
#     "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-random-20k-1gpu/mates_neg_cache_loss_100_2/mates_valid_loss_clustered.tsv"
# )


plot_extreme_clusters(
    "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-random-20k-1gpu/mates_neg_cache_loss_100_2/mates_valid_loss_clustered.tsv",
    "clusters_influce_dist.pdf",
)
