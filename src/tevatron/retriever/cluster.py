import faiss
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.metrics import silhouette_score
import torch
import torch.nn.functional as F


import torch
import torch.nn.functional as F


class KMeansTuner:
    def __init__(self, gpu: bool = True):
        """
        Initialize KMeans tuner.

        Args:
            gpu: Whether to use GPU acceleration
        """
        self.use_gpu = gpu

    def compute_objective(self, vectors: np.ndarray, k: int) -> float:
        """
        Compute KMeans objective (sum of squared distances) for a given k.
        """
        vectors = vectors.astype(np.float32)
        kmeans = faiss.Kmeans(
            d=vectors.shape[1],
            k=k,
            niter=20,
            gpu=self.use_gpu,
            spherical=True,
        )

        vectors = vectors.astype(np.float32)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            gpu_index = faiss.GpuIndexFlatL2(
                res, vectors.shape[1], cfg
            )  # GpuIndexFlatL2 ?
            kmeans.index = gpu_index

        kmeans.train(vectors)

        # Get distances to nearest centroids
        distances, _ = kmeans.index.search(vectors, 1)

        # Return average squared distance
        return float(np.mean(distances**2))

    def find_optimal_k(
        self, vectors: np.ndarray, k_range: List[int], plot: bool = True
    ) -> Tuple[int, List[float]]:
        """
        Find optimal k using elbow method.

        Args:
            vectors: Input vectors
            k_range: List of k values to try
            plot: Whether to plot the elbow curve

        Returns:
            Tuple of (optimal k, list of objectives for each k)
        """
        print("Computing objectives for different k values...")
        objectives = []

        for k in k_range:
            print(f"Testing k={k}")
            obj = self.compute_objective(vectors, k)
            objectives.append(obj)

        # Find elbow point using the kneedle algorithm
        optimal_k = self._find_elbow(k_range, objectives)

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(k_range, objectives, "b-", marker="o")
            plt.axvline(
                x=optimal_k, color="r", linestyle="--", label=f"Elbow at k={optimal_k}"
            )
            plt.xlabel("Number of Clusters (k)")
            plt.ylabel("Average Squared Distance to Centroid")
            plt.title("Elbow Method for Optimal k")
            plt.legend()
            plt.grid(True)
            plt.show()

        return optimal_k, objectives

    def _find_elbow(self, k_range: List[int], objectives: List[float]) -> int:
        """
        Find the elbow point using the kneedle algorithm.
        """
        # Normalize the curves
        x = np.array(k_range)
        y = np.array(objectives)

        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

        # Find the point of maximum distance from line
        npoints = len(k_range)
        all_coords = np.vstack((x, y)).T

        # Line from first to last point
        line_vec = all_coords[-1] - all_coords[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))

        # Distance from points to line
        vec_from_first = all_coords - all_coords[0]
        scalar_prod = np.sum(vec_from_first * line_vec_norm, axis=1)
        vec_from_line = vec_from_first - scalar_prod[:, None] * line_vec_norm

        # Find the elbow
        distances = np.sqrt(np.sum(vec_from_line**2, axis=1))
        elbow_idx = np.argmax(distances)

        return k_range[elbow_idx]


class SimpleKMeans:
    def __init__(self, n_clusters: int, use_gpu: bool):
        """
        Initialize simple KMeans clustering.

        Args:
            n_clusters: Number of clusters to create
        """
        self.n_clusters = n_clusters
        self.use_gpu = use_gpu
        self.clusters = None  # Will store list of document indices for each cluster

    def fit(self, vectors: np.ndarray) -> List[np.ndarray]:
        """
        Cluster the input vectors and return document indices for each cluster.

        Args:
            vectors: Input vectors of shape (n_docs, dim)

        Returns:
            List where each element contains indices of documents in that cluster
        """
        kmeans = faiss.Kmeans(
            d=vectors.shape[1],
            k=self.n_clusters,
            niter=20,
            gpu=self.use_gpu,
            spherical=True,
        )

        vectors = vectors.astype(np.float32)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            gpu_index = faiss.GpuIndexFlatL2(
                res, vectors.shape[1], cfg
            )  # GpuIndexFlatL2 ?
            kmeans.index = gpu_index

        kmeans.train(vectors)

        assignments = kmeans.assign(vectors)[1]
        self.assignments = assignments
        print(len(self.assignments))

        # Group document indices by cluster
        self.clusters = [[] for _ in range(self.n_clusters)]
        for doc_idx, cluster_id in enumerate(assignments):
            self.clusters[cluster_id].append(doc_idx)

        # Convert to list of lists for easier handling
        self.clusters = [sorted(cluster) for cluster in self.clusters]

        return self.clusters

    def get_cluster(self, cluster_id: int) -> np.ndarray:
        """
        Get document indices for a specific cluster.

        Args:
            cluster_id: ID of the cluster (0 to n_clusters-1)

        Returns:
            Array of document indices belonging to this cluster
        """
        if self.clusters is None:
            raise ValueError("Must call fit() before getting clusters")
        return self.clusters[cluster_id]


import pickle


def pickle_load(path):
    with open(path, "rb") as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup


if __name__ == "__main__":
    import json

    # Load your vectors
    path_to_vecs = "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-random-20k-1gpu/query-minicpm-train.pkl"
    q_reps, q_lookup = pickle_load(path_to_vecs)

    # tuner = KMeansTuner(gpu=True)

    # # Define range of k values to test
    # k_range = [1000, 1500, 2000, 2500, 3000, 3500, 4000]

    # # Find optimal k
    # optimal_k, objectives = tuner.find_optimal_k(q_reps, k_range, plot=True)

    # print(f"Optimal number of clusters: {optimal_k}")
    # exit()

    query_dict = {}

    # Read and parse each line
    with open(
        "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-random-20k-1gpu/mates_neg_cache_loss_100_2/all_queries.jsonl",
        "r",
    ) as file:
        for line in file:
            data = json.loads(line)
            query_dict[int(data["query_id"])] = data["query"]

    # Initialize KMeans
    n_clusters = 2000
    kmeans = SimpleKMeans(n_clusters=n_clusters, use_gpu=True)

    # Fit and get clusters
    print("Starting clustering...")
    clusters = kmeans.fit(q_reps)

    # Print some statistics
    print("\nClustering Results:")
    print(f"Number of clusters: {len(clusters)}")

    # Print size of first 5 clusters
    print("\nFirst 5 clusters sizes:")
    for i in range(min(5, len(clusters))):
        print(f"Cluster {i}: {len(clusters[i])} documents")

    # Print some example documents from first cluster
    print("\nExample documents from first cluster:")
    first_cluster = clusters[0]
    for idx in first_cluster[:5]:  # first 5 documents in first cluster
        print(
            f"Document {idx}: {query_dict[int(q_lookup[idx])] if q_lookup else 'No lookup available'}"
        )

    with open(
        "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-random-20k-1gpu/mates_neg_cache_loss_100_2/mates_valid_loss.tsv",
        "r",
    ) as file, open(
        "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-random-20k-1gpu/mates_neg_cache_loss_100_2/mates_valid_loss_clustered.tsv",
        "w",
    ) as outfile:
        for line, cluster in zip(file, kmeans.assignments):
            query, loss, group = line.strip().split("\t")

            outfile.write(f"{query}\t{loss}\t{cluster}\n")
