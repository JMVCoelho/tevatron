from datasets import load_dataset, load_from_disk, concatenate_datasets
import torch
import numpy as np

# losses = []
# less_scores_file = "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-minicpmembed-random-20k-8gpu/mates_neg_cache_loss_100_6_subset_corrected/mates_valid_loss.tsv"
# with open(less_scores_file, "r") as h:
#     for line in h:
#         q, loss, _ = line.strip().split("\t")
#         losses.append(float(loss))

# print(losses[:10])

# losses = np.array(losses)

# rng = np.random.default_rng()
# gumbel_noise = rng.gumbel(size=len(losses))
# noisy_losses = losses + gumbel_noise
# print(noisy_losses[:10])


# data_files = [f"en_{str(i).zfill(2)}.jsonl" for i in range(24)]
# dataset = load_dataset(
#     "XBKYS/minicpm-embedding-data", data_files=data_files, split="train"
# )

# print(len(losses))
# dataset = load_from_disk(
#     "/data/user_data/jmcoelho/datasets/minicpm_embedding_unsupervised_queries/random_subset"
# )
# print(len(dataset))

# dataset = dataset.add_column("loss", losses)
# sorted_dataset = dataset.sort("loss", reverse=False).select(range(100000))

# print(sorted_dataset[0]["loss"])
# print(sorted_dataset[-1]["loss"])

# sorted_dataset = sorted_dataset.shuffle(seed=42)

# sorted_dataset.save_to_disk(
#     "/data/user_data/jmcoelho/datasets/minicpm_embedding_unsupervised_queries/mates_neg_cache_loss_100_6_subset_corrected"
# )
# print("did denoised less")

# loss file:
q2loss = {}

path_1 = "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-minicpmembed-random-20k-8gpu/marco_all_queries_loss/mates_valid_loss.tsv"
dataset1 = load_from_disk(
    "/data/user_data/jmcoelho/datasets/minicpm_embedding_unsupervised_queries/train_marco_full"
)

with open(path_1, "r") as h:
    for line in h:
        q, loss, _ = line.strip().split("\t")
        q2loss[q.strip()] = float(loss)

losses = list(q2loss.values())

# Step 2: Initialize the random number generator
rng = np.random.default_rng()

# Step 3: Generate multidimensional Gumbel noise (with same length as losses)
gumbel_noise = rng.gumbel(size=len(losses))

# Step 4: Add the Gumbel noise to the losses
noisy_losses = np.array(losses) + gumbel_noise  # This will work if `losses` are numeric

# Step 5: Update the dictionary with noisy values
noisy_text_to_loss = {
    query: noisy_loss for query, noisy_loss in zip(q2loss.keys(), noisy_losses)
}


def add_loss_value(example):
    query_text = example["query"][1].strip()
    loss = noisy_text_to_loss.get(query_text, None)
    assert loss is not None, f"Not found: {query_text}"
    example["loss"] = loss
    return example


dataset1 = dataset1.map(add_loss_value)


dataset1 = dataset1.sort("loss", reverse=False).select(range(100000))

dataset1.save_to_disk(
    "/data/user_data/jmcoelho/datasets/minicpm_embedding_unsupervised_queries/marco_mates_gumbel_100k"
)

exit()


# path_1 = "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-minicpmembed-random-20k-8gpu/mates_neg_cache_loss_100_6_subset_corrected/mates_valid_loss.tsv"
# dataset1 = load_from_disk(
#     "/data/user_data/jmcoelho/datasets/minicpm_embedding_unsupervised_queries/random_subset"
# )

q2loss2 = {}
path_2 = "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-minicpmembed-random-20k-8gpu/mates_neg_cache_loss_100_6_subset_corrected/mates_valid_loss.tsv"
dataset2 = load_from_disk(
    "/data/user_data/jmcoelho/datasets/minicpm_embedding_unsupervised_queries/random_subset"
)

with open(path_2, "r") as h:
    for line in h:
        q, loss, _ = line.strip().split("\t")
        q2loss2[q.strip()] = float(loss)

losses = list(q2loss2.values())


def add_loss_value(example):
    query_text = example["query"][1].strip()
    loss = q2loss2.get(query_text, None)
    assert loss is not None, f"Not found: {query_text}"
    example["loss"] = loss
    return example


dataset2 = dataset2.map(add_loss_value)


merged_dataset = concatenate_datasets([dataset1, dataset2])

mixed = merged_dataset.sort("loss", reverse=False).select(range(100000))


mixed.save_to_disk(
    "/data/user_data/jmcoelho/datasets/minicpm_embedding_unsupervised_queries/mix_marco_minicpm_100_6_100k_best"
)


# losses2 = []
# path2 = "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-minicpmembed-random-20k-8gpu/marco_all_queries_loss/mates_valid_loss.tsv"
# dataset2 = load_from_disk(
#     "/data/user_data/jmcoelho/datasets/minicpm_embedding_unsupervised_queries/train_marco_full"
# )
# with open(path2, "r") as h:
#     for line in h:
#         q, loss, _ = line.strip().split("\t")
#         losses2.append(float(loss))

# print(len(losses2))
# print(dataset2[0]["query"])
# print(dataset2[-1]["query"])
# print(losses2[0])
# print(losses2[-1])

# print(dataset1)
# print(dataset2)
# exit()

# smaller = dataset.select(range(60 * 8 * 2 * 3))

# print(smaller[0]["loss"])
# print(smaller[1]["loss"])
# print(smaller[-2]["loss"])
# print(smaller[-1]["loss"])

# smaller.save_to_disk(
#     "/data/user_data/jmcoelho/datasets/minicpm_embedding_unsupervised_queries/mates_neg_cache_loss_100_6_small_subset"
# )
# print("did denoised less")
