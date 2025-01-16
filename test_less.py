# import pickle
# import torch
# import io
# from huggingface_hub import login
# from datasets import load_dataset


# class CPU_Unpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name == '_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#         else: return super().find_class(module, name)

# all_scores = []
# for i in range(4):
#     with open(f"rerank_minicpm_dataset_{i}", 'rb') as h:
#         shard_scores = CPU_Unpickler(h).load()
#         all_scores.extend([score.item() for score in shard_scores])


# login("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")

# data_files = [f"en_{str(i).zfill(2)}.jsonl" for i in range(24)]  # Adjust range if there are more/less files

# dataset = load_dataset("XBKYS/minicpm-embedding-data", data_files=data_files, split="train")

# assert len(dataset) == len(all_scores)

# dataset = dataset.add_column("bge_ranker_scores", all_scores)
# print(dataset)

# # dataset.save_to_disk("/data/jcoelho/datasets/minicpm_embedding_unsup_queries")

# sorted_dataset = dataset.sort("bge_ranker_scores", reverse=True)
# top_entries_dataset = sorted_dataset.select(range(1500000))
# print(top_entries_dataset)

# top_entries_dataset.save_to_disk("/data/jcoelho/datasets/minicpm_embedding_unsup_queries_1.5M_filtered")


# from datasets import load_from_disk

# # Load the dataset
# dataset = load_from_disk("/data/jcoelho/datasets/minicpm_embedding_unsup_queries")

# # Filter to keep only entries with bge_ranker_scores > 0
# filtered_dataset = dataset.filter(lambda x: x['bge_ranker_scores'] > 0)

# # Print the number of entries that meet the condition
# print(f"Number of entries with bge_ranker_scores > 0: {len(filtered_dataset)}")

# filtered_dataset.save_to_disk("/data/jcoelho/datasets/minicpm_embedding_unsup_queries_2.1M_filtered")

from huggingface_hub import login
from datasets import load_dataset

less_scores = []
grad_norms = []
less_scores_file = "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-random-20k-1gpu//less_neg_cache/dot_prods_neg_cache.tsv"
with open(less_scores_file, "r") as h:
    for line in h:
        q, less, grad = line.strip().split("\t")
        less_scores.append(float(less))
        grad_norms.append(float(grad))


data_files = [f"en_{str(i).zfill(2)}.jsonl" for i in range(24)]
dataset = load_dataset(
    "XBKYS/minicpm-embedding-data", data_files=data_files, split="train"
)
print(len(dataset))
print(len(less_scores))

dataset = dataset.add_column("less_scores", less_scores)


less_only = dataset.sort("less_scores", reverse=True).select(range(80000))
print(less_only[0]["less_scores"])
print(less_only[-1]["less_scores"])
less_only.save_to_disk(
    "/data/user_data/jmcoelho/datasets/minicpm_embedding_unsupervised_queries/less_nc_denoised_80ksubset"
)

# # print("did less only")

# denoised = dataset.filter(lambda x: x["grad_norms"] <= 5)
# print("done filter")
# less_scores_list = sorted(denoised["less_scores"], reverse=True)
# less_denoised = denoised.filter(
#     lambda x: x["less_scores"] >= less_scores_list[360000 - 1]
# )
# print(len(less_denoised))
# less_denoised.save_to_disk(
#     "/data/user_data/jmcoelho/datasets/minicpm_embedding_unsupervised_queries/less_nc_denoised"
# )
# print("did denoised less")
