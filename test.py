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


from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk("/data/jcoelho/datasets/minicpm_embedding_unsup_queries")

# Filter to keep only entries with bge_ranker_scores > 0
filtered_dataset = dataset.filter(lambda x: x['bge_ranker_scores'] > 0)

# Print the number of entries that meet the condition
print(f"Number of entries with bge_ranker_scores > 0: {len(filtered_dataset)}")

filtered_dataset.save_to_disk("/data/jcoelho/datasets/minicpm_embedding_unsup_queries_2.1M_filtered")
