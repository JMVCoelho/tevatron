from datasets import load_dataset

keep_vals = []
less_scores_file = "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-random-20k-1gpu//mates_neg_cache/mates_loss_neg_cache_processed.tsv"
with open(less_scores_file, "r") as h:
    for line in h:
        _, _, _, keep = line.strip().split("\t")
        keep_vals.append(int(keep))


data_files = [f"en_{str(i).zfill(2)}.jsonl" for i in range(24)]
dataset = load_dataset(
    "XBKYS/minicpm-embedding-data", data_files=data_files, split="train"
)

print(len(keep_vals))
print(len(dataset))

dataset = dataset.add_column("keep", keep_vals)

print("added cols")


filtered = dataset.filter(lambda x: x["keep"] == 1)

print(len(filtered))
filtered.save_to_disk(
    "/data/user_data/jmcoelho/datasets/minicpm_embedding_unsupervised_queries/mates_15q_6d_multiple_valid"
)
print("did denoised less")
