from datasets import Dataset
from transformers import AutoTokenizer
import json
from tqdm import tqdm


# Path to your JSONL file and tokenizer name
jsonl_file = "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-finetune-ep1/pretokenized/val_shuf_subset.jsonl"
tokenizer_name = "/data/user_data/jmcoelho/models/Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-finetune-ep1"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Initialize lists for storing decoded data
data = {"query": [], "pos": [], "neg": []}

# Read JSONL file line by line and decode
with open(jsonl_file, "r") as f:
    for line in tqdm(f):
        item = json.loads(line.strip())

        # Decode the query, positives, and negatives
        query_text = tokenizer.decode(item["query"], skip_special_tokens=True)
        pos_texts = [
            tokenizer.decode(p, skip_special_tokens=True) for p in item["positives"]
        ]
        neg_texts = [
            tokenizer.decode(n, skip_special_tokens=True) for n in item["negatives"]
        ]

        # Append decoded text to the data dictionary
        data["query"].append(["prompt", query_text])
        data["pos"].append(["prompt"] + pos_texts)
        data["neg"].append(["prompt"] + neg_texts)

# Create a Hugging Face Dataset from the decoded data
dataset = Dataset.from_dict(data)

# Verify the dataset structure
print(dataset)


dataset.save_to_disk(
    "/data/user_data/jmcoelho/datasets/minicpm_embedding_unsupervised_queries/valid_marco_2500"
)
