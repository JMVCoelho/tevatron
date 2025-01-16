import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from datasets import load_from_disk
import json

import math

from huggingface_hub import login

dataset = load_from_disk(
    "/data/user_data/jmcoelho/datasets/minicpm_embedding_unsupervised_queries/llama_generated_clean_Qwen2.5-0.5B-bidirectional-attn-mntp_all_queries_except_warmup",
)

pairs = []

for example in tqdm(dataset):
    query = example["query"][1]
    positive = example["pos"][1]
    pairs.append((query, positive))

print(pairs[0])

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
model = model.to("cuda")
model.eval()


def batch(data, batch_size=500):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


all_scores = []
with torch.no_grad():
    for sample in tqdm(batch(pairs, 500), total=math.ceil(len(pairs) / 500)):
        inputs = tokenizer(
            sample, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        sample_scores = (
            model(**inputs, return_dict=True)
            .logits.view(
                -1,
            )
            .float()
            .cpu()
            .tolist()
        )
        all_scores.extend(sample_scores)


dataset = dataset.add_column("re_ranker_score", all_scores)
dataset.save_to_disk(
    "/data/user_data/jmcoelho/datasets/minicpm_embedding_unsupervised_queries/llama_generated_clean_Qwen2.5-0.5B-bidirectional-attn-mntp_all_queries_except_warmup_with_rr_score"
)
