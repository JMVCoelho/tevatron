from datasets import Dataset
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import random

random.seed(17121998)


# Path to your JSONL file and tokenizer name
negs = "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-minicpmembed-random-20k-8gpu-RANDOM-80K/negatives.train.txt"
qrels_f = "/data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv"
qid2text_f = (
    "/data/user_data/jmcoelho/datasets/marco/documents/train.query.filtered.txt"
)
did2text_f = "/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.tsv"

qid2pos = {}
with open(qrels_f, "r") as h:
    for line in tqdm(h):
        qid, _, did, _ = line.strip().split("\t")
        if qid not in qid2pos:
            qid2pos[qid] = []
        qid2pos[qid].append(did)

qid2text = {}
with open(qid2text_f, "r") as h:
    for line in tqdm(h):
        qid, text = line.strip().split("\t")
        if qid not in qid2text:
            qid2text[qid] = text

did2text = {}
with open(did2text_f, "r") as h:
    for line in tqdm(h):
        did, title, text = line.strip().split("\t")
        if did not in did2text:
            did2text[did] = f"{title} {text}"


data = {"query": [], "pos": [], "neg": []}

with open(negs, "r") as f:
    lines = f.readlines()

# sampled_lines = random.sample(lines, 2400)
sampled_lines = lines


for line in tqdm(sampled_lines):
    query, negs = line.strip().split()

    # Decode the query, positives, and negatives
    query_text = qid2text[query]
    pos_texts = [did2text[did] for did in qid2pos[query]]

    neg_texts = [did2text[did] for did in negs.split(",")]

    # Append decoded text to the data dictionary
    data["query"].append(["prompt", query_text])
    data["pos"].append(["prompt"] + pos_texts)
    data["neg"].append(["prompt"] + neg_texts)

# Create a Hugging Face Dataset from the decoded data
dataset = Dataset.from_dict(data)

# Verify the dataset structure
print(dataset)


dataset.save_to_disk(
    "/data/user_data/jmcoelho/datasets/minicpm_embedding_unsupervised_queries/train_marco_full"
)
