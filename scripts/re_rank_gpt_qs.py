import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

PATH_TO_RUN = "/data/user_data/jmcoelho/datasets/marco/documents"
qid2text = {}
did2text = {}

SPLIT = sys.argv[1]

did2text = {}
with open(f"{PATH_TO_RUN}/corpus_firstp_2048.tsv", 'r') as h:
    for line in h:
        try:
            did, text = line.split("\t")
            did2text[did] = f"{text}"
        except Exception:
            did, title, text = line.split("\t")
            did2text[did] = f"{title} {text}"

with open(f"{PATH_TO_RUN}/gen12_{SPLIT}.query.tsv", 'r') as h:
    for line in h:
        qid, text = line.split("\t")
        qid2text[qid] = text

pairs = []

with open(f"{PATH_TO_RUN}/qrels.gen12_{SPLIT}.tsv", 'r') as in_file:
    for line in in_file:
        qid, _, did, _ = line.strip().split()

        qid_text = qid2text[qid]
        did_text = did2text[did]
        pairs.append([qid_text, did_text])

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')
model = model.to('cuda')
model.eval()

def batch(data, batch_size=100):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

all_scores = []
with torch.no_grad():
    for sample in tqdm(batch(pairs)):
        inputs = tokenizer(sample, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs = {key: value.to('cuda') for key, value in inputs.items()}
        sample_scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        all_scores.extend(sample_scores)


with open(f"{PATH_TO_RUN}/qrels.gen12_{SPLIT}.tsv", 'r') as in_file, \
    open(f"{PATH_TO_RUN}/qrels.gen12_{SPLIT}.rr.tsv", 'w') as out_file: 
        for line, new_score in zip(in_file, all_scores):
            qid, q0, did, r = line.strip().split()

            out_file.write(f"{qid}\t{q0}\t{did}\t{r}\t{new_score}\n")


# awk -F'\t' '{print $0}' /data/group_data/cx_group/query_generation_data/GPT4/qrels.gen7.rr.tsv | sort -t $'\t' -k5,5nr | head -n $(($(wc -l < /data/group_data/cx_group/query_generation_data/GPT4/qrels.gen7.rr.tsv) * 6 / 10)) > /data/group_data/cx_group/query_generation_data/GPT4/qrels.gen7.rr.60.tsv

# for perc in 30 40 50 60 70 80; do awk -F'\t' 'NR==FNR {a[$1]; next} $1 in a' /data/group_data/cx_group/query_generation_data/GPT4/re-ranker-filter/qrels.gen7.rr."$perc".tsv /data/group_data/cx_group/query_generation_data/GPT4/bm25-negatives/negs_int.tsv > /data/group_data/cx_group/query_generation_data/GPT4/bm25-negatives/negs_run_"$perc".tsv; done