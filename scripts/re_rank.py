import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

PATH_TO_RUN = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision"
subset = sys.argv[1]
qid2text = {}
did2text = {}

did2text = {}
with open("/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.tsv", 'r') as h:
    for line in h:
        did, title, text = line.split("\t")
        did2text[did] = f"{title} {text}"

with open("/data/user_data/jmcoelho/datasets/marco/documents/gen.query.tsv", 'r') as h:
    for line in h:
        qid, text = line.split("\t")
        qid2text[qid] = text

pairs = []

with open(f"{PATH_TO_RUN}/run.gen.all.queries.{subset}", 'r') as in_file:
    for line in in_file:
        qid, did, _ = line.strip().split()

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


with open(f"{PATH_TO_RUN}/run.gen.all.queries.{subset}", 'r') as in_file, \
    open(f"{PATH_TO_RUN}/run.gen.all.queries.{subset}.reranked", 'w') as out_file: 
        for line, new_score in zip(in_file, all_scores):
            qid, did, old_score = line.strip().split()

            out_file.write(f"{qid}\t{did}\t{old_score}\t{new_score}\n")