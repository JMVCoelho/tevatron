from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
from transformers import set_seed
import math
import sys

CHUNK = int(sys.argv[1])
CHUNK_SIZE = 100000
#set_seed(17121998)

def get_chunk(data, chunk_number):
    start_index = chunk_number * CHUNK_SIZE
    end_index = start_index + CHUNK_SIZE
    return data[start_index:end_index]

class T5QueryGenerator():
    def __init__(self, base_model="t5-base", max_tokens=512, device='cuda'):
        self.max_tokens = max_tokens
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_train = self.get_model_train(base_model, self.device)
        self.tokenizer = self.get_tokenizer(base_model)

    @staticmethod
    def get_model_train(base_model, device):
        return AutoModelForSeq2SeqLM.from_pretrained(base_model).to(device)
    
    @staticmethod
    def get_tokenizer(base_model):
        return AutoTokenizer.from_pretrained(base_model)
    
    def tokenize_train(self, batch):
        texts = []
        labels = []
        for example in batch:
            document = example['doc']
            query = example['query']
            texts.append(f'Generate a query for this document: {document}.')
            labels.append(query)
            
        tokenized = self.tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=self.max_tokens)
        tokenized['labels'] = self.tokenizer(labels, return_tensors='pt', padding=True, truncation='longest_first')['input_ids']

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)

        return tokenized
    
    def tokenize_inference(self, batch):
        texts = []
        for example in batch:
            document = example
            texts.append(f'Generate a query for this document: {document}.')
            
        tokenized = self.tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=self.max_tokens)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)

        return tokenized
    

    # documents: list of strings; each string a document.
    def inference(self, documents, batchsize=300, generation_config=None):
        model_eval = self.model_train.eval()
        generation_config = generation_config if generation_config is not None else model_eval.generation_config

        def batch(X, batch_size=1):
            l = len(X)
            for idx in range(0, l, batch_size):
                yield X[idx:min(idx + batch_size, l)]

        outputs = []
        for sample in tqdm(batch(documents, batchsize), total=math.ceil(len(documents)/batchsize)):
            inputs = self.tokenize_inference(sample)
            try:
                sample_outputs = model_eval.generate(**inputs, **generation_config)
            except Exception:
                sample_outputs = model_eval.generate(**inputs, generation_config=generation_config)
            outputs.extend(sample_outputs)

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    

dids_in_qrels = set()
with open("/data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv", 'r') as h:
    for line in h:
        qid, q0, did, rel = line.strip().split("\t")
        dids_in_qrels.add(did)

with open("/data/user_data/jmcoelho/datasets/marco/documents/qrels.dev.tsv", 'r') as h:
    for line in h:
        qid, q0, did, rel = line.strip().split("\t")
        dids_in_qrels.add(did)

corpus = {}
good_doc_ids = []
with open("/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.tsv", 'r') as h:
    for line in h:
        did, title, text = line.split("\t")
        if did not in dids_in_qrels:
            good_doc_ids.append(did)
            corpus[did] = f"{title} {text}"
            if len(good_doc_ids) == 800000:
                break

documents = [corpus[doc_id] for doc_id in good_doc_ids]

documents = get_chunk(documents, CHUNK)


generation_config = {
    "num_return_sequences": 2,
    "do_sample": True,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.2
}

#generation_config = None

longp = T5QueryGenerator(base_model='/data/user_data/jmcoelho/models/query_generators/pilet5-large-gpt-query-gen-512', max_tokens=512, device='cuda')
longp_queries = longp.inference(documents, batchsize=60, generation_config=generation_config)

try:
    q_id = int(CHUNK*CHUNK_SIZE) * int(generation_config["num_return_sequences"])
    good_doc_ids = [item for item in get_chunk(good_doc_ids, CHUNK) for _ in range(generation_config["num_return_sequences"])]
except Exception:
    q_id = int(CHUNK*CHUNK_SIZE)
    good_doc_ids = [item for item in get_chunk(good_doc_ids, CHUNK)]

with open(f"/data/user_data/jmcoelho/datasets/marco/documents/gen10_{CHUNK}.query.tsv", 'w', encoding='utf-8') as out_1, open(f"/data/user_data/jmcoelho/datasets/marco/documents/qrels.gen10_{CHUNK}.tsv", 'w', encoding='utf-8') as out_2:
    for doc_id, query in zip(good_doc_ids, longp_queries):
        query = query.replace('\t', '').replace('\n', '').replace('\r', '')
        try:
            out_1.write(f"{q_id}\t{query}\n")
        except:
            continue
        out_2.write(f"{q_id}\tQ0\t{doc_id}\t1\n")
        q_id += 1

print("Done")