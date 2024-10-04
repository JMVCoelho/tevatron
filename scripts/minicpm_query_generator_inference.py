from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import sys

N_Q_PER_DOC = 10
CHUNK = int(sys.argv[1])
CHUNK_SIZE = 100000
TOTAL_DOCS = 800000

def get_chunk(data, chunk_number):
    start_index = chunk_number * CHUNK_SIZE
    end_index = start_index + CHUNK_SIZE
    return data[start_index:end_index]
class QueryGenerator():
    def __init__(self, path="/data/user_data/jmcoelho/models/query_generators/minicpm-2b-stf-bf6-gpt4-query-generator/"):

        self.total_len = 512
        self.completion_len = 32
        self.llm = LLM(model=path, 
                       trust_remote_code=True, 
                       max_model_len=self.total_len)
        
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        self.sampling_params = SamplingParams(temperature=1.0, 
                                              top_p=0.9, 
                                              top_k=50,
                                              n=N_Q_PER_DOC,
                                              repetition_penalty=1.2)

    def apply_prompt(self, t):
        t = self.tokenizer.decode(self.tokenizer.encode(t)[1:self.total_len-self.completion_len])
        return f"<用户>Generate a query for this document: {t}.<AI>"
    
    def inference(self, documents):
        
        queries = []
        documents = [self.apply_prompt(d) for d in tqdm(documents)]
        outputs = self.llm.generate(documents, self.sampling_params)

        for output in outputs:
            for sub_out in output.outputs:
                generated_text = sub_out.text
                queries.append(generated_text)

        return queries


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
            if len(good_doc_ids) == TOTAL_DOCS:
                break

documents = [corpus[doc_id] for doc_id in good_doc_ids]
documents = get_chunk(documents, CHUNK)


query_generator = QueryGenerator()

queries = query_generator.inference(documents)

q_id = int(CHUNK*CHUNK_SIZE) * int(N_Q_PER_DOC)
good_doc_ids = [item for item in get_chunk(good_doc_ids, CHUNK) for _ in range(N_Q_PER_DOC)]

print(len(queries))
print(len(good_doc_ids))

with open(f"/data/user_data/jmcoelho/datasets/marco/documents/gen12_{CHUNK}.query.tsv", 'w', encoding='utf-8') as out_1, open(f"/data/user_data/jmcoelho/datasets/marco/documents/qrels.gen12_{CHUNK}.tsv", 'w', encoding='utf-8') as out_2:
    for doc_id, query in zip(good_doc_ids, queries):
        query = query.strip().replace('\t', '').replace('\n', '').replace('\r', '')
        try:
            out_1.write(f"{q_id}\t{query}\n")
        except:
            continue
        out_2.write(f"{q_id}\tQ0\t{doc_id}\t1\n")
        q_id += 1

print("Done")