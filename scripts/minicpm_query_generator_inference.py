from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import sys

print(sys.argv)

N_Q_PER_DOC = 1
CHUNK = int(sys.argv[1])
MODEL= sys.argv[2]
GEN=sys.argv[3]
SKIP=int(sys.argv[4])-1 #input is in terms of episode; ep2 skips 1 set of documents.

#="/data/user_data/jmcoelho/models/query_generators/minicpm-2b-stf-bf6-gpt4-query-generator/
CHUNK_SIZE = 350000
TOTAL_DOCS = 2800000

QUERY_OUT_PATH = f"/data/user_data/jmcoelho/datasets/marco/documents/{GEN}_{CHUNK}.query.tsv"
QREL_OUT_PATH = f"/data/user_data/jmcoelho/datasets/marco/documents/qrels.{GEN}_{CHUNK}.tsv"
QREL_LP_OUT_PATH = f"/data/user_data/jmcoelho/datasets/marco/documents/qrels.{GEN}_{CHUNK}.lp.tsv"

print(f"Writing queries to: {QUERY_OUT_PATH}")
print(f"Writing qrels to: {QREL_OUT_PATH}")
print(f"Skipping {SKIP} set of {TOTAL_DOCS} elegible documents")

def get_chunk(data, chunk_number):
    start_index = chunk_number * CHUNK_SIZE
    end_index = start_index + CHUNK_SIZE
    return data[start_index:end_index]

class QueryGenerator():
    def __init__(self, path):

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
                                              repetition_penalty=1.2,
                                              logprobs=1)
        
        self.sampling_params_greedy = SamplingParams(temperature=0.0, 
                                              top_k=1, 
                                              n=1,
                                              logprobs=1)

    def apply_prompt(self, t):
        t = self.tokenizer.decode(self.tokenizer.encode(t)[1:self.total_len-self.completion_len])
        return f"<用户>Generate a query for this document: {t}.<AI>"
    
    def inference(self, documents):
        
        queries = []
        log_probs = []
        documents = [self.apply_prompt(d) for d in tqdm(documents)]
        outputs = self.llm.generate(documents, self.sampling_params_greedy)


        for output in outputs:
            for sub_out in output.outputs:
                generated_text = sub_out.text
                log_prob = sub_out.cumulative_logprob
                queries.append(generated_text)
                log_probs.append(float(log_prob))

        return queries, log_probs


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
    doc_counter = 0
    for line in h:
        did, title, text = line.split("\t")
        if did not in dids_in_qrels:
            
            if SKIP >= 1:
                doc_counter += 1  
                if doc_counter <= TOTAL_DOCS*SKIP:
                    continue

            good_doc_ids.append(did)
            corpus[did] = f"{title} {text}"
            if len(good_doc_ids) == TOTAL_DOCS:
                break

print(f"Starting doc: {good_doc_ids[0]}")
documents = [corpus[doc_id] for doc_id in good_doc_ids]
documents = get_chunk(documents, CHUNK)


query_generator = QueryGenerator(path=MODEL)

queries, log_probs = query_generator.inference(documents)

q_id = int(CHUNK*CHUNK_SIZE) * int(N_Q_PER_DOC)
good_doc_ids = [item for item in get_chunk(good_doc_ids, CHUNK) for _ in range(N_Q_PER_DOC)]

print(len(queries))
print(len(good_doc_ids))

with open(QUERY_OUT_PATH, 'w', encoding='utf-8') as out_1, \
    open(QREL_OUT_PATH, 'w', encoding='utf-8') as out_2, \
    open(QREL_LP_OUT_PATH, 'w', encoding='utf-8') as out_3:
    for doc_id, query, log_prob in zip(good_doc_ids, queries, log_probs):
        query = query.strip().replace('\t', '').replace('\n', '').replace('\r', '')
        try:
            out_1.write(f"{q_id}\t{query}\n")
        except:
            continue
        out_2.write(f"{q_id}\tQ0\t{doc_id}\t1\n")
        out_3.write(f"{q_id}\tQ0\t{doc_id}\t1\t{log_prob}\n")
        q_id += 1

print("Done")