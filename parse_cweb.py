import json
path_data = "/data/group_data/cx_group/query_generation_data/'Query Generation Llama.jsonl'"
path_final = "/data/user_data/jmcoelho/datasets/cweb_subset"

queries = []
pos_docs = []

qid = 0
with open(path_data, 'r') as file, \
    open(f"{path_final}/train.qrel.tsv") as o1, \
    open(f"{path_final}/query.train.jsonl") as o2:
        
        for line in file:
            data = json.loads(line)

            query = data.get("query")
            pos_doc = data.get("positive_document_id")


