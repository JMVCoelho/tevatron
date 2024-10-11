import ast
import re
import json

from tqdm import tqdm

data_out_list_train = []
data_out_list_eval = []

DID2DOC = {}
QID2QUERY = {}
with open("/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.tsv", 'r') as f_corpus, \
        open("/data/user_data/jmcoelho/datasets/marco/documents/train.query.txt", 'r') as f_queries:
     
    for line in tqdm(f_corpus):
        did, title, text = line.strip().split("\t")
        DID2DOC[did] = f'Title: {title.strip()} Text: {text.strip()}'.strip()

    for line in tqdm(f_queries):
        qid, text = line.strip().split("\t")
        QID2QUERY[qid] = f'{text.strip()}'


k = 0
with open("/data/group_data/cx_group/query_generation_data/MARCO/marco_data_for_minicpm_train.json", 'w') as out1, \
    open("/data/group_data/cx_group/query_generation_data/MARCO/marco_data_for_minicpm_eval.json", 'w') as out2, \
    open("/data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv", 'r') as f_qrels:
    for line in tqdm(f_qrels):
        qid, _, did, _ = line.strip().split()
        query = QID2QUERY[qid]
        doc_pos = DID2DOC[did]

        query = query.replace('\n', '').replace('\r','').replace('\t', '').replace('\"', '').replace('\'', '')
        doc_pos = doc_pos.replace('\n', '').replace('\r','').replace('\t', '').replace('\"', '').replace('\'', '')

        if query == "" or doc_pos == "":
            continue 

        data_out = {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Generate a query for this document: {doc_pos}.",
                        },
                        {
                            "role": "assistant",
                            "content": query,
                        },
                    ]
                }
        
        if k < 360000:
            data_out_list_train.append(data_out)
        else:
            data_out_list_eval.append(data_out)
        
        k+=1

    json.dump(data_out_list_train, out1, ensure_ascii=False, indent=4)
    json.dump(data_out_list_eval, out2, ensure_ascii=False, indent=4)
