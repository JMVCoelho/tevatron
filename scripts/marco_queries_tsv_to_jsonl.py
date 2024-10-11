import json

from tqdm import tqdm
import sys

inpath = sys.argv[1]
outpath = sys.argv[2]
# converts a tsv "text_id, title, text" to jsonl {text_id, text, title}

def tsv_to_jsonl(input_tsv, output_jsonl):
    with open(input_tsv, 'r') as tsv_file, open(output_jsonl, 'w') as jsonl_file:
        for line in tqdm(tsv_file):
            line = line.strip().split('\t')
            try:
                qid, text = line
            except Exception as e:
                qid = line[0]
                text = "EMPTY"
                print("zau")
            json_data = {'query_id': qid, 'query': f"{text}"}
            jsonl_file.write(json.dumps(json_data) + '\n')

# input_file = f"/data/user_data/jmcoelho/datasets/marco/documents/{gen}.query.tsv"
# output = f"/data/user_data/jmcoelho/datasets/marco/documents/{gen}.query.jsonl"

tsv_to_jsonl(inpath, outpath)