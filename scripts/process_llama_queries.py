import ast
import re

qid = 0
did = 3213835
def clean_text(input_string):
    return re.sub(r'[^a-zA-Z0-9\s]', '', input_string)

with open("/data/group_data/cx_group/query_generation_data/synth_corpus.tsv", 'w') as out1, \
    open("/data/group_data/cx_group/query_generation_data/qrels.gen4.tsv", 'w') as out2, \
    open("/data/group_data/cx_group/query_generation_data/gen4.query.tsv", 'w') as out3, \
    open("/data/group_data/cx_group/query_generation_data/train.200k.subset.jsonl", 'r') as h:
    for line in h:
        data = ast.literal_eval(line)
        query = data['query']
        doc = data['positive_document']

        query = clean_text(query.replace('\n', '').replace('\r','').replace('\t', '').replace('\"', '').replace('\'', ''))

        doc = clean_text(doc.replace('\n', '').replace('\r','').replace('\t', '').replace('\"', '').replace('\'', ''))
        
        if query == "" or doc == "":
            continue 
    
        out1.write(f"{did}\t{doc}\n")
        out2.write(f"{qid}\tQ0\t{did}\t{1}\n")
        out3.write(f"{qid}\t{query}\n")

        qid += 1
        did += 1


        
