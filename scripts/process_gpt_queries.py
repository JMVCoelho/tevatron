import ast
import re

# qid = 0
# did = 0
# def clean_text(input_string):
#     return re.sub(r'[^a-zA-Z0-9\s]', '', input_string)

# with open("/data/group_data/cx_group/query_generation_data/GPT4/synth_corpus.tsv", 'w') as out1, \
#     open("/data/group_data/cx_group/query_generation_data/GPT4/qrels.gen7.tsv", 'w') as out2, \
#     open("/data/group_data/cx_group/query_generation_data/GPT4/gen7.query.tsv", 'w') as out3, \
#     open("/data/group_data/cx_group/query_generation_data/GPT4/negs_run.tsv", 'w') as out4, \
#     open("/data/group_data/cx_group/query_generation_data/GPT4/gpt4.jsonl", 'r') as h:
#     for line in h:
#         data = ast.literal_eval(line)
#         query = data['user_query']
#         doc_pos = data['metadata']['doc_a']
#         doc_neg = data['metadata']['doc_b']

#         query = query.replace('\n', '').replace('\r','').replace('\t', '').replace('\"', '').replace('\'', '')

#         doc_pos = doc_pos.replace('\n', '').replace('\r','').replace('\t', '').replace('\"', '').replace('\'', '')
#         doc_neg = doc_neg.replace('\n', '').replace('\r','').replace('\t', '').replace('\"', '').replace('\'', '')
        
#         if query == "" or doc_pos == "" or doc_neg == "":
#             continue 
    
#         out1.write(f"{did}\t{doc_pos}\n")
#         out1.write(f"{did+1}\t{doc_neg}\n")
#         out2.write(f"{qid}\tQ0\t{did}\t{1}\n")
#         out3.write(f"{qid}\t{query}\n")
#         out4.write(f"{qid}\t{did+1}\n")

#         qid += 1
#         did += 2


# qid = 0
# with open("/data/group_data/cx_group/query_generation_data/GPT4/qrels.gen7.cwebids.tsv", 'w') as out2, \
#     open("/data/group_data/cx_group/query_generation_data/GPT4/gpt4.jsonl", 'r') as h:
#     for line in h:
#         data = ast.literal_eval(line)
#         doc_pos_id = data['metadata']['logging']['doc_a_id']
#         doc_neg_id = data['metadata']['logging']['doc_b_id']

#         query = data['user_query']
#         doc_pos = data['metadata']['doc_a']
#         doc_neg = data['metadata']['doc_b']
#         query = query.replace('\n', '').replace('\r','').replace('\t', '').replace('\"', '').replace('\'', '')
#         doc_pos = doc_pos.replace('\n', '').replace('\r','').replace('\t', '').replace('\"', '').replace('\'', '')
#         doc_neg = doc_neg.replace('\n', '').replace('\r','').replace('\t', '').replace('\"', '').replace('\'', '')
#         if query == "" or doc_pos == "" or doc_neg == "":
#             continue 

#         out2.write(f"{qid}\tQ0\t{doc_pos_id}\t{1}\n") 
#         out2.write(f"{qid}\tQ0\t{doc_neg_id}\t{0}\n") 

#         qid += 1



    # 转换为 ChatML 格式

import json
data_out_list_train = []
data_out_list_eval = []

k = 0
with open("/data/group_data/cx_group/query_generation_data/GPT4/gpt_data_for_minicpm_train.json", 'w') as out1, \
    open("/data/group_data/cx_group/query_generation_data/GPT4/gpt_data_for_minicpm_eval.json", 'w') as out2, \
    open("/data/group_data/cx_group/query_generation_data/GPT4/gpt4.jsonl", 'r') as h:
    for line in h:
        data = ast.literal_eval(line)
        query = data['user_query']
        doc_pos = data['metadata']['doc_a']
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
        
        if k < 80000:
            data_out_list_train.append(data_out)
        else:
            data_out_list_eval.append(data_out)
        
        k+=1

    json.dump(data_out_list_train, out1, ensure_ascii=False, indent=4)
    json.dump(data_out_list_eval, out2, ensure_ascii=False, indent=4)
