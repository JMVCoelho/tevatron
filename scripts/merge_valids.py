import json

valid1 = "/data/user_data/jmcoelho/datasets/marco/documents/processed_data/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision-1gpu/random_all_queries_10k_two_valid/val_1.jsonl"
valid2 = "/data/user_data/jmcoelho/datasets/marco/documents/processed_data/pythia-160m-1024-marco-docs-bow-contrastive-pretrain/random_all_queries_10k_two_valid/val_1.jsonl"

valid_merged = "/data/user_data/jmcoelho/datasets/marco/documents/processed_data/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision-1gpu/random_all_queries_10k_two_valid/val_1_with_momentum.jsonl"

with open(valid1, 'r') as f1, open(valid2, 'r') as f2, open(valid_merged, 'w') as f3:
    for line1, line2 in zip(f1, f2):
        # Parse each line as JSON
        data1 = json.loads(line1)
        data2 = json.loads(line1)

        assert data1['query'] == data2['query']
        assert data1['positives'] == data2['positives']
        
        negs1 = data1['negatives']
        negs2 = data2['negatives']
        
        all_negs = negs1 + negs2

        new_data = {"query":data1['query'], "positives":data1['positives'], "negatives":all_negs}
        f3.write(json.dumps(new_data) + '\n')
