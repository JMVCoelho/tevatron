# from tqdm import tqdm


# dids_in_qrels = set()
# with open("/data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv", 'r') as h:
#     for line in h:
#         qid, q0, did, rel = line.strip().split("\t")
#         dids_in_qrels.add(did)

# with open("/data/user_data/jmcoelho/datasets/marco/documents/qrels.dev.tsv", 'r') as h:
#     for line in h:
#         qid, q0, did, rel = line.strip().split("\t")
#         dids_in_qrels.add(did)

# good_doc_ids = []
# with open("/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.tsv", 'r') as h:
#     for line in tqdm(h):
#         did, title, text = line.split("\t")
#         if did not in dids_in_qrels:
#             good_doc_ids.append(did)


# print(len(good_doc_ids))


from datasets import load_dataset
from huggingface_hub import login

# Login with your Hugging Face token (replace with your token)

# Load the dataset from Hugging Face repository (or local directory)
dataset = load_dataset("XBKYS/minicpm-embedding-data", cache_dir="/data/datasets/hf_cache", data_files={
    'train': [f'en_{i:02d}.jsonl' for i in range(23)]
})

# Inspect the loaded dataset