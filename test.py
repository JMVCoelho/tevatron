# import transformers
# import torch

# state_dict = torch.load("/data/user_data/jmcoelho/models/pythia-160m-1024-marco-docs-hf/pytorch_model.bin")

# model = transformers.AutoModel.from_pretrained(
#     "/data/user_data/jmcoelho/models/pythia-160m-1024-marco-docs-hf/", local_files_only=True, state_dict=state_dict, attn_implementation="flash_attention_2"
# )

# print(model.__class__)

import pickle

with open('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs/corpus.1.pkl', 'rb') as h:
    x = pickle.load(h)

print(len(x))
print(len(x[1]))

with open('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs/corpus.2.pkl', 'rb') as h:
    x = pickle.load(h)

print(len(x))
print(len(x[1]))

with open('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs/corpus.3.pkl', 'rb') as h:
    x = pickle.load(h)

print(len(x))
print(len(x[1]))

with open('/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs/corpus.0.pkl', 'rb') as h:
    x = pickle.load(h)

print(len(x))
print(len(x[1]))