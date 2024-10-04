from transformers import TrainingArguments, Trainer, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import numpy as np
import sys
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score


from transformers import AutoModelForSequenceClassification

import random
random.seed(17121998)

model_to_train =  "google-bert/bert-base-uncased"

train_data_path = f"/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/llama_query_influence_mates_sample_level_v4/datamodel_full_minmax_train"
val_data_path =  f"/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/llama_query_influence_mates_sample_level_v4/datamodel_full_minmax_test"
model_out=f"temp"


# ##### TRAIN ######
# ##################

queries_path = "/data/user_data/jmcoelho/datasets/marco/documents/gen6.query.tsv"
corpus_path = "/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.tsv"
qrel_path = "/data/user_data/jmcoelho/datasets/marco/documents/qrels.gen6.tsv"
DID2TEXT = {}
QID2TEXT = {}
QID2POS = {}

with open(corpus_path, 'r') as h:
    for line in h:
        did, title, text = line.strip().split("\t")
        DID2TEXT[did] = f'Title: {title.strip()} Text: {text.strip()}'.strip()

with open(queries_path, 'r') as h:
    for line in h:
        qid, text = line.strip().split("\t")
        QID2TEXT[qid] = text

with open(qrel_path, 'r') as h:
    for line in h:
        qid,_,did,_ = line.strip().split("\t")
        QID2POS[qid] = DID2TEXT[did]



class TrainCollator:
    def __init__(self, tokenizer, remove_first_token=False):
        self.tokenizer = tokenizer
        self.remove_first_token = remove_first_token

    def __call__(self, features):
        input_ids = [{'input_ids': f['input_ids'][i]} for f in features for i in range(len(f['input_ids']))]
        #input_ids = [{'input_ids': f['input_ids']} for f in features]

        example = self.tokenizer.pad(
            input_ids,
            padding=True, 
            return_attention_mask=True,
            return_tensors='pt',
        )

        if 'label' in features[0]:
            example['labels'] = torch.tensor([feature['label'] for feature in features])

        return example
    
class DatamodelDataset(Dataset):
    def __init__(self, data_files, tokenizer):
        self.train_data = load_dataset(
            "text",
            data_files=data_files,
            split="train",
        )
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        example = self.train_data[item]['text']

        items = example.strip().split("\t")

        query, label = items
        

        tokenized_sentences = self.tokenizer(
            [f"Query: {QID2TEXT[query]}. {QID2POS[query]}"],
            padding=False, 
            truncation=True,
            max_length=512,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

        if label is not None:
            tokenized_sentences['label'] = torch.tensor(float(label), dtype=torch.float)


        return tokenized_sentences
    


def compute_metrics(preds):
    p, l = preds
    mse = np.mean((p.flatten() - l.flatten()) ** 2)
    pearson_corr = pearsonr(p.flatten(), l.flatten())[0]
    spearman_corr = spearmanr(p.flatten(), l.flatten())[0]
    r2 = r2_score(p.flatten(), l.flatten())
    return {'mse': mse,
            'pearson':pearson_corr,
            'spearman_corr':spearman_corr,
            'r2': r2}


tokenizer = AutoTokenizer.from_pretrained(model_to_train)


collator = TrainCollator(tokenizer)


model = AutoModelForSequenceClassification.from_pretrained(model_to_train, num_labels=1) 

train_dataset = DatamodelDataset(data_files=train_data_path, tokenizer=tokenizer)
val_dataset = DatamodelDataset(data_files=val_data_path, tokenizer=tokenizer)


collator = TrainCollator(tokenizer)

training_args = TrainingArguments(
    output_dir=model_out,
    learning_rate=1e-6,
    per_device_train_batch_size=20,
    per_device_eval_batch_size=800,
    num_train_epochs=1,
    evaluation_strategy="steps",
    save_strategy="no",
    eval_steps=10,
    logging_steps=1,
    weight_decay=0.01,
    fp16=False,
    run_name=f"bert-datamodel-single-valid-group"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=lambda x: compute_metrics(x),
)

trainer.train() 

preds = trainer.predict(val_dataset)
print(preds)

trainer.save_model(model_out, safe_serialization=False)



