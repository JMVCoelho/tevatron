from transformers import T5Model, T5Config, TrainingArguments, Trainer, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import numpy as np
import sys
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import MSELoss
from scipy.stats import pearsonr, spearmanr

from tevatron.retriever.modeling import T5FusionRegression, T5Regression, BERTRegression

import random
random.seed(17121998)
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqModelOutput,
)


#model_to_train =  "jmvcoelho/t5-base-marco-crop-pretrain-2048"
model_to_train =  "google-bert/bert-base-uncased"

train_data_path = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T1/group_level_10000_two_valid_orcale/datamodel_train_globalstd.tsv"
val_data_path =  "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T1/group_level_10000_two_valid_orcale/datamodel_val_globalstd.tsv"
test_data_path = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T1/group_level_10000_two_valid_orcale/datamodel_test_globalstd.tsv"
model_out=f"temp"


# ##### TRAIN ######
# ##################

corpus_path = "/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.tsv"
queries_path = "/data/user_data/jmcoelho/datasets/marco/documents/train.query.filtered.txt"
DID2TEXT = {}
QID2TEXT = {}

with open(corpus_path, 'r') as h:
    for line in h:
        did, title, text = line.strip().split("\t")
        DID2TEXT[did] = f'Title: {title.strip()} Text: {text.strip()}'.strip()

with open(queries_path, 'r') as h:
    for line in h:
        qid, text = line.strip().split("\t")
        QID2TEXT[qid] = text


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

        example['labels'] = torch.tensor([feature['label'] for feature in features])
        return example
    
class DatamodelDataset(Dataset):
    def __init__(self, data_files, tokenizer):
        self.train_data = load_dataset(
            "text",
            data_files=data_files,
            split="train",
        )

        #self.train_data = self.select_percentage_of_dataset(self.train_data, percentage=10)

        self.tokenizer = tokenizer

    def select_percentage_of_dataset(self, dataset, percentage):
        assert percentage > 0 and percentage <= 100, "Percentage must be between 0 and 100"
        
        num_examples = len(dataset)
        num_to_select = int(num_examples * (percentage / 100.0))
        
        indices = random.sample(range(num_examples), num_to_select)
        
        # Create a new dataset with selected examples
        selected_examples = [dataset[idx] for idx in indices]
        
        return selected_examples

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        example = self.train_data[item]['text']
        query, docs, label = example.strip().split("\t")
        docs = docs.split(",")

        tokenized_sentences = self.tokenizer(
            #[QID2TEXT[query]] + [DID2TEXT[d] for d in docs],
            [f"Query: {QID2TEXT[query]}. Document: {DID2TEXT[d]}" for d in docs],
            padding=False, 
            truncation=True,
            max_length=256,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )

      
        #tokenized_sentences['input_ids'] = [item for sublist in tokenized_sentences['input_ids'] for item in sublist[:-1]]

        tokenized_sentences['label'] = torch.tensor(float(label), dtype=torch.float)

        return tokenized_sentences
    


def compute_metrics(preds):
    p, l = preds
    mse = np.mean((p.flatten() - l.flatten()) ** 2)
    pearson_corr = pearsonr(p.flatten(), l.flatten())[0]
    spearman_corr = spearmanr(p.flatten(), l.flatten())[0]
    return {'mse': mse,
            'pearson':pearson_corr,
            'spearman_corr':spearman_corr}


tokenizer = AutoTokenizer.from_pretrained(model_to_train)


collator = TrainCollator(tokenizer)


model = BERTRegression(model_to_train, n_fusion=10) 

train_dataset = DatamodelDataset(data_files=train_data_path, tokenizer=tokenizer)
val_dataset = DatamodelDataset(data_files=val_data_path, tokenizer=tokenizer)
test_dataset = DatamodelDataset(data_files=test_data_path, tokenizer=tokenizer)


collator = TrainCollator(tokenizer)

training_args = TrainingArguments(
    output_dir=model_out,
    learning_rate=1e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=100,
    num_train_epochs=1,
    evaluation_strategy="steps",
    save_strategy="no",
    eval_steps=500,
    logging_steps=1,
    weight_decay=0.01,
    fp16=True,
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
predictions = trainer.predict(test_dataset)

print("RANDOM PRED")
print(predictions.metrics)

trainer.train() 
print("FINAL PRED")
predictions = trainer.predict(test_dataset)
print(predictions.metrics)

trainer.save_model(model_out)




