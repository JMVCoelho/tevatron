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
from sklearn.metrics import r2_score

from tevatron.retriever.modeling import T5FusionRegression, T5Regression, BERTRegression

import random
random.seed(17121998)
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqModelOutput,
)

#normalization = ["log", "boxcox", "yeojohnson", "robust", "log_std"]
normalization = sys.argv[1]
n = sys.argv[2]

model_to_train =  "jmvcoelho/t5-base-marco-crop-pretrain-2048"

#model_to_train =  "google-bert/bert-base-uncased"

train_data_path = f"/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1/group_level_10000_two_valid_orcale_momentum_600_query_2k/datamodel_train_independency_{normalization}.tsv"
val_data_path =  f"/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1/group_level_10000_two_valid_orcale_momentum_600_query_2k/datamodel_val_independency_{normalization}.tsv"
test_data_path =  f"/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1/group_level_10000_two_valid_orcale_momentum_600_query_2k/datamodel_test_independency_split{n}"
out_preds =  f"/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1/group_level_10000_two_valid_orcale_momentum_600_query_2k/datamodel_test_independency_w_preds_{n}.tsv"
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

        items = example.strip().split("\t")

        if len(items) == 4:
            query, pos, neg, label = items
        elif len(items) == 3:
            query, pos, neg = items
            label = None
        else:
            raise ValueError(f"Wrong dataset format: {example}")

        tokenized_sentences = self.tokenizer(
            #[QID2TEXT[query]] + [DID2TEXT[pos]] + [DID2TEXT[neg]],
            ["Query: " + " ".join(QID2TEXT[query].split(" ")[:32]) + " " + \
             "Positive: " + " ".join(DID2TEXT[pos].split(" ")[:400]) + " " + \
             "Negative: " + " ".join(DID2TEXT[neg].split(" ")[:400]) + " " ],
            padding=False, 
            truncation=True,
            max_length=1024,
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


model = T5Regression(model_to_train) 

train_dataset = DatamodelDataset(data_files=train_data_path, tokenizer=tokenizer)
val_dataset = DatamodelDataset(data_files=val_data_path, tokenizer=tokenizer)
test_dataset = DatamodelDataset(data_files=test_data_path, tokenizer=tokenizer)


collator = TrainCollator(tokenizer)

training_args = TrainingArguments(
    output_dir=model_out,
    learning_rate=1e-5,
    per_device_train_batch_size=20,
    per_device_eval_batch_size=800,
    num_train_epochs=1,
    evaluation_strategy="steps",
    save_strategy="no",
    eval_steps=500,
    logging_steps=1,
    weight_decay=0.01,
    fp16=True,
    run_name=f"t52k-independency-q-{normalization}"
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
preds = trainer.predict(test_dataset=test_dataset).predictions

with open(test_data_path, 'r') as h, \
    open(out_preds, 'w') as out:
        for line, pred in zip(h, preds):
            qid, pos, neg = line.strip().split()
            out.write(f"{qid}\t{neg}\t{pred[0]}\n")

trainer.save_model(model_out, safe_serialization=False)




