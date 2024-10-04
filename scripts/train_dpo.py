

import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig
#from trl.trainer.utils import DPODataCollatorWithPadding
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

retriever = "pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision"
gen = "gen10"
perc = "5"

intial_generator = "/data/user_data/jmcoelho/models/query_generators/pilet5-large-gpt-query-gen-512"
dpod_generator = "/data/user_data/jmcoelho/models/query_generators/pilet5-large-gpt-query-gen-512-dpo-ep1"

path_to_triplets = f"/data/user_data/jmcoelho/embeddings/marco_docs/{retriever}/{gen}-shnegs/dpo-data-ids-{perc}.tsv"

PATH_TO_RUN = "/data/user_data/jmcoelho/datasets/marco/documents"
QID2TEXT = {}
DID2TEXT = {}

with open(f"{PATH_TO_RUN}/corpus_firstp_2048.tsv", 'r') as h:
    for line in h:
        try:
            did, text = line.split("\t")
            DID2TEXT[did] = f"{text}"
        except Exception:
            did, title, text = line.split("\t")
            DID2TEXT[did] = f"{title} {text}"

with open(f"{PATH_TO_RUN}/gen10.query.tsv", 'r') as h:
    for line in h:
        qid, text = line.split("\t")
        QID2TEXT[qid] = text

def prompt_mapper(did):
    return f'Generate a query for this document: {DID2TEXT[did]}.' 


def generation_mapper(qid):
    return QID2TEXT[qid]

# Function to load and convert TSV to Huggingface dataset
def create_hf_dataset_from_tsv(tsv_file):
    # Load the TSV file into a pandas DataFrame
    df = pd.read_csv(tsv_file, sep='\t')
    
    # Convert the DataFrame to a Hugging Face dataset
    hf_dataset = Dataset.from_pandas(df)
    
    return hf_dataset

dataset = create_hf_dataset_from_tsv(path_to_triplets)

      
dataset = dataset.map(lambda row: {'prompt': prompt_mapper(str(row['prompt'])),
                                   'chosen': generation_mapper(str(row['chosen'])),
                                   'rejected': generation_mapper(str(row['rejected']))})


train_test_split = dataset.train_test_split(test_size=0.1)

train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

tokenizer = AutoTokenizer.from_pretrained(intial_generator, model_max_length=512)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSeq2SeqLM.from_pretrained(intial_generator)
model_ref = AutoModelForSeq2SeqLM.from_pretrained(intial_generator)



args = DPOConfig(
    output_dir=dpod_generator,             
    num_train_epochs=1,                     
    per_device_train_batch_size=32,         
    per_device_eval_batch_size=32,           
    gradient_accumulation_steps=1,          
    gradient_checkpointing=True,            
    learning_rate=1e-5,                     
    warmup_ratio=0.1,                       
    lr_scheduler_type="cosine",             
    logging_steps=1,                       
    save_steps=0,                            
    save_total_limit=2,
    is_encoder_decoder=True,                     
    evaluation_strategy="steps",            
    eval_steps=250,     
    max_length = 512,
    max_prompt_length = 480,
    max_completion_length = 32,                    
    report_to="wandb",                      
    run_name="dpo_test",
    label_pad_token_id=tokenizer.pad_token_id,
)

if args.remove_unused_columns:
    args.remove_unused_columns = False

dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()
dpo_trainer.save_model()