# -*- coding: utf-8 -*-
import json
from dataclasses import dataclass, field
from typing import Dict, Optional
from trl import DPOTrainer, DPOConfig
import torch
import transformers
from transformers import (AutoModelForCausalLM, AutoTokenizer)
from datasets import Dataset
import pandas as pd

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-2B-sft-bf16")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

PATH_TO_TRIPLETS = f"/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/gen12-shnegs/dpo-data-ids-5.tsv"
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

with open(f"{PATH_TO_RUN}/gen12.query.tsv", 'r') as h:
    for line in h:
        qid, text = line.split("\t")
        QID2TEXT[qid] = text

def prompt_mapper(did):
    return f'<用户>Generate a query for this document: {DID2TEXT[did]}.<AI>' 


def generation_mapper(qid):
    return QID2TEXT[qid]

def create_hf_dataset_from_tsv(tsv_file):
    df = pd.read_csv(tsv_file, sep='\t')
    hf_dataset = Dataset.from_pandas(df)
    return hf_dataset


def load_model_and_tokenizer(
    model_path: str,
):
    """load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    return model, ref_model, tokenizer


if __name__ == "__main__":
    

    OUTPATH = "/data/user_data/jmcoelho/models/query_generators/minicpm-2b-stf-bf6-gpt4-query-generator-dpo-e1/"
    model, ref_model, tokenizer = load_model_and_tokenizer(
        model_path="/data/user_data/jmcoelho/models/query_generators/minicpm-2b-stf-bf6-gpt4-query-generator/",
    )

    dataset = create_hf_dataset_from_tsv(PATH_TO_TRIPLETS)

      
    dataset = dataset.map(lambda row: {'prompt': prompt_mapper(str(row['prompt'])),
                                    'chosen': generation_mapper(str(row['chosen'])),
                                    'rejected': generation_mapper(str(row['rejected']))})


    train_test_split = dataset.train_test_split(test_size=0.1)

    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']


    args = DPOConfig(
        output_dir=OUTPATH,             
        num_train_epochs=1,                     
        per_device_train_batch_size=4,         
        per_device_eval_batch_size=4,           
        gradient_accumulation_steps=2,          
        gradient_checkpointing=True,            
        learning_rate=1e-5,                     
        warmup_ratio=0.1,                       
        lr_scheduler_type="cosine",             
        logging_steps=1,                       
        save_steps=0,                            
        evaluation_strategy="steps",            
        eval_steps=250,     
        max_length = 512,
        max_prompt_length = 480,
        max_completion_length = 32,                    
        report_to="wandb",                      
        run_name="dpo_test",
        deepspeed="/home/jmcoelho/tevatron/deepspeed/ds_zero2_config.json"
    )

    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    dpo_trainer.train()
    dpo_trainer.save_model()