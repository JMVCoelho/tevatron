#!/bin/bash
#SBATCH --job-name=pythia_dr
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --time=2-00:00:00

export TRANSFORMERS_CACHE=/data/datasets/hf_cache

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

trained_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs64-10pc-sample-random-negs-weighted-unif
model_to_train=/data/user_data/jmcoelho/models/pre-trained/pythia-160m-1024-marco-docs-bow-contrastive-pretrain/

# base, 1st epoch data
#/data/user_data/jmcoelho/datasets/marco/documents/processed_data/pythia-160m-marco-documents-2048/train/train.jsonl

# 2nd epoch data
#/data/user_data/jmcoelho/datasets/marco/documents/processed_data/pythia-160m-marco-docs-bow-pretrain/gradnorm_cluster/train.jsonl" \

# 2nd epoch model
#model_to_train=/data/user_data/jmcoelho/models/fine-tuned/pythia-160m-marco-docs-bow-pretrain/

deepspeed --include localhost:0 --master_port 24500 --module tevatron.retriever.driver.train_weighted \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir /data/user_data/jmcoelho/models/fine-tuned/$trained_model_name \
  --model_name_or_path $model_to_train\
  --dataset_path /data/user_data/jmcoelho/datasets/marco/documents/processed_data/pythia-160m-1024-marco-docs-bow-contrastive-pretrain/random_negs_weighted_queries/train.jsonl \
  --dataset_cache_dir /data/datasets/hf_cache \
  --cache_dir /data/datasets/hf_cache \
  --save_steps 1000 \
  --bf16 \
  --pooling eos \
  --loss contrastive \
  --gradient_checkpointing \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 64 \
  --train_group_size 10 \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 1024 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 2 \
  --report_to wandb \
  --run_name $trained_model_name
