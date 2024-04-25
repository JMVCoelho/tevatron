#!/bin/bash
#SBATCH --job-name=pythia_dr
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:6000Ada:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --time=2-00:00:00

export TRANSFORMERS_CACHE=/data/user_data/hf_cache

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

trained_model_name=pythia-160m-marco-docs-bow-pretain-contrastive-pretrain-bs64

deepspeed --include localhost:0,1,2,3 --master_port 26500 --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir /data/user_data/jmcoelho/models/fine-tuned/$trained_model_name \
  --dataset_cache_dir /data/datasets/hf_cache \
  --cache_dir /data/datasets/hf_cache \
  --model_name_or_path "/data/user_data/jmcoelho/models/pre-trained/pythia-160m-1024-marco-docs-bow-contrastive-pretrain/" \
  --dataset_path "/data/user_data/jmcoelho/datasets/marco/documents/processed_data/pythia-160m-marco-documents-2048/train/train.jsonl" \
  --save_steps 1000 \
  --bf16 \
  --pooling eos \
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
