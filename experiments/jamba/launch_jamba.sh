#!/bin/bash
#SBATCH --job-name=jamba-retriever
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A100_80GB:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --exclude=babel-4-28,babel-1-27,babel-8-11
#SBATCH --time=2-00:00:00

export TRANSFORMERS_CACHE=/data/datasets/hf_cache

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

trained_model_name=jamba-52b-marco-docs-lora

WORLD_SIZE=4 torchrun --nproc_per_node=4 --master_port=1234 --module tevatron.retriever.driver.train \
  --output_dir /data/user_data/jmcoelho/models/$trained_model_name \
  --model_name_or_path ai21labs/Jamba-v0.1 \
  --dataset_cache_dir /data/datasets/hf_cache \
  --lora \
  --lora_target_modules embed_tokens,x_proj,in_proj,out_proj \
  --save_steps 10000 \
  --dataset_path "/data/user_data/jmcoelho/datasets/marco/documents/processed_data/jamba-marco-documents-2048/train.jsonl" \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 2 \
  --train_group_size 5 \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 2048 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 4 \
  --report_to wandb \
  --run_name $trained_model_name