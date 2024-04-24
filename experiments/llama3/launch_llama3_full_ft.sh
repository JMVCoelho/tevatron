#!/bin/bash
#SBATCH --job-name=llama3-full-ft
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:L40:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00

export TRANSFORMERS_CACHE=/data/datasets/hf_cache

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

trained_model_name=llama3-8b-marco-passage

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 27500 --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir /data/user_data/jmcoelho/models/$trained_model_name \
  --model_name_or_path meta-llama/Meta-Llama-3-8B \
  --dataset_cache_dir /data/datasets/hf_cache \
  --cache_dir /data/datasets/hf_cache \
  --save_steps 1000 \
  --dataset_name Tevatron/msmarco-passage \
  --query_prefix "Query: " \
  --passage_prefix "Passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 1 \
  --gradient_checkpointing \
  --train_group_size 1 \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 156 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 4 \
  --report_to wandb \
  --run_name $trained_model_name