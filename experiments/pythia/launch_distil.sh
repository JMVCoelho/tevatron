#!/bin/bash
#SBATCH --job-name=pythia_dr
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=300G
#SBATCH --time=2-00:00:00

export TRANSFORMERS_CACHE=/data/datasets/hf_cache

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

teacher_model=llama3-8b-marco-passage-lora
trained_model_name=pythia-160m-marco-passage-bow-pretrain-distil-llama-3-score+embed
base_model=/data/user_data/jmcoelho/models/pre-trained/pythia-160m-1024-marco-docs-bow-hf/

deepspeed --include localhost:0 --master_port 23500 --module tevatron.retriever.driver.distil \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir /data/user_data/jmcoelho/models/fine-tuned/$trained_model_name \
  --model_name_or_path $base_model \
  --embeddings_path /data/user_data/jmcoelho/embeddings/marco_passage/$teacher_model \
  --dataset_cache_dir /data/datasets/hf_cache \
  --cache_dir /data/datasets/hf_cache \
  --save_steps 391 \
  --dataset_name Tevatron/msmarco-passage \
  --query_prefix "Query: " \
  --passage_prefix "Passage: " \
  --bf16 \
  --pooling eos \
  --gradient_checkpointing \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 256 \
  --train_group_size 16 \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 156 \
  --num_train_epochs 2 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 4 \
  --report_to wandb \
  --run_name $trained_model_name
