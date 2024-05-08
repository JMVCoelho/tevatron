#!/bin/bash
#SBATCH --job-name=pythia_dr
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00

export TRANSFORMERS_CACHE=/data/datasets/hf_cache

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

model_name=pythia-160m-marco-docs-bow-pretrain
model_to_valid=/data/user_data/jmcoelho/models/fine-tuned/$model_name
save_grad=/data/user_data/jmcoelho/embeddings/marco_docs/$model_name/valid_grads


python -m tevatron.retriever.driver.valid \
  --output_dir temp \
  --model_name_or_path $model_to_valid \
  --save_gradient_path $save_grad \
  --dataset_path /data/user_data/jmcoelho/datasets/marco/documents/processed_data/pythia-160m-marco-docs-bow-pretrain/random/val.jsonl \
  --dataset_cache_dir /data/datasets/hf_cache \
  --cache_dir /data/datasets/hf_cache \
  --bf16 \
  --pooling eos \
  --loss contrastive \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --train_group_size 10 \
  --query_max_len 32 \
  --passage_max_len 1024