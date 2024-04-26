#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=2-00:00:00

export TRANSFORMERS_CACHE=/data/datasets/hf_cache

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

trained_model_name=pythia-160m-marco-passage-bow-pretrain-distil-llama-3-score

EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_passage
mkdir $EMBEDDING_OUTPUT_DIR/$trained_model_name

python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path /data/user_data/jmcoelho/models/fine-tuned/$trained_model_name/ \
  --dataset_cache_dir /data/datasets/hf_cache \
  --cache_dir /data/datasets/hf_cache \
  --query_prefix "Query: " \
  --passage_prefix "Passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --encode_is_query \
  --per_device_eval_batch_size 128 \
  --query_max_len 32 \
  --passage_max_len 156 \
  --dataset_name Tevatron/msmarco-passage \
  --dataset_split train \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-train.pkl

python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path /data/user_data/jmcoelho/models/fine-tuned/$trained_model_name/ \
  --dataset_cache_dir /data/datasets/hf_cache \
  --cache_dir /data/datasets/hf_cache \
  --query_prefix "Query: " \
  --passage_prefix "Passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --encode_is_query \
  --per_device_eval_batch_size 128 \
  --query_max_len 32 \
  --passage_max_len 156 \
  --dataset_name Tevatron/msmarco-passage \
  --dataset_split dev \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-dev.pkl

python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path /data/user_data/jmcoelho/models/fine-tuned/$trained_model_name/ \
  --dataset_cache_dir /data/datasets/hf_cache \
  --cache_dir /data/datasets/hf_cache \
  --query_prefix "Query: " \
  --passage_prefix "Passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --encode_is_query \
  --per_device_eval_batch_size 128 \
  --query_max_len 32 \
  --passage_max_len 156 \
  --dataset_name Tevatron/msmarco-passage \
  --dataset_split dl19 \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-dl19.pkl


python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path /data/user_data/jmcoelho/models/fine-tuned/$trained_model_name/ \
  --dataset_cache_dir /data/datasets/hf_cache \
  --cache_dir /data/datasets/hf_cache \
  --query_prefix "Query: " \
  --passage_prefix "Passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --encode_is_query \
  --per_device_eval_batch_size 128 \
  --query_max_len 32 \
  --passage_max_len 156 \
  --dataset_name Tevatron/msmarco-passage \
  --dataset_split dl20 \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-dl20.pkl