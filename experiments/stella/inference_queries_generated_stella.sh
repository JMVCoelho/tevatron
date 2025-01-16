#!/bin/bash

#SBATCH --job-name=stella-retriever-inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:6000Ada:1
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-10-17


eval "$(conda shell.bash hook)"
conda activate tevatron-xformers

export HF_HOME=/data/datasets/hf_cache

trained_model_name=stella_en_400M_v5
pooling=avg

echo "Using model $trained_model_name to encode generated queries with $pooling pooling"

EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs/

mkdir $EMBEDDING_OUTPUT_DIR/$trained_model_name



python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --dataset_cache_dir /data/datasets/hf_cache \
  --cache_dir /data/datasets/hf_cache \
  --model_name_or_path /data/user_data/jmcoelho/models/$trained_model_name/ \
  --query_prefix "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: " \
  --passage_prefix "" \
  --pooling $pooling \
  --matrioshka_linear_layer 1024 \
  --per_device_eval_batch_size 600 \
  --query_max_len 64 \
  --passage_max_len 512 \
  --encode_is_query \
  --dataset_path "/data/user_data/jmcoelho/datasets/llama_generator/softmax_score_subset.jsonl" \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-gen-softmax.pkl



