#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-4-36,babel-8-3,babel-4-28


export TRANSFORMERS_CACHE=/data/datasets/hf_cache

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

trained_model_name=$1


EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs
mkdir -p $EMBEDDING_OUTPUT_DIR/$trained_model_name

python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path /data/user_data/jmcoelho/models/fine-tuned/$trained_model_name/ \
  --dataset_cache_dir /data/datasets/hf_cache \
  --cache_dir /data/datasets/hf_cache \
  --query_prefix "" \
  --passage_prefix "" \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --encode_is_query \
  --per_device_eval_batch_size 300 \
  --query_max_len 32 \
  --passage_max_len 1024 \
  --dataset_path "/data/user_data/jmcoelho/datasets/marco/documents/1000.valid.query.jsonl" \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-val.pkl

  # --dataset_path "/data/user_data/jmcoelho/datasets/marco/documents/10.percent.sample.v3.train.query.filtered.jsonl" \
  # --encode_output_path $EMBEDDING_OUTPUT_DIR/$trained_model_name/10-percent-sample-query-train-v3.pkl





