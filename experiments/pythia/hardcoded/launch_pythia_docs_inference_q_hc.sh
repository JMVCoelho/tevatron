#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:6000Ada:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --time=2-00:00:00

export TRANSFORMERS_CACHE=/data/datasets/hf_cache

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-12.1

trained_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-generator-marco-greedy-RR-ep1
prefix=fine-tuned

EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs
mkdir -p $EMBEDDING_OUTPUT_DIR/$trained_model_name

python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path /data/user_data/jmcoelho/models/$prefix/$trained_model_name/ \
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
  --dataset_path "/data/user_data/jmcoelho/datasets/marco/documents/gen17.query.rr.sample.jsonl" \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-gen17-rr-sample.pkl




