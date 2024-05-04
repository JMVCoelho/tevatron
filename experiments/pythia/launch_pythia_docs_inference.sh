#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=2-00:00:00

export TRANSFORMERS_CACHE=/data/datasets/hf_cache

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

trained_model_name=pythia-160m-marco-docs-bow-pretrain-bs64-self-hn1-less-normal-sample

shard=$1

echo "running for shard $shard"

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
  --per_device_eval_batch_size 300 \
  --query_max_len 32 \
  --passage_max_len 1024 \
  --dataset_path "/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.jsonl" \
  --add_markers True \
  --dataset_number_of_shards 4 \
  --dataset_shard_index ${shard} \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus.${shard}.pkl
