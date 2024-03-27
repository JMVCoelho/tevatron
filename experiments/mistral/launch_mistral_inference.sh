#!/bin/bash
#SBATCH --job-name=mistral_embed_docs
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=2-00:00:00

export TRANSFORMERS_CACHE=/data/user_data/jmcoelho/hf_cache

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

trained_model_name=mistral-7b-marco-passage-lora

shard=$1

echo "running for shard $shard"

EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_passage
mkdir $EMBEDDING_OUTPUT_DIR/$trained_model_name

python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --lora_name_or_path /data/user_data/jmcoelho/models/$trained_model_name \
  --lora \
  --query_prefix "Query: " \
  --passage_prefix "Passage: " \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --per_device_eval_batch_size 128 \
  --query_max_len 32 \
  --passage_max_len 156 \
  --dataset_name Tevatron/msmarco-passage-corpus \
  --dataset_number_of_shards 8 \
  --dataset_shard_index $shard \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus.$shard.pkl

