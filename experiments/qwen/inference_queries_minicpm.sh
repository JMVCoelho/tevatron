#!/bin/bash

#SBATCH --job-name=qwen-retriever-inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --gres=gpu:6000Ada:1
#SBATCH --time=2-00:00:00

eval "$(conda shell.bash hook)"
conda activate tevatron

export TRANSFORMERS_CACHE=/data/datasets/hf_cache


trained_model_name=$1
pooling=$2
echo "Using model $trained_model_name to encode MINICPM train queries with $pooling pooling"

EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs/
mkdir $EMBEDDING_OUTPUT_DIR/$trained_model_name

OUTPUT_FILE=$EMBEDDING_OUTPUT_DIR/$trained_model_name/query-minicpm-train.pkl

if [ -f "$OUTPUT_FILE" ]; then
  echo "File $OUTPUT_FILE already exists. Skipping encoding."
else
  echo "Encoding train queries..."

  python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --model_name_or_path /data/user_data/jmcoelho/models/$trained_model_name/ \
    --dataset_cache_dir /data/datasets/hf_cache \
    --cache_dir /data/datasets/hf_cache \
    --query_prefix "" \
    --passage_prefix "" \
    --bf16 \
    --pooling $pooling \
    --append_eos_token \
    --normalize \
    --encode_is_query \
    --per_device_eval_batch_size 300 \
    --query_max_len 32 \
    --passage_max_len 512 \
    --dataset_path "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-random-20k-1gpu/mates_neg_cache_loss_100_2/all_queries.jsonl" \
    --encode_output_path $OUTPUT_FILE
fi