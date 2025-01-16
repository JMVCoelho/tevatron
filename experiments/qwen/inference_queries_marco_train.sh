#!/bin/bash

#SBATCH --job-name=qwen-retriever-inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-13-13,babel-14-37,babel-6-9,babel-7-9,babel-3-21,babel-13-25,babel-13-1,babel-14-1,babel-12-9


eval "$(conda shell.bash hook)"
conda activate tevatron

export TRANSFORMERS_CACHE=/data/datasets/hf_cache

module load cuda-12.4

trained_model_name=$1
pooling=$2
echo "Using model $trained_model_name to encode MARCO train queries with $pooling pooling"

EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs/
mkdir $EMBEDDING_OUTPUT_DIR/$trained_model_name

OUTPUT_FILE=$EMBEDDING_OUTPUT_DIR/$trained_model_name/query-marco-train.pkl

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
    --dataset_path "/data/user_data/jmcoelho/datasets/marco/documents/train.query.jsonl" \
    --encode_output_path $OUTPUT_FILE
fi