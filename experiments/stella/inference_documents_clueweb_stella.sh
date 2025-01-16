#!/bin/bash

#SBATCH --job-name=stella-retriever-inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=preempt
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-13-13,babel-14-37,babel-6-9,babel-7-9,babel-3-21,babel-13-25,babel-13-1,babel-14-1,babel-12-9,babel-13-17


eval "$(conda shell.bash hook)"
conda activate tevatron-xformers

export HF_HOME=/data/datasets/hf_cache

shard=$1
trained_model_name=stella_en_400M_v5
pooling=avg

echo "Using model $trained_model_name to encode shard $shard of CLUEWEB corpus with $pooling pooling"

EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs/
OUTPUT_FILE=$EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus.cweb.${shard}.pkl

mkdir $EMBEDDING_OUTPUT_DIR/$trained_model_name

if [ -f "$OUTPUT_FILE" ]; then
  echo "File $OUTPUT_FILE already exists. Skipping shard $shard."
else
  echo "Encoding shard $shard..."

  

  python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --dataset_cache_dir /data/datasets/hf_cache \
    --cache_dir /data/datasets/hf_cache \
    --model_name_or_path /data/user_data/jmcoelho/models/$trained_model_name/ \
    --query_prefix "" \
    --passage_prefix "" \
    --pooling $pooling \
    --matrioshka_linear_layer 1024 \
    --per_device_eval_batch_size 600 \
    --query_max_len 32 \
    --passage_max_len 512 \
    --dataset_path "/data/group_data/cx_group/query_generation_data/cweb_subset/original_pos_neg.jsonl" \
    --dataset_number_of_shards 32 \
    --dataset_shard_index ${shard} \
    --encode_output_path $OUTPUT_FILE

fi