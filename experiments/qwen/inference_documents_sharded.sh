#!/bin/bash

#SBATCH --job-name=qwen-retriever-inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=preempt
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:L40S:8
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-13-13,babel-13-17,babel-14-37,babel-6-9,babel-7-9,babel-3-21,babel-13-25,babel-13-1,babel-14-1,babel-12-9,babel-4-9

eval "$(conda shell.bash hook)"
conda activate tevatron

export TRANSFORMERS_CACHE=/data/datasets/hf_cache

module load cuda-12.4

trained_model_name=$1
pooling=$2

EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs/
mkdir $EMBEDDING_OUTPUT_DIR/$trained_model_name


echo "Using model $trained_model_name to encode MARCO corpus with $pooling pooling"


for shard in {0..7}; do
  (
  OUTPUT_FILE=$EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus.${shard}.pkl

  if [ -f "$OUTPUT_FILE" ]; then
    echo "File $OUTPUT_FILE already exists. Skipping shard $shard."
  else
    echo "Encoding shard $shard..."

    CUDA_VISIBLE_DEVICES=$shard python -m tevatron.retriever.driver.encode \
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
      --per_device_eval_batch_size 600 \
      --query_max_len 32 \
      --passage_max_len 512 \
      --dataset_path "/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.jsonl" \
      --add_markers True \
      --dataset_number_of_shards 8 \
      --dataset_shard_index ${shard} \
      --encode_output_path $OUTPUT_FILE
  fi
  ) &
done

wait