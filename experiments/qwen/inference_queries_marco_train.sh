#!/bin/bash

#SBATCH --job-name=qwen-pretrain
# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id

#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 12 # number cpus (threads) per task

# 327680
#SBATCH --mem=15G # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit

#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1

#export TRANSFORMERS_CACHE=/data/user_data/jmcoelho/hf_cach

  # --dataset_cache_dir /data/datasets/hf_cache \
  # --cache_dir /data/datasets/hf_cache \


eval "$(conda shell.bash hook)"
conda activate cmu-llms-hw3


trained_model_name=$1
pooling=$2
echo "Using model $trained_model_name to encode MARCO train queries with $pooling pooling"

EMBEDDING_OUTPUT_DIR=/data/jcoelho/embeddings/babel/
mkdir $EMBEDDING_OUTPUT_DIR/$trained_model_name

OUTPUT_FILE=$EMBEDDING_OUTPUT_DIR/$trained_model_name/query-marco-train.pkl

if [ -f "$OUTPUT_FILE" ]; then
  echo "File $OUTPUT_FILE already exists. Skipping encoding."
else
  echo "Encoding train queries..."

  python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --model_name_or_path /user/home/jcoelho/Qwen/models/$trained_model_name/ \
    --query_prefix "" \
    --passage_prefix "" \
    --bf16 \
    --pooling $pooling \
    --append_eos_token \
    --normalize \
    --encode_is_query \
    --per_device_eval_batch_size 300 \
    --query_max_len 32 \
    --passage_max_len 1024 \
    --dataset_path "/data/jcoelho/datasets/babel/train.query.jsonl" \
    --encode_output_path $OUTPUT_FILE
fi