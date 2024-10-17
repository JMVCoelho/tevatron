#!/bin/bash

#SBATCH --job-name=qwen-pretrain
# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id

#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 12 # number cpus (threads) per task

# 327680
#SBATCH --mem=50G # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit

#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1

#export TRANSFORMERS_CACHE=/data/user_data/jmcoelho/hf_cache

eval "$(conda shell.bash hook)"
conda activate cmu-llms-hw3

shard=$1
trained_model_name=$2

echo "Using model $2 to encode shard $shard of corpus"

EMBEDDING_OUTPUT_DIR=/data/jcoelho/embeddings/babel/
mkdir $EMBEDDING_OUTPUT_DIR/$trained_model_name


  # --dataset_cache_dir /data/datasets/hf_cache \
  # --cache_dir /data/datasets/hf_cache \

python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path /user/home/jcoelho/Qwen/models/$trained_model_name/ \
  --query_prefix "" \
  --passage_prefix "" \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --per_device_eval_batch_size 600 \
  --query_max_len 32 \
  --passage_max_len 512 \
  --dataset_path "/data/jcoelho/datasets/babel/corpus_firstp_2048.jsonl" \
  --add_markers True \
  --dataset_number_of_shards 4 \
  --dataset_shard_index ${shard} \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus.${shard}.pkl
