#!/bin/bash

#SBATCH --job-name=qwen-retriever-train
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --gres=gpu:6000Ada:1
#SBATCH --time=2-00:00:00


eval "$(conda shell.bash hook)"
conda activate tevatron

export HF_HOME=/data/datasets/hf_cache
export HF_TOKEN=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

shard=$1

model_to_use=Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-random-20k-1gpu
out_path=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_use/less/

mkdir -p $out_path 


python scripts/less_influence.py \
    --output_dir $out_path/dot_prods.tsv \
    --model_name_or_path "/data/user_data/jmcoelho/models/$model_to_use" \
    --bf16 \
    --pooling wavg \
    --append_eos_token \
    --normalize \
    --temperature 0.01 \
    --query_max_len 32 \
    --passage_max_len 512 \
    --dataset_number_of_shards 8 \
    --dataset_shard_index $shard \
    --dataset_cache_dir /data/datasets/hf_cache \
    --cache_dir /data/datasets/hf_cache
    

