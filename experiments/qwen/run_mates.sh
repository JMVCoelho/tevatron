#!/bin/bash

#SBATCH --job-name=qwen-retriever-train
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=preempt
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-14-1,babel-13-13,babel-13-21,babel-13-29,babel-4-17


eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-12.4

export HF_HOME=/data/datasets/hf_cache
export HF_TOKEN=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

shard=$1

model_to_use=Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-minicpmembed-random-20k-synth-only-8gpu-6negs
out_path=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_use/mates_neg_cache_loss_100_6_100k_synth_only_5_q_per_doc_1e-4lr_baseset2/

mkdir -p $out_path 


python scripts/mates_influence.py \
    --output_dir $out_path/DELETE_mates_valid_loss.tsv \
    --model_name_or_path "/data/user_data/jmcoelho/models/$model_to_use" \
    --bf16 \
    --pooling avg \
    --append_eos_token \
    --learning_rate 1e-4 \
    --normalize \
    --temperature 0.01 \
    --query_max_len 32 \
    --passage_max_len 512 \
    --dataset_number_of_shards 32 \
    --dataset_shard_index $shard \
    --dataset_cache_dir /data/datasets/hf_cache \
    --cache_dir /data/datasets/hf_cache
    

