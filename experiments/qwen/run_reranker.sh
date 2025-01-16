#!/bin/bash

#SBATCH --job-name=qwen-retriever-train
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-14-1,babel-13-13


eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-12.4

export HF_HOME=/data/datasets/hf_cache
export HF_TOKEN=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

shard=$1

python scripts/rerank.py #$shard
