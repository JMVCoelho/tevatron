#!/bin/bash
#SBATCH --job-name=rr_run
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-4-36,babel-8-3,babel-4-28


eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.3
export TRANSFORMERS_CACHE=/data/datasets/hf_cache

echo "split $1"
python scripts/re_rank_gpt_qs.py $1