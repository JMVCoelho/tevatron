#!/bin/bash
#SBATCH --job-name=gen_queries
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:6000Ada:1
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=babel-4-36,babel-8-3,babel-4-28


eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.3
export TRANSFORMERS_CACHE=/data/datasets/hf_cache
python scripts/generate_queries.py