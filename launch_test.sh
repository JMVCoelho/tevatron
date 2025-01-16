#!/bin/bash

#SBATCH --job-name=build_dataset
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-0-19
#SBATCH --gres=gpu:6000Ada:1


eval "$(conda shell.bash hook)"
conda activate tevatron

# export HF_HOME=/data/datasets/hf_cache
# export HF_TOKEN=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

python /home/jmcoelho/tevatron/src/tevatron/retriever/cluster.py