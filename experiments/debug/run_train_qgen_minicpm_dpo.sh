#!/bin/bash
#SBATCH --job-name=gen_queries_train
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=200GB
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-4-36,babel-8-3,babel-4-28,babel-5-19

eval "$(conda shell.bash hook)"
conda activate dpo

module load cuda-12.1
#export TRANSFORMERS_CACHE=/data/datasets/hf_cache
export TRANSFORMERS_CACHE=/data/group_data/cx_group/query_generation_data/hf_cache


deepspeed --include localhost:0,1,2,3 scripts/minicpm_query_generator_train_dpo.py