#!/bin/bash

#SBATCH --job-name=qwen-retriever-inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --time=2-00:00:00


eval "$(conda shell.bash hook)"
conda activate tevatron

if [ -e "$3" ]; then
    echo "Output file $3 already exists -- negatives have been already sampled."
    exit 0
fi

if [ -z "$5" ]; then
    python scripts/hn_mining_ids.py --qrels_path "$1" --run_path "$2" --out_path "$3" --n "$4"
else
    python scripts/hn_mining_ids.py --qrels_path "$1" --run_path "$2" --out_path "$3" --n "$4" --prev_run_path "$5"
fi