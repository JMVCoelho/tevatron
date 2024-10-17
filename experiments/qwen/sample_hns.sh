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




eval "$(conda shell.bash hook)"
conda activate cmu-llms-hw3

python scripts/hn_mining_ids.py --qrels_path $1 --run_path $2 --out_path $3 --n $4