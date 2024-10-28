#!/bin/bash

#SBATCH --job-name=scripts
# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id

#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 64 # number cpus (threads) per task

# 327680
#SBATCH --mem=100G # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit

#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1

eval "$(conda shell.bash hook)"
conda activate cmu-llms-hw3

#python scripts/parse_minicpm_queries.py 
python scripts/rerank.py $1
