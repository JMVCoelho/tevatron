#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#fdsafdsdfasdfdsSBATCH --gres=gpu:A6000:1

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

python test.py