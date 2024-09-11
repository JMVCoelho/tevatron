#!/bin/bash
#SBATCH --job-name=datamodel_train
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --gres=gpu:6000Ada:1
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-4-28,babel-8-11

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

python scripts/train_datamodel_regression.py $1 $2