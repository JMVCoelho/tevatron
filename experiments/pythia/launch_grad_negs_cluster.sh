#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

python scripts/gradient_based_negatives_cluster.py