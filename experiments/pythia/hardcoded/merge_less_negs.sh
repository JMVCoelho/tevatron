#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=babel-8-3,babel-11-25

export TRANSFORMERS_CACHE=/data/datasets/hf_cache
eval "$(conda shell.bash hook)"
conda activate tevatron
module load cuda-11.8

model=pythia-160m-1024-marco-docs-bow-contrastive-pretrain

cat /data/user_data/jmcoelho/embeddings/marco_docs/$model/group_level_10000_two_valid_orcale_top200/*_best > /data/user_data/jmcoelho/embeddings/marco_docs/$model/group_level_10000_two_valid_orcale_top200/group_hardnegs_full_best
cat /data/user_data/jmcoelho/embeddings/marco_docs/$model/group_level_10000_two_valid_orcale_top200/*_worst > /data/user_data/jmcoelho/embeddings/marco_docs/$model/group_level_10000_two_valid_orcale_top200/group_hardnegs_full_worst


python /home/jmcoelho/tevatron/scripts/build_run_from_sampling_log.py 