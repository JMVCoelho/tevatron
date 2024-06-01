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

model=pythia-160m-marco-docs-bow-ct-pretrain-bs128-20pc-sample-less-negs-triplet-topk

cat /data/user_data/jmcoelho/embeddings/marco_docs/$model/20pc-sample-run-splits/less-opacus-triplet/*_topk > /data/user_data/jmcoelho/embeddings/marco_docs/$model/20pc-sample-run-splits/less-opacus-triplet/hardnegs_less_opacus.20.pc.full.topk
cat /data/user_data/jmcoelho/embeddings/marco_docs/$model/20pc-sample-run-splits/less-opacus-triplet/*_sample_t1 > /data/user_data/jmcoelho/embeddings/marco_docs/$model/20pc-sample-run-splits/less-opacus-triplet/hardnegs_less_opacus.20.pc.full.t1