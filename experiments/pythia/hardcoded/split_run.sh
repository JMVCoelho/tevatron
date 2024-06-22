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


model=pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-valid-5-group-level-best

split -d -a 2 -l 2293900 /data/user_data/jmcoelho/embeddings/marco_docs/$model/run.train.txt /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.

mv /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.00 /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.0
mv /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.01 /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.1
mv /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.02 /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.2
mv /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.03 /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.3
mv /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.04 /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.4
mv /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.05 /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.5
mv /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.06 /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.6
mv /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.07 /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.7
mv /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.08 /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.8
mv /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.09 /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.9