#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

model=pythia-160m-marco-docs-bow-pretrain 
n_negatives=9

python -m tevatron.retriever.driver.select_hard_negatives \
    --method random \
    --validation_set /data/user_data/jmcoelho/datasets/marco/documents/processed_data/pythia-160m-marco-docs-bow-pretrain/random/val.jsonl \
    --train_run_path /data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-pretrain/run.train.random.txt \
    --train_qrels /data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv \
    --embedding_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/ \
    --number_of_negatives $n_negatives \
    --negatives_out_file /data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-pretrain/random_train_run_splits/random/hardnegs_random.random.txt \
    --output_dir temp \
    --model_name_or_path /data/user_data/jmcoelho/models/fine-tuned/$model \
    --dataset_cache_dir /data/datasets/hf_cache \
    --cache_dir /data/datasets/hf_cache 