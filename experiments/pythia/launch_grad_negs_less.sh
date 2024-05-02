#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=1-00:00:00

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

model=pythia-160m-marco-docs-bow-pretrain 
n_negatives=9


python -m tevatron.retriever.driver.select_hard_negatives \
    --method less \
    --validation_set /data/user_data/jmcoelho/datasets/marco/documents/processed_data/pythia-160m-marco-docs-bow-pretrain/random/val.jsonl \
    --train_run_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/run.train.txt \
    --train_qrels /data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv \
    --embedding_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/ \
    --number_of_negatives $n_negatives \
    --negatives_out_file hardnegs_less.txt \
    --output_dir temp \
    --model_name_or_path /data/user_data/jmcoelho/models/fine-tuned/$model \
    --dataset_cache_dir /data/datasets/hf_cache \
    --cache_dir /data/datasets/hf_cache \
    --bf16 \
    --pooling eos \
    --loss contrastive \
    --append_eos_token \
    --normalize \
    --temperature 0.01 \
    --query_max_len 32 \
    --passage_max_len 1024
