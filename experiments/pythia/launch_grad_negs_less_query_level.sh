#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --gres=gpu:L40:1
#SBATCH --time=2-00:00:00

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model=pythia-160m-1024-marco-docs-bow-contrastive-pretrain
n_negatives=9

subset=$1

python -m tevatron.retriever.driver.select_hard_negatives \
    --method less_query_level \
    --validation_set /data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model/random/val.jsonl \
    --train_run_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/run.train.10.percent.sample.txt \
    --train_qrels /data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv \
    --embedding_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/valid_grads_larger/ \
    --number_of_negatives $n_negatives \
    --negatives_out_file /data/user_data/jmcoelho/embeddings/marco_docs/$model/query_level_weights_random/random_negs_with_query_weight.10pc.txt \
    --output_dir temp \
    --model_name_or_path /data/user_data/jmcoelho/models/pre-trained/$model \
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