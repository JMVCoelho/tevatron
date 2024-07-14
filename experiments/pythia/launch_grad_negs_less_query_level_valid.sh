#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --gres=gpu:6000Ada:1
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-4-28

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model=pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1
prefix=fine-tuned
n_negatives=9

subset=$1

python -m tevatron.retriever.driver.select_hard_negatives \
    --method valid_query_level \
    --validation_set /data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model/random_all_queries_10k_two_valid/val_2.jsonl \
    --train_run_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/group-level-valid-oracle/run.train.all.queries.$1 \
    --train_qrels /data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv \
    --embedding_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/valid_grads_bs64/ \
    --number_of_negatives $n_negatives \
    --negatives_out_file /data/user_data/jmcoelho/embeddings/marco_docs/$model/group_level_10000_two_valid_orcale_momentum_single_query_2k/group_hardnegs_$1 \
    --output_dir temp \
    --model_name_or_path /data/user_data/jmcoelho/models/$prefix/$model \
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