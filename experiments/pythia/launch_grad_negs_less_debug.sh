#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=45G
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-4-28


eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
n_negatives=9


#model=pythia-160m-1024-marco-docs-bow-contrastive-pretrain
#prefix=pre-trained

model=pythia-160m-marco-docs-bow-ct-pretrain-bs128-20pc-sample-less-negs-triplet-topk
prefix=fine-tuned

subset=$1

python -m tevatron.retriever.driver.select_hard_negatives \
    --method less \
    --validation_set /data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model/random/val.jsonl \
    --train_run_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/20pc-sample-run-splits/less-opacus-triplet/run.train.20pc.sample.$1 \
    --train_qrels /data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv \
    --embedding_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/valid_grads_bs64_denoised \
    --number_of_negatives $n_negatives \
    --negatives_out_file /data/user_data/jmcoelho/embeddings/marco_docs/$model/20pc-sample-run-splits/less-opacus-triplet-denoised-delete/hardnegs_less_opacus.20.pc.$1.txt \
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



# model=pythia-160m-marco-docs-bow-ct-pretrain-bs128-all-queries-less-topk-negs-top50
# prefix=fine-tuned

# python -m tevatron.retriever.driver.select_hard_negatives \
#     --method less \
#     --validation_set /data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model/random/val.jsonl \
#     --train_run_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/less-opacus-triplet/run.train.all.queries.$1 \
#     --train_qrels /data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv \
#     --embedding_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/valid_grads_bs64\
#     --number_of_negatives $n_negatives \
#     --negatives_out_file /data/user_data/jmcoelho/embeddings/marco_docs/$model/full-queries-run-splits/less-opacus-triplet/hardnegs_less_opacus.full.qs.$1.txt \
#     --output_dir temp \
#     --model_name_or_path /data/user_data/jmcoelho/models/$prefix/$model \
#     --dataset_cache_dir /data/datasets/hf_cache \
#     --cache_dir /data/datasets/hf_cache \
#     --bf16 \
#     --pooling eos \
#     --loss contrastive \
#     --append_eos_token \
#     --normalize \
#     --temperature 0.01 \
#     --query_max_len 32 \
#     --passage_max_len 1024