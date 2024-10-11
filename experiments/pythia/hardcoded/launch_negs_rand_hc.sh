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

#model=pythia-160m-marco-docs-bow-ct-pretrain-bs64-10pc-sample-less-neg
#--train_run_path /data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-1024-marco-docs-bow-contrastive-pretrain/msmarco-doctrain-top100-seqids.txt\


# model=pythia-160m-1024-marco-docs-bow-contrastive-pretrain

# n_negatives=9

# python -m tevatron.retriever.driver.select_hard_negatives \
#     --method random \
#     --validation_set /data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model/bm25/val.jsonl \
#     --train_run_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/run.train.20pc.sample.txt \
#     --train_qrels /data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv \
#     --embedding_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/ \
#     --number_of_negatives $n_negatives \
#     --negatives_out_file /data/user_data/jmcoelho/embeddings/marco_docs/$model/random_train_run_splits/random/20pc.tain+val.random.top50.txt \
#     --output_dir temp \
#     --model_name_or_path /data/user_data/jmcoelho/models/pre-trained/$model \
#     --dataset_cache_dir /data/datasets/hf_cache \
#     --cache_dir /data/datasets/hf_cache 


# model=pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision
# outfolder=/data/user_data/jmcoelho/embeddings/marco_docs/$model/random

# mkdir -p $outfolder
# n_negatives=9

# python -m tevatron.retriever.driver.select_hard_negatives \
#     --method random \
#     --validation_set /data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model/bm25/val.jsonl \
#     --train_run_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/run.train.txt \
#     --train_qrels /data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv \
#     --embedding_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/ \
#     --number_of_negatives $n_negatives \
#     --negatives_out_file $outfolder/full.queries.train+val.random.top100.txt \
#     --output_dir temp \
#     --model_name_or_path /data/user_data/jmcoelho/models/fine-tuned/$model \
#     --dataset_cache_dir /data/datasets/hf_cache \
#     --cache_dir /data/datasets/hf_cache 


model=pythia-160m-marco-docs-bow-ct-pretrain-bs256-generator-marco-greedy-RR-ep1
prefix=pre-trained
outfolder=/data/user_data/jmcoelho/embeddings/marco_docs/$model/gen17-shnegs/

mkdir -p $outfolder
n_negatives=9

python -m tevatron.retriever.driver.select_hard_negatives \
    --method random \
    --validation_set /data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model/bm25/val.jsonl \
    --train_run_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/run.gen17.rr.sample.txt \
    --train_qrels /data/user_data/jmcoelho/datasets/marco/documents/qrels.gen17.rr.sample.tsv \
    --embedding_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/ \
    --number_of_negatives $n_negatives \
    --negatives_out_file $outfolder/queries.random.shn.top100.rr.txt \
    --output_dir temp \
    --model_name_or_path /data/user_data/jmcoelho/models/$prefix/$model \
    --dataset_cache_dir /data/datasets/hf_cache \
    --cache_dir /data/datasets/hf_cache 


# head -1 /data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs64-10pc-sample-less-negs/random_train_run_splits/random/10pc.val.random.txt