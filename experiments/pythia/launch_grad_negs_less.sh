#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --time=2-00:00:00

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

model=pythia-160m-marco-docs-bow-pretrain 
n_negatives=9

#vals=("0" "1" "2")
#vals=("3" "4" "5")
vals=("6" "7" "8" "9")


for i in "${!vals[@]}"; do
python -m tevatron.retriever.driver.select_hard_negatives \
    --method less \
    --validation_set /data/user_data/jmcoelho/datasets/marco/documents/processed_data/pythia-160m-marco-docs-bow-pretrain/random/val.jsonl \
    --train_run_path /data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-pretrain/random_train_run_splits/less/run.train.${vals[i]} \
    --train_qrels /data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv \
    --embedding_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/valid_grads_larger \
    --number_of_negatives $n_negatives \
    --negatives_out_file /data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-pretrain/random_train_run_splits/less/hardnegs_less_${vals[i]}.txt \
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
    --passage_max_len 1024 \
    2>&1 &
done
wait