#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

model=pythia-160m-marco-docs-bow-pretrain 
n_negatives=9


python -m tevatron.retriever.driver.select_hard_negatives \
    --method indi \
    --train_run_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/run.train.txt \
    --train_qrels /data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv \
    --embedding_path /data/user_data/jmcoelho/embeddings/marco_docs/$model/ \
    --number_of_negatives $n_negatives \
    --negatives_out_file hardnegs_indi.txt
