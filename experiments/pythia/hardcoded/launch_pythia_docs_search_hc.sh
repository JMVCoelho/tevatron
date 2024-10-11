#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:6000Ada:1
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=babel-8-3,babel-11-25


eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-12.1

trained_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-generator-marco-greedy-RR-ep1
#trained_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision-1gpu

EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs
mkdir $EMBEDDING_OUTPUT_DIR/$trained_model_name

set -f && OMP_NUM_THREADS=24 python -m tevatron.retriever.driver.search_gpu \
    --query_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-gen17-rr-sample.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus*.pkl \
    --depth 100 \
    --batch_size 128 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.gen17.rr.sample.txt


python src/tevatron/utils/format/convert_result_to_trec.py \
    --input $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.gen17.rr.sample.txt \
    --output $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.gen17.rr.sample.trec


python scripts/eval_trec.py /data/user_data/jmcoelho/datasets/marco/documents/qrels.gen17.tsv $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.gen17.rr.sample.trec