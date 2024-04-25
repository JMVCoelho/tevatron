#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00

export TRANSFORMERS_CACHE=/data/datasets/hf_cache

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

trained_model_name=pythia-160m-marco-docs-bow-pretain-contrastive-pretrain-bs64

EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs
mkdir $EMBEDDING_OUTPUT_DIR/$trained_model_name

set -f && python -m tevatron.retriever.driver.search \
    --query_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-dev.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus*.pkl \
    --depth 1000 \
    --batch_size 128 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dev.txt


python src/tevatron/utils/format/convert_result_to_trec.py \
    --input $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dev.txt \
    --output $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dev.trec


qrels=./qrels/marco.docs.dev.qrel.tsv
trec_run=$EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dev.trec

python scripts/evaluate.py $qrels $trec_run
