#!/bin/bash

#SBATCH --job-name=qwen-retriever-inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=preempt
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-13-13,babel-14-37,babel-6-9,babel-7-9,babel-3-21,babel-13-25,babel-13-1,babel-14-1,babel-12-9

eval "$(conda shell.bash hook)"
conda activate tevatron
module load cuda-12.4

trained_model_name=$1

EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs/
mkdir $EMBEDDING_OUTPUT_DIR/$trained_model_name

set -f && OMP_NUM_THREADS=24 python -m tevatron.retriever.driver.search \
    --query_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-marco-dev.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus*.pkl \
    --depth 100 \
    --batch_size 128 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dev.txt


python src/tevatron/utils/format/convert_result_to_trec.py \
    --input $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dev.txt \
    --output $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dev.trec


qrels=./qrels/marco.docs.dev.qrel.tsv
trec_run=$EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dev.trec

python scripts/eval_trec.py $qrels $trec_run
python scripts/eval_trec.py -m mrr_cut.100 $qrels $trec_run