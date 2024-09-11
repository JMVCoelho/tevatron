#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:A6000:1
#SBATCH --exclude=babel-4-36,babel-8-3,babel-4-28


eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

trained_model_name=$1

EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs
mkdir $EMBEDDING_OUTPUT_DIR/$trained_model_name

echo "VALID 1"
echo "########################"

set -f && OMP_NUM_THREADS=24 python -m tevatron.retriever.driver.search_gpu \
    --query_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-val-1.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus*.pkl \
    --depth 100 \
    --batch_size 128 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.valid.1.txt


python src/tevatron/utils/format/convert_result_to_trec.py \
    --input $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.valid.1.txt \
    --output $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.valid.1.trec


qrels=./qrels/marco.docs.5000val1.qrel.tsv
trec_run=$EMBEDDING_OUTPUT_DIR/$trained_model_name/run.valid.1.trec

python scripts/evaluate.py $qrels $trec_run
python scripts/evaluate.py -m mrr_cut.100 $qrels $trec_run


echo "VALID 2"
echo "########################"

set -f && OMP_NUM_THREADS=24 python -m tevatron.retriever.driver.search_gpu \
    --query_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-val-2.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus*.pkl \
    --depth 100 \
    --batch_size 128 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.valid.2.txt


python src/tevatron/utils/format/convert_result_to_trec.py \
    --input $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.valid.2.txt \
    --output $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.valid.2.trec


qrels=./qrels/marco.docs.5000val2.qrel.tsv
trec_run=$EMBEDDING_OUTPUT_DIR/$trained_model_name/run.valid.2.trec

python scripts/eval_trec.py $qrels $trec_run
python scripts/eval_trec.py -m mrr_cut.100 $qrels $trec_run