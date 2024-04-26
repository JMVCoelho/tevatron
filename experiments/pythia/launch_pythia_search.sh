#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00

export TRANSFORMERS_CACHE=/data/user_data/jmcoelho/hf_cache

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

trained_model_name=pythia-160m-marco-passage-bow-pretrain-distil-llama-3-score+embed
#trained_model_name=pythia-160m-marco-passage-bow-pretrain-32bs-2gpu

EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_passage
mkdir $EMBEDDING_OUTPUT_DIR/$trained_model_name


# echo "TRAIN"
# echo "######################################"
# set -f && python -m tevatron.retriever.driver.search \
#     --query_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-train.pkl \
#     --passage_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus*.pkl \
#     --depth 1000 \
#     --batch_size 0 \
#     --save_text \
#     --save_ranking_to $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.train.txt


# python src/tevatron/utils/format/convert_result_to_trec.py \
#     --input $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.train.txt \
#     --output $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.train.trec


# exit


echo "DEV"
echo "######################################"
set -f && python -m tevatron.retriever.driver.search \
    --query_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-dev.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus*.pkl \
    --depth 1000 \
    --batch_size 0 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dev.txt


python src/tevatron/utils/format/convert_result_to_trec.py \
    --input $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dev.txt \
    --output $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dev.trec


qrels=./qrels/marco.passage.dev.small.qrel.tsv
trec_run=$EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dev.trec

python scripts/evaluate.py $qrels $trec_run > $EMBEDDING_OUTPUT_DIR/$trained_model_name/results.dev.trec
python scripts/evaluate.py -m mrr_cut.10 $qrels $trec_run
python scripts/evaluate.py -m recall.1000 $qrels $trec_run


echo "DL19"
echo "######################################"
set -f && python -m tevatron.retriever.driver.search \
    --query_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-dl19.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus*.pkl \
    --depth 1000 \
    --batch_size 0 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dl19.txt


python src/tevatron/utils/format/convert_result_to_trec.py \
    --input $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dl19.txt \
    --output $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dl19.trec


qrels=./qrels/trecdl19.qrel.tsv
trec_run=$EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dl19.trec

python scripts/evaluate.py $qrels $trec_run > $EMBEDDING_OUTPUT_DIR/$trained_model_name/results.dl19.trec
python scripts/evaluate.py -m ndcg_cut.10 $qrels $trec_run


echo "DL20"
echo "######################################"
set -f && python -m tevatron.retriever.driver.search \
    --query_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-dl20.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus*.pkl \
    --depth 1000 \
    --batch_size 0 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dl20.txt


python src/tevatron/utils/format/convert_result_to_trec.py \
    --input $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dl20.txt \
    --output $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dl20.trec


qrels=./qrels/trecdl20.qrel.tsv
trec_run=$EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dl20.trec

python scripts/evaluate.py $qrels $trec_run > $EMBEDDING_OUTPUT_DIR/$trained_model_name/results.dl20.trec
python scripts/evaluate.py -m ndcg_cut.10 $qrels $trec_run