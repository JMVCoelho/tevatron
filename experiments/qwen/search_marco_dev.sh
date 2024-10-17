#!/bin/bash

#SBATCH --job-name=qwen-pretrain
# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id

#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 12 # number cpus (threads) per task

# 327680
#SBATCH --mem=50G # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit

#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1


eval "$(conda shell.bash hook)"
conda activate cmu-llms-hw3

trained_model_name=$1

EMBEDDING_OUTPUT_DIR=/data/jcoelho/embeddings/babel/
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