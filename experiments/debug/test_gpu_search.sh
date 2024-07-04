#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=babel-4-36,babel-8-3,babel-4-28


eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.3

trained_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-valid-5-group-level-best

EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs
mkdir $EMBEDDING_OUTPUT_DIR/$trained_model_name


set -f && python src/tevatron/retriever/driver/search.py \
    --query_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-val.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus*.pkl \
    --depth 100 \
    --batch_size 128 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/$trained_model_name/debug_cpu.txt

set -f && accelerate launch --num_processes 1 --main_process_port 29777 src/tevatron/retriever/driver/search_gpu.py \
    --query_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-val.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus*.pkl \
    --depth 100 \
    --batch_size 128 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/$trained_model_name/debug_gpu.txt


# python src/tevatron/utils/format/convert_result_to_trec.py \
#     --input $EMBEDDING_OUTPUT_DIR/$trained_model_name/debug2.txt \
#     --output $EMBEDDING_OUTPUT_DIR/$trained_model_name/debug2.trec


# qrels=./qrels/marco.docs.val.qrel.tsv
# trec_run=$EMBEDDING_OUTPUT_DIR/$trained_model_name/debug2.trec

# python scripts/evaluate.py $qrels $trec_run
# python scripts/evaluate.py -m mrr_cut.100 $qrels $trec_run


# https://github.com/facebookresearch/faiss/issues/1463
# https://github.com/facebookresearch/faiss/wiki/Troubleshooting#crashes-in-pure-python-code
# https://github.com/facebookresearch/faiss/issues/505
# https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
