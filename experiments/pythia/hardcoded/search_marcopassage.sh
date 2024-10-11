#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00

export TRANSFORMERS_CACHE=/data/datasets/hf_cache

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-12.1



EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs
trained_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-llama-clueweb

echo $test_data

set -f && OMP_NUM_THREADS=24 python -m tevatron.retriever.driver.search_gpu \
    --query_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-test-marco-passage.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus.marco-passage.*.pkl \
    --depth 100 \
    --batch_size 128 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.marco-passage.txt


python src/tevatron/utils/format/convert_result_to_trec.py \
      --input $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.marco-passage.txt \
      --output $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.marco-passage.trec 

rm $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.marco-passage.txt

eval "$(conda shell.bash hook)"
conda activate pyserini

python -m pyserini.eval.trec_eval -c -mndcg_cut.10 msmarco-passage-dev-subset $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.marco-passage.trec