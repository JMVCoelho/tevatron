#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:6000Ada:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00

export TRANSFORMERS_CACHE=/data/datasets/hf_cache

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-12.1


all_data=(
    'arguana'
    'climate-fever'
    'cqadupstack-android'
    'cqadupstack-english'
    'cqadupstack-gaming'
    'cqadupstack-gis'
    'cqadupstack-mathematica'
    'cqadupstack-physics'
    'cqadupstack-programmers'
    'cqadupstack-stats'
    'cqadupstack-tex'
    'cqadupstack-unix'
    'cqadupstack-webmasters'
    'cqadupstack-wordpress'
    'dbpedia-entity'
    'fever'
    'fiqa'
    'hotpotqa'
    'nfcorpus'
    'quora'
    'scidocs'
    'scifact'
    'trec-covid'
    'webis-touche2020'
    'nq'
)

for test_data in "${all_data[@]}"; do

  EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs
  trained_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-llama-clueweb-supervision-e2

  echo $test_data

  set -f && OMP_NUM_THREADS=24 python -m tevatron.retriever.driver.search_gpu \
      --query_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-test-$test_data.pkl \
      --passage_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus.$test_data.*.pkl \
      --depth 100 \
      --batch_size 128 \
      --save_text \
      --save_ranking_to $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.$test_data.txt

  
  python src/tevatron/utils/format/convert_result_to_trec.py \
    --input $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.$test_data.txt \
    --output $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.$test_data.trec \
    --remove_query

  rm $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.$test_data.txt

  echo "################"
done

eval "$(conda shell.bash hook)"
conda activate pyserini

for test_data in "${all_data[@]}"; do
  echo $test_data
  python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 beir-v1.0.0-${test_data}-test $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.$test_data.trec
  echo "################"
  rm $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus.$test_data.*.pkl
  rm $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-test-$test_data.pkl
  rm $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.$test_data.trec
done