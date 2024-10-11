#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:6000Ada:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
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
  trained_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-llama-clueweb
  prefix=fine-tuned

  echo $test_data

  mkdir -p $EMBEDDING_OUTPUT_DIR

  python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --model_name_or_path /data/user_data/jmcoelho/models/$prefix/$trained_model_name \
    --bf16 \
    --pooling eos \
    --dataset_cache_dir /data/datasets/hf_cache \
    --cache_dir /data/datasets/hf_cache \
    --query_prefix "" \
    --passage_prefix "" \
    --append_eos_token \
    --normalize \
    --encode_is_query \
    --per_device_eval_batch_size 300 \
    --query_max_len 32 \
    --passage_max_len 1024 \
    --dataset_name Tevatron/beir \
    --dataset_config $test_data \
    --dataset_split test \
    --encode_output_path $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-test-$test_data.pkl

    echo "################"

done