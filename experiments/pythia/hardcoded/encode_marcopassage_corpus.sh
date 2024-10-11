#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:6000Ada:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --time=2-00:00:00

export TRANSFORMERS_CACHE=/data/datasets/hf_cache

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-12.1


shard=$1


EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs
trained_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-llama-clueweb
prefix=fine-tuned

echo $test_data

mkdir -p $EMBEDDING_OUTPUT_DIR

python -m tevatron.retriever.driver.encode \
    --output_dir=temp \
    --model_name_or_path /data/user_data/jmcoelho/models/$prefix/$trained_model_name \
    --bf16 \
    --dataset_cache_dir /data/datasets/hf_cache \
    --cache_dir /data/datasets/hf_cache \
    --pooling eos \
    --query_prefix "" \
    --passage_prefix "" \
    --append_eos_token \
    --normalize \
    --per_device_eval_batch_size 300 \
    --query_max_len 32 \
    --passage_max_len 1024 \
    --dataset_name Tevatron/msmarco-passage-corpus \
    --encode_output_path $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus.marco-passage.${shard}.pkl \
    --add_markers True \
    --dataset_number_of_shards 4 \
    --dataset_shard_index ${shard}


