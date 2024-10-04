#!/bin/bash
#SBATCH --job-name=curate_dataset
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-4-36,babel-8-3,babel-4-28

MODEL=pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision
PATH_TO_RUN="/data/user_data/jmcoelho/datasets/marco/documents"
gen=gen12

echo "Mapping files from multi-gpu generation"

# cat $PATH_TO_RUN/qrels.${gen}_*.rr.tsv > $PATH_TO_RUN/qrels.${gen}.rr.tsv
# cat $PATH_TO_RUN/qrels.${gen}_[0-9].tsv > $PATH_TO_RUN/qrels.${gen}.tsv

# cat $PATH_TO_RUN/${gen}_*.query.tsv > $PATH_TO_RUN/$gen.query.tsv
# python scripts/marco_queries_tsv_to_jsonl.py $gen


#### REFACTOR NEEDED:
# Instead of running the top-N for all queries and then select, select first and the run top-N.
export TRANSFORMERS_CACHE=/data/datasets/hf_cache

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-12.1

trained_model_name=$MODEL
prefix=fine-tuned

EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs
mkdir -p $EMBEDDING_OUTPUT_DIR/$trained_model_name

echo "Encoding generated queries"

python -m tevatron.retriever.driver.encode \
  --output_dir=temp \
  --model_name_or_path /data/user_data/jmcoelho/models/$prefix/$trained_model_name/ \
  --dataset_cache_dir /data/datasets/hf_cache \
  --cache_dir /data/datasets/hf_cache \
  --query_prefix "" \
  --passage_prefix "" \
  --bf16 \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --encode_is_query \
  --per_device_eval_batch_size 300 \
  --query_max_len 32 \
  --passage_max_len 1024 \
  --dataset_path "/data/user_data/jmcoelho/datasets/marco/documents/$gen.query.jsonl" \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-$gen.pkl


echo "Generating run with top-100 negatives for each query"

set -f && OMP_NUM_THREADS=24 python -m tevatron.retriever.driver.search_gpu \
    --query_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-$gen.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus*.pkl \
    --depth 100 \
    --batch_size 128 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.$gen.txt


echo "Sampling 9 random negatives per query"

outfolder=/data/user_data/jmcoelho/embeddings/marco_docs/$trained_model_name/$gen-shnegs/

mkdir -p $outfolder
n_negatives=9

python -m tevatron.retriever.driver.select_hard_negatives \
    --method random \
    --validation_set /data/user_data/jmcoelho/datasets/marco/documents/processed_data/$trained_model_name/bm25/val.jsonl \
    --train_run_path /data/user_data/jmcoelho/embeddings/marco_docs/$trained_model_name/run.$gen.txt \
    --train_qrels /data/user_data/jmcoelho/datasets/marco/documents/qrels.$gen.tsv \
    --embedding_path /data/user_data/jmcoelho/embeddings/marco_docs/$trained_model_name/ \
    --number_of_negatives $n_negatives \
    --negatives_out_file $outfolder/queries.random.shn.top100.txt \
    --output_dir temp \
    --model_name_or_path /data/user_data/jmcoelho/models/$prefix/$model \
    --dataset_cache_dir /data/datasets/hf_cache \
    --cache_dir /data/datasets/hf_cache 


echo "Generating embedding and dpo training datasets"

python scripts/filter_queries_reranker.py $MODEL $gen 5