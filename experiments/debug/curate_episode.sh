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

export TRANSFORMERS_CACHE=/data/datasets/hf_cache
eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-12.1


MODEL=pythia-160m-marco-docs-bow-ct-pretrain-bs256-supervised-warmup-minicpm-gpt4-teacher-shn-5pc-CE-filter-ep1
MODEL_FOR_MOMENTUM=pythia-160m-1024-marco-docs-bow-contrastive-pretrain
PATH_TO_RUN="/data/user_data/jmcoelho/datasets/marco/documents"
gen=gen17
percentage=5

EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs

# to write retriever training data
outfolder=/data/user_data/jmcoelho/embeddings/marco_docs/$MODEL/$gen-shnegs/
outfolder_momentum=/data/user_data/jmcoelho/embeddings/marco_docs/$MODEL_FOR_MOMENTUM/$gen-shnegs/

mkdir -p $outfolder
mkdir -p $outfolder_momentum

echo "Mapping files from multi-gpu generation"

cat $PATH_TO_RUN/qrels.${gen}_*.rr.tsv > $PATH_TO_RUN/qrels.${gen}.rr.tsv
cat $PATH_TO_RUN/qrels.${gen}_[0-9].tsv > $PATH_TO_RUN/qrels.${gen}.tsv
cat $PATH_TO_RUN/${gen}_*.query.tsv > $PATH_TO_RUN/$gen.query.tsv


echo "Filtering..."
python scripts/filter_queries_reranker.py $MODEL $gen $percentage
python scripts/marco_queries_tsv_to_jsonl.py $PATH_TO_RUN/$gen.query.$percentage.subset.tsv $PATH_TO_RUN/$gen.query.$percentage.subset.jsonl 

trained_model_name=$MODEL
prefix=fine-tuned


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
  --dataset_path "/data/user_data/jmcoelho/datasets/marco/documents/$gen.query.$percentage.subset.jsonl" \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-$gen-subset-$percentage.pkl


echo "Generating run with top-100 negatives for each query"

set -f && OMP_NUM_THREADS=24 python -m tevatron.retriever.driver.search_gpu \
    --query_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-$gen-subset-$percentage.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus*.pkl \
    --depth 100 \
    --batch_size 128 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.$gen.subset.$percentage.txt


echo "Sampling 9 random negatives per query"

n_negatives=9

python -m tevatron.retriever.driver.select_hard_negatives \
    --method random \
    --validation_set /data/user_data/jmcoelho/datasets/marco/documents/processed_data/$trained_model_name/bm25/val.jsonl \
    --train_run_path /data/user_data/jmcoelho/embeddings/marco_docs/$trained_model_name/run.$gen.subset.$percentage.txt \
    --train_qrels /data/user_data/jmcoelho/datasets/marco/documents/qrels.$gen.$percentage.subset.tsv \
    --embedding_path /data/user_data/jmcoelho/embeddings/marco_docs/$trained_model_name/ \
    --number_of_negatives $n_negatives \
    --negatives_out_file $outfolder/queries.random.shn.top100.subset.$percentage.txt \
    --output_dir temp \
    --model_name_or_path /data/user_data/jmcoelho/models/$prefix/$trained_model_name \
    --dataset_cache_dir /data/datasets/hf_cache \
    --cache_dir /data/datasets/hf_cache 


echo "Generating embedding and dpo training datasets"

trained_model_name=$MODEL_FOR_MOMENTUM
prefix=pre-trained
mkdir -p $EMBEDDING_OUTPUT_DIR/$trained_model_name

echo "Encoding generated queries for momentum"

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
  --dataset_path "/data/user_data/jmcoelho/datasets/marco/documents/$gen.query.$percentage.subset.jsonl" \
  --encode_output_path $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-$gen-subset-$percentage.pkl


echo "Generating run with top-100 negatives for each query"

set -f && OMP_NUM_THREADS=24 python -m tevatron.retriever.driver.search_gpu \
    --query_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-$gen-subset-$percentage.pkl \
    --passage_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus*.pkl \
    --depth 100 \
    --batch_size 128 \
    --save_text \
    --save_ranking_to $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.$gen.subset.$percentage.txt


echo "Sampling 9 random negatives per query"

n_negatives=9

python -m tevatron.retriever.driver.select_hard_negatives \
    --method random \
    --validation_set /data/user_data/jmcoelho/datasets/marco/documents/processed_data/$trained_model_name/bm25/val.jsonl \
    --train_run_path /data/user_data/jmcoelho/embeddings/marco_docs/$trained_model_name/run.$gen.subset.$percentage.txt \
    --train_qrels /data/user_data/jmcoelho/datasets/marco/documents/qrels.$gen.$percentage.subset.tsv \
    --embedding_path /data/user_data/jmcoelho/embeddings/marco_docs/$trained_model_name/ \
    --number_of_negatives $n_negatives \
    --negatives_out_file $outfolder_momentum/queries.random.shn.top100.subset.$percentage.txt \
    --output_dir temp \
    --model_name_or_path /data/user_data/jmcoelho/models/$prefix/$trained_model_name \
    --dataset_cache_dir /data/datasets/hf_cache \
    --cache_dir /data/datasets/hf_cache 

python scripts/add_momentum_to_run.py $outfolder/queries.random.shn.top100.subset.$percentage.txt $outfolder_momentum/queries.random.shn.top100.subset.$percentage.txt $outfolder/queries.random.shn.top100.subset.$percentage.momentum.txt