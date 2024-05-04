#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00

export TRANSFORMERS_CACHE=/data/datasets/hf_cache
eval "$(conda shell.bash hook)"
conda activate tevatron
module load cuda-11.8

trained_model_name="pythia-160m-marco-docs-bow-pretrain"
text_length=1024

data_path=/data/user_data/jmcoelho/datasets/marco/documents

train_qrels=$data_path/qrels.train.tsv
corpus=$data_path/corpus_firstp_2048.tsv
train_queries=$data_path/train.query.filtered.txt

initial_data_save_folder=$data_path/processed_data/$trained_model_name/random_normal_sample

mkdir -p $initial_data_save_folder

python scripts/pretokenize.py \
   --tokenizer_name /data/user_data/jmcoelho/models/fine-tuned/$trained_model_name \
   --negative_file /data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-pretrain/random_train_run_splits/random/hardnegs_random.random.txt \
   --qrels $train_qrels  \
   --queries $train_queries  \
   --collection $corpus \
   --truncate $text_length \
   --save_to $initial_data_save_folder  \
   --doc_template "Title: <title> Text: <text>" \
   --n_sample 9

cat $initial_data_save_folder/split*.jsonl > $initial_data_save_folder/full.jsonl
rm $initial_data_save_folder/split*.jsonl

line_count=$(wc -l $initial_data_save_folder/full.jsonl | awk '{print $1}')
n_val=500
n_train=$((line_count - n_val))

echo $n_train

tail -n $n_val $initial_data_save_folder/full.jsonl > $initial_data_save_folder/val.jsonl
head -n $n_train $initial_data_save_folder/full.jsonl > $initial_data_save_folder/train.jsonl

rm $initial_data_save_folder/full.jsonl