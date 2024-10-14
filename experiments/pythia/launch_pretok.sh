#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=babel-4-36,babel-8-3,babel-4-28,babel-4-11

export TRANSFORMERS_CACHE=/data/datasets/hf_cache
eval "$(conda shell.bash hook)"
conda activate tevatron
module load cuda-12.1

trained_model_name=$1
save_pretok=$2
negative_file=$3
n_val=$4

text_length=1024

#data_path=/data/user_data/jmcoelho/datasets/marco/documents

# train_qrels=$data_path/qrels.train.tsv
# corpus=$data_path/corpus_firstp_2048.tsv
# train_queries=$data_path/train.query.filtered.txt


# data_path=/data/user_data/jmcoelho/datasets/marco/documents

# train_qrels=$data_path/qrels.gen5.tsv
# corpus=$data_path/corpus_firstp_2048.tsv
# train_queries=$data_path/gen5.query.tsv

data_path=/data/user_data/jmcoelho/datasets/marco/documents

train_qrels=$data_path/qrels.train.tsv
corpus=$data_path/corpus_firstp_2048.tsv
train_queries=$data_path/train.query.filtered.txt

# data_path=/data/group_data/cx_group/query_generation_data/GPT4/bm25-negatives

# train_qrels=$data_path/qrels.gen7.int.tsv
# corpus=$data_path/corpus_int.tsv
# train_queries=$data_path/../gen7.query.tsv

initial_data_save_folder=$save_pretok

#bs64_contrastive_topk
mkdir -p $initial_data_save_folder

python scripts/pretokenize.py \
   --tokenizer_name /data/user_data/jmcoelho/models/$trained_model_name \
   --negative_file $negative_file\
   --qrels $train_qrels  \
   --queries $train_queries  \
   --collection $corpus \
   --truncate $text_length \
   --save_to $initial_data_save_folder  \
   --doc_template "Title: <title> Text: <text>" \
   --n_sample 5

cat $initial_data_save_folder/split*.jsonl > $initial_data_save_folder/train.jsonl
rm $initial_data_save_folder/split*.jsonl

n_train=$((line_count - n_val))

echo $n_train
# cat $initial_data_save_folder/split*.jsonl > $initial_data_save_folder/full.jsonl


# line_count=$(wc -l $initial_data_save_folder/full.jsonl | awk '{print $1}')

# n_train=$((line_count - n_val))

# echo $n_train

# tail -n $n_val $initial_data_save_folder/full.jsonl > $initial_data_save_folder/val.jsonl
# head -n $n_train $initial_data_save_folder/full.jsonl > $initial_data_save_folder/train.jsonl

# rm $initial_data_save_folder/full.jsonl