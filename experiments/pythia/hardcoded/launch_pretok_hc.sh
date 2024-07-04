#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=babel-8-3,babel-11-25

export TRANSFORMERS_CACHE=/data/datasets/hf_cache
eval "$(conda shell.bash hook)"
conda activate tevatron
module load cuda-11.8

trained_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-dev-overfit
prefix=fine-tuned
save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$trained_model_name/dev_overfit
negative_file=/data/user_data/jmcoelho/embeddings/marco_docs/$trained_model_name/random_train_run_splits/random/full.queries.dev.top100.txt

text_length=1024

data_path=/data/user_data/jmcoelho/datasets/marco/documents

# train_qrels=$data_path/qrels.train.tsv
# corpus=$data_path/corpus_firstp_2048.tsv
# train_queries=$data_path/train.query.filtered.txt

train_qrels=$data_path/qrels.dev.tsv
corpus=$data_path/corpus_firstp_2048.tsv
train_queries=$data_path/dev.query.txt

initial_data_save_folder=$save_pretok

#bs64_contrastive_topk
mkdir -p $initial_data_save_folder

python scripts/pretokenize.py \
   --tokenizer_name /data/user_data/jmcoelho/models/$prefix/$trained_model_name \
   --negative_file $negative_file\
   --qrels $train_qrels  \
   --queries $train_queries  \
   --collection $corpus \
   --truncate $text_length \
   --save_to $initial_data_save_folder  \
   --doc_template "Title: <title> Text: <text>" \
   --n_sample 9

cat $initial_data_save_folder/split*.jsonl > $initial_data_save_folder/dev.jsonl
rm $initial_data_save_folder/split*.jsonl
exit


cat $initial_data_save_folder/split*.jsonl > $initial_data_save_folder/full.jsonl
rm $initial_data_save_folder/split*.jsonl

line_count=$(wc -l $initial_data_save_folder/full.jsonl | awk '{print $1}')
n_val=10000
n_subsets=5000
n_train=$((line_count - n_val))

echo $n_train

tail -n $n_val $initial_data_save_folder/full.jsonl > $initial_data_save_folder/val.jsonl
head -n $n_train $initial_data_save_folder/full.jsonl > $initial_data_save_folder/train.jsonl

head -n $n_subsets $initial_data_save_folder/val.jsonl > $initial_data_save_folder/val_1.jsonl
tail -n $n_subsets $initial_data_save_folder/val.jsonl > $initial_data_save_folder/val_2.jsonl

rm $initial_data_save_folder/full.jsonl