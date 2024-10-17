#!/bin/bash

#SBATCH --job-name=qwen-pretrain
# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id

#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 64 # number cpus (threads) per task

# 327680
#SBATCH --mem=200G # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit

eval "$(conda shell.bash hook)"
conda activate cmu-llms-hw3

trained_model_name=$1
save_pretok=$2
negative_file=$3
n_val=$4
negs=$5

text_length=512

data_path=/data/jcoelho/datasets/babel/

train_qrels=$data_path/qrels.train.tsv
corpus=$data_path/corpus_firstp_2048.tsv
train_queries=$data_path/train.query.filtered.txt

initial_data_save_folder=$save_pretok

mkdir -p $initial_data_save_folder

python scripts/pretokenize.py \
   --tokenizer_name /user/home/jcoelho/Qwen/models/$trained_model_name \
   --negative_file $negative_file\
   --qrels $train_qrels  \
   --queries $train_queries  \
   --collection $corpus \
   --truncate $text_length \
   --save_to $initial_data_save_folder  \
   --doc_template "Title: <title> Text: <text>" \
   --n_sample $5

cat $initial_data_save_folder/split*.jsonl > $initial_data_save_folder/train.jsonl
rm $initial_data_save_folder/split*.jsonl

# n_train=$((line_count - n_val))

# echo $n_train

# cat $initial_data_save_folder/split*.jsonl > $initial_data_save_folder/full.jsonl


# line_count=$(wc -l $initial_data_save_folder/full.jsonl | awk '{print $1}')

# n_train=$((line_count - n_val))

# echo $n_train

# tail -n $n_val $initial_data_save_folder/full.jsonl > $initial_data_save_folder/val.jsonl
# head -n $n_train $initial_data_save_folder/full.jsonl > $initial_data_save_folder/train.jsonl

# rm $initial_data_save_folder/full.jsonl