#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:6000Ada:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=babel-4-36,babel-8-3,babel-4-28

export TRANSFORMERS_CACHE=/data/datasets/hf_cache
eval "$(conda shell.bash hook)"
conda activate tevatron
module load cuda-11.8

export WANDB_MODE=disabled

start=$(( ($1 - 1) * 1000 + 1 ))
end=$(( $1 * 1000 ))

for i in $(seq $start $end);
do  
    echo ====$i====
    single_neg_index=$i
    port=$((RANDOM % (23000 - 20000 + 1) + 20000))

    #query=1185869
    #query=466896
    #query=152842
    #query=950963

    model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1
    prefix=fine-tuned
    final_model_name=$model_to_train-single-update-$single_neg_index

    mkdir /data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/group_level_all_samples/

    save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/temp_$single_neg_index
    training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/group_level_all_samples/group_hardnegs_oneq_sample_$single_neg_index

    python scripts/build_random_single_example_all_q.py $i $model_to_train $training_data

    echo ======= negs
    cat $training_data
    echo ============

    trained_model_name=$prefix/$model_to_train 
    save_pretok=$save_pretok
    negative_file=$training_data

    text_length=1024

    data_path=/data/user_data/jmcoelho/datasets/marco/documents

    # train_qrels=$data_path/qrels.train.tsv
    # corpus=$data_path/corpus_firstp_2048.tsv
    # train_queries=$data_path/train.query.filtered.txt

    train_qrels=$data_path/qrels.gen.2.tsv
    corpus=$data_path/corpus_firstp_2048.tsv
    train_queries=$data_path/gen.query.tsv

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
        --n_sample 9

    cat $initial_data_save_folder/split*.jsonl > $initial_data_save_folder/train.jsonl
    rm $initial_data_save_folder/split*.jsonl

    trained_model_name=$final_model_name
    training_data=$save_pretok/train.jsonl


    deepspeed --include localhost:0 --master_port $port --module tevatron.retriever.driver.train \
        --deepspeed deepspeed/ds_zero3_config_const_lr.json \
        --output_dir /data/user_data/jmcoelho/models/fine-tuned/$trained_model_name \
        --model_name_or_path /data/user_data/jmcoelho/models/$prefix/$model_to_train \
        --dataset_path $training_data \
        --dataset_cache_dir /data/datasets/hf_cache \
        --cache_dir /data/datasets/hf_cache \
        --lr_scheduler_type 'constant' \
        --save_steps 1000 \
        --bf16 \
        --pooling eos \
        --loss contrastive \
        --gradient_checkpointing \
        --append_eos_token \
        --normalize \
        --temperature 0.01 \
        --per_device_train_batch_size 1 \
        --train_group_size 10 \
        --learning_rate 1e-5 \
        --query_max_len 32 \
        --passage_max_len 1024 \
        --num_train_epochs 1 \
        --logging_steps 1 \
        --overwrite_output_dir \
        --gradient_accumulation_steps 1
        #   --report_to wandb \
        #   --run_name $trained_model_name
    
    model_name=$final_model_name
    model_to_valid=/data/user_data/jmcoelho/models/fine-tuned/$model_name

    model_valid_data=$model_to_train

    rm /data/user_data/jmcoelho/models/fine-tuned/$model_name/model.safetensors

    deepspeed --include localhost:0 --master_port $port --module tevatron.retriever.driver.train \
        --deepspeed deepspeed/ds_zero3_config.json \
        --output_dir temp_$1 \
        --model_name_or_path $model_to_valid\
        --dataset_path /data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_valid_data/random_all_queries_10k_two_valid/val_1_subset.jsonl \
        --dataset_cache_dir /data/datasets/hf_cache \
        --cache_dir /data/datasets/hf_cache \
        --save_steps 1000 \
        --bf16 \
        --pooling eos \
        --loss contrastive \
        --gradient_checkpointing \
        --append_eos_token \
        --normalize \
        --temperature 0.01 \
        --train_group_size 10 \
        --query_max_len 32 \
        --passage_max_len 1024 \
        --overwrite_output_dir \
        --normalize \
        --per_device_train_batch_size 64 \
        --learning_rate 0 \
        --num_train_epochs 1 \
        --logging_steps 1 \
        --overwrite_output_dir \
        --gradient_accumulation_steps 1
        #   --report_to wandb \
        #   --run_name $model_name-valid

    rm -r $model_to_valid
    rm -r $initial_data_save_folder

done

