#!/bin/bash
#SBATCH --job-name=pythia_dr
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:6000Ada:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-4-36,babel-8-3,babel-4-28

export TRANSFORMERS_CACHE=/data/datasets/hf_cache

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

#sampling_temp=inf
#trained_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs64-10pc-sample-less-negs-self-hn1-less-temp$sampling_temp
#"/data/user_data/jmcoelho/datasets/marco/documents/processed_data/pythia-160m-marco-docs-bow-ct-pretrain-bs64-10pc-sample-less-negs/less_10_pc_sample/bs64_contrastive_sample_temp_$sampling_temp/train.jsonl"

trained_model_name=$1
training_data=$2
eval_data=$3
model_to_train=$4
port=$5


deepspeed --include localhost:0,1,2,3 --master_port $port --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir /data/user_data/jmcoelho/models/fine-tuned/$trained_model_name \
  --model_name_or_path /data/user_data/jmcoelho/models/$model_to_train \
  --dataset_path $training_data \
  --eval_dataset_path $eval_data \
  --dataset_cache_dir /data/datasets/hf_cache \
  --cache_dir /data/datasets/hf_cache \
  --save_steps 1000 \
  --eval_steps 25 \
  --evaluation_strategy steps \
  --bf16 \
  --pooling eos \
  --loss contrastive \
  --gradient_checkpointing \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 32 \
  --train_group_size 10 \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 1024 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 2 \
  --report_to wandb \
  --run_name $trained_model_name


rm /data/user_data/jmcoelho/models/fine-tuned/$trained_model_name/model.safetensors