#!/bin/bash
#SBATCH --job-name=pythia_dr
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:6000Ada
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-4-28

export TRANSFORMERS_CACHE=/data/datasets/hf_cache

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

#model_name=$1
#port=$2
model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision-unsupervised-lower-valid-loss-delete
iii=$1
chkpt=checkpoint-$iii
port=$((RANDOM % (23000 - 20000 + 1) + 20000))
#model_to_valid=/data/user_data/jmcoelho/models/fine-tuned/$model_name
model_to_valid=/data/user_data/jmcoelho/models/fine-tuned/$model_name/$chkpt
rm /data/user_data/jmcoelho/models/fine-tuned/$model_name/$chkpt/model.safetensors

cp /data/user_data/jmcoelho/models/fine-tuned/$model_name/special_tokens_map.json /data/user_data/jmcoelho/models/fine-tuned/$model_name/$chkpt
cp /data/user_data/jmcoelho/models/fine-tuned/$model_name/tokenizer_config.json /data/user_data/jmcoelho/models/fine-tuned/$model_name/$chkpt
cp /data/user_data/jmcoelho/models/fine-tuned/$model_name/tokenizer.json /data/user_data/jmcoelho/models/fine-tuned/$model_name/$chkpt


model_valid_data=pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision


deepspeed --include localhost:0 --master_port $port --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir temp_z$iii \
  --model_name_or_path $model_to_valid \
  --dataset_path /data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_valid_data/random_all_queries_10k_two_valid/val_1.jsonl \
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
  --gradient_accumulation_steps 1 \
  --report_to wandb \
  --run_name $model_name-valid