#!/bin/bash

#SBATCH --job-name=qwen-retriever-train
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --gres=gpu:L40S:8
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-6-9,babel-13-5,babel-13-13,babel-12-9


eval "$(conda shell.bash hook)"
conda activate tevatron

export HF_HOME=/data/datasets/hf_cache
module load cuda-12.4
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model_to_train=$1
trained_model_name=$2
data=$3
group_size=$(( $4 + 1 ))
pooling=$5
port=$((RANDOM % (23000 - 20000 + 1) + 20000))

# --checkpoint "/data/user_data/jmcoelho/models/$model_to_train" \


deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port $port --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir /data/user_data/jmcoelho/models/$trained_model_name \
  --model_name_or_path "/data/user_data/jmcoelho/models/$model_to_train" \
  --eval_dataset_path /data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-finetune-ep1/pretokenized/val_shuf_subset.jsonl \
  --eval_steps 100 \
  --per_device_eval_batch_size 100 \
  --evaluation_strategy steps \
  --dataset_path "$data" \
  --save_steps 10000000 \
  --bf16 \
  --pooling $pooling \
  --gradient_checkpointing \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 48 \
  --train_group_size $group_size \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 512 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 2 \
  --report_to wandb \
  --run_name $trained_model_name
