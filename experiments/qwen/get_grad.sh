#!/bin/bash

#SBATCH --job-name=qwen-retriever-train
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --gres=gpu:6000Ada:8
#SBATCH --time=2-00:00:00


eval "$(conda shell.bash hook)"
conda activate tevatron

#export TRANSFORMERS_CACHE=/data/datasets/hf_cache
export HF_HOME=/data/datasets/hf_cache
export HF_TOKEN=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

model_to_train=Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-random-20k-1gpu
trained_model_name=Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-random-20k-1gpu-valid-grads
group_size=10
pooling=wavg
port=$((RANDOM % (23000 - 20000 + 1) + 20000))


deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port $port --module tevatron.retriever.driver.get_gradients \
  --deepspeed deepspeed/ds_zero3_config.json \
  --dataset_cache_dir /data/datasets/hf_cache \
  --cache_dir /data/datasets/hf_cache \
  --output_dir "/data/user_data/jmcoelho/models/$model_to_train-valid-grads"  \
  --model_name_or_path "/data/user_data/jmcoelho/models/$model_to_train" \
  --dataset_path /data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-finetune-ep1/pretokenized/val.jsonl \
  --save_steps 1000000 \
  --bf16 \
  --pooling $pooling \
  --gradient_checkpointing \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 60 \
  --train_group_size $group_size \
  --learning_rate 0 \
  --query_max_len 32 \
  --passage_max_len 512 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 1 \
  --report_to wandb \
  --run_name $trained_model_name
