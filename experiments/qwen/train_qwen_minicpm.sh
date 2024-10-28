#!/bin/bash

#SBATCH --job-name=qwen-pretrain
# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id

#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 12 # number cpus (threads) per task

# 327680
#SBATCH --mem=300000 # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit

#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:4


eval "$(conda shell.bash hook)"
conda activate cmu-llms-hw3

model_to_train=Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp
trained_model_name=Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-RR-2.1M-filtered-unsupervised-queries
group_size=6
pooling=wavg
port=$((RANDOM % (23000 - 20000 + 1) + 20000))


deepspeed --include localhost:0,1,2,3 --master_port $port --module tevatron.retriever.driver.train \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir /user/home/jcoelho/Qwen/models/$trained_model_name \
  --model_name_or_path "/user/home/jcoelho/Qwen/models/$model_to_train" \
  --dataset_name "XBKYS/minicpm-embedding-data" \
  --save_steps 1000000 \
  --bf16 \
  --pooling $pooling \
  --gradient_checkpointing \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 60 \
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
