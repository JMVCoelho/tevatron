#!/bin/bash

#SBATCH --job-name=qwen-pretrain
# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id

#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -n 1 # Number of tasks
#SBATCH --cpus-per-task 12 # number cpus (threads) per task

# 327680
#SBATCH --mem=100000 # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit

#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:4


eval "$(conda shell.bash hook)"
conda activate cmu-llms-hw3

trained_model_name=Qwen2.5-0.5B-marco-cpt-512-bidirectional-attn-contrastive-pretrain-avg-pool

deepspeed --include localhost:0,1,2,3 --master_port 26500 --module tevatron.retriever.driver.pretrain \
  --deepspeed deepspeed/ds_zero3_config.json \
  --output_dir /user/home/jcoelho/Qwen/models/$trained_model_name \
  --model_name_or_path "/user/home/jcoelho/Qwen/models/Qwen2.5-0.5B-marco-cpt-512-bidirectional-attn" \
  --dataset_path "/data/jcoelho/datasets/babel/corpus_firstp_2048.jsonl" \
  --save_steps 1000000 \
  --bf16 \
  --pooling avg \
  --gradient_checkpointing \
  --append_eos_token \
  --normalize \
  --temperature 0.01 \
  --per_device_train_batch_size 128 \
  --train_group_size 2 \
  --learning_rate 1e-4 \
  --query_max_len 32 \
  --passage_max_len 512 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 2 \
  --report_to wandb \
  --run_name $trained_model_name
