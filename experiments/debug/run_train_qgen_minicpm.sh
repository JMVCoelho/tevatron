#!/bin/bash
#SBATCH --job-name=gen_queries_train
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:A6000:2
#SBATCH --mem=50G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-4-36,babel-8-3,babel-4-28

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-12.4
#export TRANSFORMERS_CACHE=/data/datasets/hf_cache
export TRANSFORMERS_CACHE=/data/group_data/cx_group/query_generation_data/hf_cache


trained_model_name=minicpm-2b-stf-bf6-gpt4-query-generator

deepspeed --include localhost:0,1 scripts/minicpm_query_generator_train.py \
    --model_name_or_path openbmb/MiniCPM-2B-sft-bf16 \
    --output_dir /data/user_data/jmcoelho/models/query_generators/$trained_model_name \
    --learning_rate 5e-5 --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 --bf16 \
    --gradient_accumulation_steps 4 --warmup_steps 750 \
    --weight_decay 0.01 \
    --evaluation_strategy steps --eval_steps 1500 \
    --save_steps 10000 \
    --seed 42 \
    --logging_strategy steps --logging_steps 1 \
    --deepspeed deepspeed/ds_zero2_config.json \
    --report_to wandb \
    --run_name $trained_model_name