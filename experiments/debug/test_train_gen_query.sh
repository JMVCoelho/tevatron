#!/bin/bash
#SBATCH --job-name=gen_queries_train
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-4-36,babel-8-3,babel-4-28


eval "$(conda shell.bash hook)"
conda activate tevatron

export TRANSFORMERS_CACHE=/data/datasets/hf_cache

final_model_name=pilet5-large-swim-ir-query-gen-512

python scripts/train_query_generator.py \
    --output_model_path /data/user_data/jmcoelho/models/query_generators/$final_model_name \
	--wandb_run_name $final_model_name \
    --train_pairs_path /data/group_data/cx_group/swim_ir_v1/monolingual/en/train.jsonl \
    --eval_pairs_path /data/group_data/cx_group/swim_ir_v1/monolingual/en/val.jsonl \
   --base_model EleutherAI/pile-t5-large \
   --max_tokens 512 \
   --save_every_n_steps 0 \
   --per_device_train_batch_size 12 \
   --per_device_eval_batch_size 200 \
   --gradient_accumulation_steps 8 \
   --warmup_steps 400 \
   --epochs 1 \
   --dataloader_num_workers 0

# final_model_name=pilet5-large-llama-query-gen-512

# python scripts/train_query_generator.py \
#     --output_model_path /data/user_data/jmcoelho/models/query_generators/$final_model_name \
# 	--wandb_run_name $final_model_name \
#     --train_pairs_path /data/group_data/cx_group/query_generation_data/train.jsonl \
#     --eval_pairs_path /data/group_data/cx_group/query_generation_data/val.jsonl \
#    --base_model EleutherAI/pile-t5-large \
#    --max_tokens 512 \
#    --save_every_n_steps 0 \
#    --per_device_train_batch_size 12 \
#    --per_device_eval_batch_size 200 \
#    --warmup_steps 400 \
#    --epochs 1 \
#    --dataloader_num_workers 0 