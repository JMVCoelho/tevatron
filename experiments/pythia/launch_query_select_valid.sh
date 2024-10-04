#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --gres=gpu:L40:1
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-4-28,babel-3-19,babel-5-11

eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

warmed_up_model=pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision-1gpu   #warmedup model was trained with shns from base_model. We'll select from those as well, to minimize grad shifts
base_model=pythia-160m-1024-marco-docs-bow-contrastive-pretrain
prefix=fine-tuned
n_negatives=9

 
# gen10_less_influence | valid is warmed_up_model shn
# gen10_less_influence_v2 | valid is base_model shn     
# gen10_less_influence_v3 | valid is base_model shn + warmed_up_model
subset=$1

python -m tevatron.retriever.driver.select_queries \
    --method valid_query_level \
    --validation_set /data/user_data/jmcoelho/datasets/marco/documents/processed_data/$warmed_up_model/random_all_queries_10k_two_valid/val_1.jsonl \
    --train_run_path /data/user_data/jmcoelho/embeddings/marco_docs/$base_model/gen10-shnegs/run.split.$1 \
    --train_qrels /data/user_data/jmcoelho/datasets/marco/documents/qrels.gen10.tsv \
    --embedding_path /data/user_data/jmcoelho/embeddings/marco_docs/$warmed_up_model/valid_grads_bs64_with_mom/ \
    --number_of_negatives $n_negatives \
    --negatives_out_file /data/user_data/jmcoelho/embeddings/marco_docs/$warmed_up_model/gen10_less_influence_v3/run_split_$1 \
    --output_dir temp \
    --model_name_or_path /data/user_data/jmcoelho/models/$prefix/$warmed_up_model \
    --dataset_cache_dir /data/datasets/hf_cache \
    --cache_dir /data/datasets/hf_cache \
    --bf16 \
    --pooling eos \
    --loss contrastive \
    --append_eos_token \
    --normalize \
    --temperature 0.01 \
    --query_max_len 32 \
    --passage_max_len 512


#--path_to_optim_states /data/user_data/jmcoelho/models/$prefix/$warmed_up_model/checkpoint-100/global_step100/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt \
