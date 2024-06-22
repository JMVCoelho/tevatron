
# model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs128-all-queries-less-topk-negs-top50
# prefix=fine-tuned
# final_model_name=$model_to_train-self-hn1-random-negs-3momentum-denoise
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/random_all_queries_top100_3mom_denoise
# do_pretok=True 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/random_train_run_splits/random/full.queries.train+val.random.3.momentum.denoise.348182

# model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs128-all-queries-random-negs-top100
# prefix=fine-tuned
# final_model_name=$model_to_train-self-hn1-random-negs-3momentum-denoise
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/random_all_queries_top100_3mom_denoise
# do_pretok=True 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/random_train_run_splits/random/full.stain+val.random.top100.txt.3momentum.denoise


# model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs128-all-queries-less-5-group-level-best
# prefix=fine-tuned
# final_model_name=$model_to_train-self-hn1-less-5-group-level-with-random-best
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/less_group_level_random_best
# do_pretok=True 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/group_level_with_random/group_hardnegs_full_best

# model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-random-negs-top100
# prefix=fine-tuned
# final_model_name=$model_to_train-self-hn1-random-negs
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/random
# do_pretok=True 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/random_train_run_splits/random/full.queries.train+val.random.top100.txt


# model_to_train=pythia-160m-1024-marco-docs-bow-contrastive-pretrain
# prefix=pre-trained
# final_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-valid-5-group-level-worst
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/valid-oracle-group-level-worst
# do_pretok=True 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/group_level_valid_orcale/group_hardnegs_full_worst

# model_to_train=pythia-160m-1024-marco-docs-bow-contrastive-pretrain
# prefix=pre-trained
# final_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-random-negs-top100
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/random_all_queries_top100
# do_pretok=False 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/group_level_valid_orcale/group_hardnegs_full_worst

# model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-valid-5-group-level-best
# prefix=fine-tuned
# final_model_name=$model_to_train-self-hn1-less-5-group-level-worst-20pc-subset
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/less_group_level_random_worst_subset_20pc
# do_pretok=False 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/group_level_valid_orcale/group_hardnegs_full_best

model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-valid-5-group-level-best
prefix=fine-tuned
final_model_name=$model_to_train-self-hn1-less-5-group-level-5k-valid-worst
save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/group_level_valid_5k_worst
do_pretok=True 
training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/group_level_5k_valid_orcale/group_hardnegs_full_worst

n_val=5000

port=$((RANDOM % (23000 - 20000 + 1) + 20000))

if [ "$do_pretok" = "True" ]; then
    mkdir -p $save_pretok
    JOB0_ID=$(sbatch experiments/pythia/launch_pretok.sh $prefix/$model_to_train $save_pretok $training_data $n_val | awk '{print $NF}')
    echo "Submitted batch job $JOB0_ID"

    JOB1_ID=$(sbatch -d afterok:$JOB0_ID experiments/pythia/launch_pythia_docs_selfhn1.sh $final_model_name $save_pretok/train.jsonl $prefix/$model_to_train $port | awk '{print $NF}')
    echo "Submitted batch job $JOB1_ID"
else
    JOB1_ID=$(sbatch experiments/pythia/launch_pythia_docs_selfhn1.sh $final_model_name $save_pretok/train.jsonl $prefix/$model_to_train $port | awk '{print $NF}')
    echo "Submitted batch job $JOB1_ID"
fi


JOB2_ID=$(sbatch -d afterok:$JOB1_ID experiments/pythia/launch_pythia_docs_inference.sh $final_model_name 0 | awk '{print $NF}')
JOB3_ID=$(sbatch -d afterok:$JOB1_ID experiments/pythia/launch_pythia_docs_inference.sh $final_model_name 1 | awk '{print $NF}')
JOB4_ID=$(sbatch -d afterok:$JOB1_ID experiments/pythia/launch_pythia_docs_inference.sh $final_model_name 2 | awk '{print $NF}')
JOB5_ID=$(sbatch -d afterok:$JOB1_ID experiments/pythia/launch_pythia_docs_inference.sh $final_model_name 3 | awk '{print $NF}')
JOB6_ID=$(sbatch -d afterok:$JOB1_ID experiments/pythia/launch_pythia_docs_inference_q.sh $final_model_name | awk '{print $NF}')
JOB7_ID=$(sbatch -d afterok:$JOB1_ID experiments/pythia/launch_pythia_docs_inference_valid_q.sh $final_model_name | awk '{print $NF}')

echo "Submitted batch job $JOB2_ID"
echo "Submitted batch job $JOB3_ID"
echo "Submitted batch job $JOB4_ID"
echo "Submitted batch job $JOB5_ID"
echo "Submitted batch job $JOB6_ID"
echo "Submitted batch job $JOB7_ID"

JOB8_ID=$(sbatch -d afterok:$JOB2_ID,$JOB3_ID,$JOB4_ID,$JOB5_ID,$JOB6_ID experiments/pythia/launch_pythia_docs_search.sh $final_model_name | awk '{print $NF}')
echo "Results Test MRR: $JOB8_ID"

JOB9_ID=$(sbatch -d afterok:$JOB2_ID,$JOB3_ID,$JOB4_ID,$JOB5_ID,$JOB7_ID experiments/pythia/launch_pythia_docs_search_valid.sh $final_model_name | awk '{print $NF}')
echo "Results Valid MRR: $JOB9_ID"

JOB10_ID=$(sbatch -d afterok:$JOB8_ID,$JOB9_ID experiments/pythia/launch_valid.sh $final_model_name $port | awk '{print $NF}')
echo "Results Valid loss: $JOB10_ID"

sbatch -d afterok:$JOB10_ID experiments/pythia/clean_up.sh $final_model_name $save_pretok