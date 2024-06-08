# model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs128-20pc-sample-less-negs-triplet-topk
# prefix=fine-tuned
# final_model_name=$model_to_train-self-hn1-less-negs-topk-denoised-valid
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/less_20_pc_sample_topk_denoised_valid
# do_pretok=True 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/20pc-sample-run-splits/less-opacus-tripled-denoised-valid/hardnegs_less_opacus.20.pc.full.topk

# model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs128-20pc-sample-less-negs-triplet-topk
# prefix=fine-tuned
# final_model_name=$model_to_train-self-hn1-random-negs-6negs
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/random_20_pc_sample_6negs
# do_pretok=True 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/random_train_run_splits/random/20pc.tain+val.random.6negs.txt


model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs128-all-queries-less-topk-negs-top50
prefix=fine-tuned
final_model_name=$model_to_train-self-hn1-random-negs
save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/random_all_queries_top100
do_pretok=True 
training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/random_train_run_splits/random/full.queries.train+val.random.txt 


# model_to_train=pythia-160m-1024-marco-docs-bow-contrastive-pretrain
# prefix=pre-trained
# final_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs128-all-queries-less-topk-negs-top50
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/less_topk_all_queries_top50
# do_pretok=True 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/full-queries-run-splits/less-opacus-triplet/hardnegs_less_opacus.all.queries.full.topk


port=$((RANDOM % (23000 - 20000 + 1) + 20000))

if [ "$do_pretok" = "True" ]; then
    mkdir -p $save_pretok
    JOB0_ID=$(sbatch experiments/pythia/launch_pretok.sh $prefix/$model_to_train $save_pretok $training_data | awk '{print $NF}')
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

echo "Submitted batch job $JOB2_ID"
echo "Submitted batch job $JOB3_ID"
echo "Submitted batch job $JOB4_ID"
echo "Submitted batch job $JOB5_ID"
echo "Submitted batch job $JOB6_ID"

JOB7_ID=$(sbatch -d afterok:$JOB2_ID,$JOB3_ID,$JOB4_ID,$JOB5_ID,$JOB6_ID experiments/pythia/launch_pythia_docs_search.sh $final_model_name | awk '{print $NF}')
echo "Results Job: $JOB7_ID"

sbatch -d afterok:$JOB7_ID experiments/pythia/clean_up.sh $final_model_name $save_pretok