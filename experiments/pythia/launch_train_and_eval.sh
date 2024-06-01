model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs64-10pc-sample-less-negs

final_model_name=$model_to_train-self-hn1-random-top-100-18negs-joint-random-denoised-less
save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/less_10_pc_sample/bs64_contrastive_sample_top_100_18negs-joint-random-denoised-less

do_pretok=False 
training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/less_train_run_splits/less_grad_bs64_temperature_top100/joint/joint.10pc.random-denoisedless.txt

# final_model_name=$model_to_train-self-hn1-rande2-denoised-18negs
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/random_10_pc_sample/random_top100_18negs-random-denoised

# do_pretok=False
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/random_train_run_splits/random/10pc.train.random.18negs.momentum.3headrep.txt

port=$((RANDOM % (23000 - 20000 + 1) + 20000))

if [ "$do_pretok" = "True" ]; then
    mkdir -p $save_pretok
    JOB0_ID=$(sbatch -d afterok:317659 experiments/pythia/launch_pretok.sh $model_to_train $save_pretok $training_data | awk '{print $NF}')
    echo "Submitted batch job $JOB0_ID"

    JOB1_ID=$(sbatch -d afterok:$JOB0_ID experiments/pythia/launch_pythia_docs_selfhn1.sh $final_model_name $save_pretok/train.jsonl $model_to_train $port | awk '{print $NF}')
    echo "Submitted batch job $JOB1_ID"
else
    JOB1_ID=$(sbatch experiments/pythia/launch_pythia_docs_selfhn1.sh $final_model_name $save_pretok/train.jsonl $model_to_train $port | awk '{print $NF}')
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


