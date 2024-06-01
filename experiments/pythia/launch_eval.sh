model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs64-10pc-sample-less-negs

final_model_name=$model_to_train-self-hn1-random-top-100-1090mixtrue-t1

# final_model_name=$model_to_train-self-hn1-rande2-momentum-headreplace-3
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/random_10_pc_sample/random_top100_momentum_headreplace_3

# do_pretok=True 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/random_train_run_splits/random/10pc.train.random.momentum.3headrep.txt



JOB2_ID=$(sbatch experiments/pythia/launch_pythia_docs_inference.sh $final_model_name 0 | awk '{print $NF}')
JOB3_ID=$(sbatch experiments/pythia/launch_pythia_docs_inference.sh $final_model_name 1 | awk '{print $NF}')
JOB4_ID=$(sbatch experiments/pythia/launch_pythia_docs_inference.sh $final_model_name 2 | awk '{print $NF}')
JOB5_ID=$(sbatch experiments/pythia/launch_pythia_docs_inference.sh $final_model_name 3 | awk '{print $NF}')
JOB6_ID=$(sbatch experiments/pythia/launch_pythia_docs_inference_q.sh $final_model_name | awk '{print $NF}')

echo "Submitted batch job $JOB2_ID"
echo "Submitted batch job $JOB3_ID"
echo "Submitted batch job $JOB4_ID"
echo "Submitted batch job $JOB5_ID"
echo "Submitted batch job $JOB6_ID"

JOB7_ID=$(sbatch -d afterok:$JOB2_ID,$JOB3_ID,$JOB4_ID,$JOB5_ID,$JOB6_ID experiments/pythia/launch_pythia_docs_search.sh $final_model_name | awk '{print $NF}')
echo "Results Job: $JOB7_ID"

sbatch -d afterok:$JOB7_ID experiments/pythia/clean_up.sh $final_model_name


