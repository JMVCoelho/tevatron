model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-random-negs-top100
prefix=fine-tuned
final_model_name=$model_to_train-self-hn1-random-negs
save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/random
do_pretok=True 
training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/random_train_run_splits/random/full.queries.train+val.random.top100.txt

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

JOB2_ID=$(sbatch -d afterok:$JOB1_ID experiments/pythia/launch_valid.sh $final_model_name $port | awk '{print $NF}')
echo "Valid results job $JOB2_ID"

sbatch -d afterok:$JOB2_ID experiments/pythia/clean_up.sh $final_model_name $save_pretok