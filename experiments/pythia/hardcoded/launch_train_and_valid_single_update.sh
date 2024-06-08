
model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs128-20pc-sample-less-negs-triplet-topk
prefix=fine-tuned
single_neg_index=$1
final_model_name=$model_to_train-single-update-$single_neg_index
save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/temp_$single_neg_index
do_pretok=True 
training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/20pc-sample-run-splits/less-opacus-triplet/single_training_examples/test_$single_neg_index

cat $training_data


port=$((RANDOM % (23000 - 20000 + 1) + 20000))

if [ "$do_pretok" = "True" ]; then
    mkdir -p $save_pretok
    JOB0_ID=$(sbatch experiments/pythia/launch_pretok.sh $prefix/$model_to_train $save_pretok $training_data | awk '{print $NF}')
    echo "Submitted batch job $JOB0_ID"

    JOB1_ID=$(sbatch -d afterok:$JOB0_ID experiments/pythia/launch_pythia_docs_selfhn1_single_example.sh $final_model_name $save_pretok/val.jsonl $prefix/$model_to_train $port | awk '{print $NF}')
    echo "Submitted batch job $JOB1_ID"
else
    JOB1_ID=$(sbatch experiments/pythia/launch_pythia_docs_selfhn1_single_example.sh $final_model_name $save_pretok/val.jsonl $prefix/$model_to_train $port | awk '{print $NF}')
    echo "Submitted batch job $JOB1_ID"
fi


# JOB2_ID=$(sbatch -d afterok:$JOB1_ID experiments/pythia/launch_valid.sh $final_model_name $port | awk '{print $NF}')
# echo "Submitted batch job $JOB2_ID"

# sbatch -d afterok:$JOB2_ID experiments/pythia/clean_up.sh $final_model_name $save_pretok