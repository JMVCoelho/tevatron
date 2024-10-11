

model_to_train=pythia-160m-1024-marco-docs-bow-contrastive-pretrain
prefix=pre-trained
final_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-llama-clueweb


# model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs256-generator-marco-greedy--ep1
# prefix=fine-tuned
# final_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-generator-marco-greedy--ep2
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/generator-marco-greedy-ep2
# do_pretok=True 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/gen17-shnegs/queries.random.shn.top100.txt

# valid_data_path=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/random_all_queries_10k_two_valid
# n_val=0

port=$((RANDOM % (23000 - 20000 + 1) + 20000))


JOB1_ID=$(sbatch experiments/pythia/launch_pythia_docs_selfhn1_minicpmdata.sh $final_model_name $prefix/$model_to_train $port | awk '{print $NF}')
echo "Submitted batch job $JOB1_ID"


JOB2_ID=$(sbatch -d afterok:$JOB1_ID experiments/pythia/launch_pythia_docs_inference.sh $final_model_name 0 | awk '{print $NF}')
JOB3_ID=$(sbatch -d afterok:$JOB1_ID experiments/pythia/launch_pythia_docs_inference.sh $final_model_name 1 | awk '{print $NF}')
JOB4_ID=$(sbatch -d afterok:$JOB1_ID experiments/pythia/launch_pythia_docs_inference.sh $final_model_name 2 | awk '{print $NF}')
JOB5_ID=$(sbatch -d afterok:$JOB1_ID experiments/pythia/launch_pythia_docs_inference.sh $final_model_name 3 | awk '{print $NF}')
JOB6_ID=$(sbatch -d afterok:$JOB1_ID experiments/pythia/launch_pythia_docs_inference_q.sh $final_model_name | awk '{print $NF}')
#JOB7_ID=$(sbatch -d afterok:$JOB1_ID experiments/pythia/launch_pythia_docs_inference_valid_q.sh $final_model_name | awk '{print $NF}')


echo "Submitted batch job $JOB2_ID"
echo "Submitted batch job $JOB3_ID"
echo "Submitted batch job $JOB4_ID"
echo "Submitted batch job $JOB5_ID"
echo "Submitted batch job $JOB6_ID"
#echo "Submitted batch job $JOB7_ID"

JOB8_ID=$(sbatch -d afterok:$JOB2_ID,$JOB3_ID,$JOB4_ID,$JOB5_ID,$JOB6_ID experiments/pythia/launch_pythia_docs_search.sh $final_model_name | awk '{print $NF}')
echo "Results Test MRR: $JOB8_ID"

# JOB9_ID=$(sbatch -d afterok:$JOB2_ID,$JOB3_ID,$JOB4_ID,$JOB5_ID,$JOB7_ID experiments/pythia/launch_pythia_docs_search_valid.sh $final_model_name | awk '{print $NF}')
# echo "Results Valid MRR: $JOB9_ID"

#sbatch -d afterok:$JOB8_ID,$JOB9_ID experiments/pythia/clean_up.sh $final_model_name $save_pretok
sbatch -d afterok:$JOB8_ID experiments/pythia/clean_up.sh $final_model_name $save_pretok


#jq -r '.query_id' 500.valid.query.jsonl | awk '{print "^"$1"\t"}' | grep -E -f - /home/jmcoelho/tevatron/qrels/marco.docs.val.qrel.tsv > /home/jmcoelho/tevatron/qrels/marco.docs.500val.qrel.tsv

#jq -r '.query_id' 5000.valid.2.query.jsonl | awk '{print "^"$1"\t"}' | grep -E -f - /home/jmcoelho/tevatron/qrels/marco.docs.val.qrel.tsv > /home/jmcoelho/tevatron/qrels/marco.docs.5000val2.qrel.tsv