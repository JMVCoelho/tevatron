# model_to_train=pythia-160m-1024-marco-docs-bow-contrastive-pretrain
# prefix=pre-trained
# final_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/random_all_queries_top100_10k2v_T0.1
# do_pretok=True 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/group_level_10000_two_valid_orcale/group_hardnegs_softmax_t0.1

# model_to_train=pythia-160m-1024-marco-docs-bow-contrastive-pretrain
# prefix=pre-trained
# final_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-random
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/random_all_queries_top100_10k2v_random
# do_pretok=True 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/random_train_run_splits/random/full.stain+val.random.top100.txt 


# model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-10-group-level-T0.1-self-hn-1-random-negs-momentum
# prefix=fine-tuned
# final_model_name=$model_to_train-self-hn-2-random-negs-2momentum
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/random_momentum_2
# do_pretok=True 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/random_train_run_splits/random/full.queries.train+val.random.top100.2momentum.txt

# model_to_train=pythia-160m-1024-marco-docs-bow-contrastive-pretrain
# prefix=pre-trained
# final_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-top200-T0.1
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/random_all_queries_10g_top100_10k2v_top200_T0.1
# do_pretok=True 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/group_level_10000_two_valid_orcale_top200/group_hardnegs_softmax_t0.1

# model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision
# prefix=fine-tuned
# final_model_name=$model_to_train-unsupervised-mates-best-llama-queries-sample-wise-v3-loss-filter
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/mates-llama-best-sample_v3_loss_filter
# do_pretok=True 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/llama_query_influence_mates_sample_level_v3/lower_loss_filtered_ready

# valid_data_path=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/random_all_queries_10k_two_valid
# n_val=0

# model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision
# prefix=fine-tuned
# perc=$1
# final_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-supervised-warmup-minicpm-gpt4-teacher-shn-$perc-CE-filter
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/gpt4-teacher-queries-shn-neg-$perc-ce
# do_pretok=True 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/gen11-shnegs/queries.random.shn.top100.$perc.txt

# valid_data_path=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/random_all_queries_10k_two_valid
# n_val=0


# model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision
# prefix=fine-tuned
# final_model_name=pythia-160m-marco-docs-bow-ct-pretrain-small-supervision-bs256-generator-dpo-final-top-logprobs
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/ss-generator-dpo-final-top-logprobs
# do_pretok=False 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/gen14-shnegs/queries.random.shn.top100.txt

# valid_data_path=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/random_all_queries_10k_two_valid
# n_val=0


# model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs256-generator-dpo-final-top-logprobs
# prefix=fine-tuned
# final_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-generator-dpo-final-top-logprobs-e2
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/generator-dpo-final-top-logprobs-e2
# do_pretok=True 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/gen14-shnegs/queries.random.shn.top100.momentum.txt

# valid_data_path=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/random_all_queries_10k_two_valid
# n_val=0

# model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs256-generator-marco-greedy-ALL-ep1
# prefix=fine-tuned
# final_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-generator-marco-greedy-ALL-ep2-momentum
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/generator-marco-greedy-ALL-ep2-momentum
# do_pretok=True 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/gen17-shnegs/queries.random.shn.top100.momentum.txt 

# valid_data_path=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/random_all_queries_10k_two_valid
# n_val=0

model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs256-llama-clueweb-supervision
prefix=fine-tuned
final_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-llama-clueweb-supervision-e2
save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/marco-train-ALL-e2
do_pretok=True 
training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/marco-train/queries.random.shn.top100.momentum.txt 

valid_data_path=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/random_all_queries_10k_two_valid
n_val=0


# model_to_train=pythia-160m-marco-docs-bow-ct-pretrain-bs256-generator-marco-greedy--ep1
# prefix=fine-tuned
# final_model_name=pythia-160m-marco-docs-bow-ct-pretrain-bs256-generator-marco-greedy--ep2
# save_pretok=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/$model_to_train/generator-marco-greedy-ep2
# do_pretok=True 
# training_data=/data/user_data/jmcoelho/embeddings/marco_docs/$model_to_train/gen17-shnegs/queries.random.shn.top100.txt

# valid_data_path=/data/user_data/jmcoelho/datasets/marco/documents/processed_data/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/random_all_queries_10k_two_valid
# n_val=0

port=$((RANDOM % (23000 - 20000 + 1) + 20000))

if [ "$do_pretok" = "True" ]; then
    mkdir -p $save_pretok
    JOB0_ID=$(sbatch experiments/pythia/launch_pretok.sh $prefix/$model_to_train $save_pretok $training_data $n_val | awk '{print $NF}')
    echo "Submitted batch job $JOB0_ID"

    JOB1_ID=$(sbatch -d afterok:$JOB0_ID experiments/pythia/launch_pythia_docs_selfhn1.sh $final_model_name $save_pretok/train.jsonl $valid_data_path/val_1.jsonl $prefix/$model_to_train $port | awk '{print $NF}')
    echo "Submitted batch job $JOB1_ID"
else
    JOB1_ID=$(sbatch experiments/pythia/launch_pythia_docs_selfhn1.sh $final_model_name $save_pretok/train.jsonl $valid_data_path/val_1.jsonl $prefix/$model_to_train $port | awk '{print $NF}')
    echo "Submitted batch job $JOB1_ID"
fi

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