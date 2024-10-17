
BASE_MODEL=Qwen2.5-0.5B-marco-cpt-512-contrastive-pretrain
FINAL_MODEL_NAME=Qwen2.5-0.5B-marco-cpt-512-contrastive-pretrain-fine-tune-ep1
EMBEDDING_OUTPUT_DIR=/data/jcoelho/embeddings/babel/
NUM_NEGS=9

# SAMPLE SELF-HARD NEGATIVES 
#############################

# JOB1_ID=$(sbatch experiments/qwen/inference_documents.sh 0 $BASE_MODEL | awk '{print $NF}')
# JOB2_ID=$(sbatch experiments/qwen/inference_documents.sh 1 $BASE_MODEL | awk '{print $NF}')
# JOB3_ID=$(sbatch experiments/qwen/inference_documents.sh 2 $BASE_MODEL | awk '{print $NF}')
# JOB4_ID=$(sbatch experiments/qwen/inference_documents.sh 3 $BASE_MODEL | awk '{print $NF}')
# JOB5_ID=$(sbatch experiments/qwen/inference_queries_marco_train.sh $BASE_MODEL | awk '{print $NF}')

# echo "Submitted batch job $JOB1_ID"
# echo "Submitted batch job $JOB2_ID"
# echo "Submitted batch job $JOB3_ID"
# echo "Submitted batch job $JOB4_ID"
# echo "Submitted batch job $JOB5_ID"

# JOB6_ID=$(sbatch -d afterok:$JOB1_ID,$JOB2_ID,$JOB3_ID,$JOB4_ID,$JOB5_ID experiments/qwen/search_marco_train.sh $BASE_MODEL | awk '{print $NF}')

# echo "Submitted batch job $JOB6_ID"


#JOB7_ID=$(sbatch -d afterok:$JOB6_ID experiments/qwen/sample_hns.sh /data/jcoelho/datasets/babel/qrels.train.tsv $EMBEDDING_OUTPUT_DIR/$BASE_MODEL/run.train.txt $EMBEDDING_OUTPUT_DIR/$BASE_MODEL/negatives.train.txt $NUM_NEGS | awk '{print $NF}')


# PRE-TOK
#############################

#JOB8_ID=$(sbatch -d afterok:$JOB7_ID experiments/qwen/pretokenize.sh $BASE_MODEL $EMBEDDING_OUTPUT_DIR/$BASE_MODEL/pretokenized $EMBEDDING_OUTPUT_DIR/$BASE_MODEL/negatives.train.txt 0 $NUM_NEGS | awk '{print $NF}')

# TRAIN
#############################

JOB9_ID=$(sbatch experiments/qwen/train_qwen.sh $BASE_MODEL $FINAL_MODEL_NAME $EMBEDDING_OUTPUT_DIR/$BASE_MODEL/pretokenized/train.jsonl $NUM_NEGS | awk '{print $NF}')

echo "Submitted batch job $JOB9_ID"

# EVAL
#############################

JOB10_ID=$(sbatch -d afterok:$JOB9_ID experiments/qwen/inference_documents.sh 0 $FINAL_MODEL_NAME | awk '{print $NF}')
JOB11_ID=$(sbatch -d afterok:$JOB9_ID experiments/qwen/inference_documents.sh 1 $FINAL_MODEL_NAME | awk '{print $NF}')
JOB12_ID=$(sbatch -d afterok:$JOB9_ID experiments/qwen/inference_documents.sh 2 $FINAL_MODEL_NAME | awk '{print $NF}')
JOB13_ID=$(sbatch -d afterok:$JOB9_ID experiments/qwen/inference_documents.sh 3 $FINAL_MODEL_NAME | awk '{print $NF}')
JOB14_ID=$(sbatch -d afterok:$JOB9_ID experiments/qwen/inference_queries_marco_dev.sh $FINAL_MODEL_NAME | awk '{print $NF}')

echo "Submitted batch job $JOB10_ID"
echo "Submitted batch job $JOB11_ID"
echo "Submitted batch job $JOB12_ID"
echo "Submitted batch job $JOB13_ID"
echo "Submitted batch job $JOB14_ID"

JOB6_ID=$(sbatch -d afterok:$JOB10_ID,$JOB11_ID,$JOB12_ID,$JOB13_ID,$JOB14_ID experiments/qwen/search_marco_dev.sh $FINAL_MODEL_NAME | awk '{print $NF}')

echo "Submitted batch job $JOB6_ID"