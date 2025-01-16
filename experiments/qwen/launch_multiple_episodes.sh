# Base Model and Directories
BASE_MODEL="Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-minicpmembed-100k-ranker-dpo"
FINAL_MODEL_NAME="Qwen2.5-0.5B-bidirectional-attn-avg-pool-mntp-minicpmembed-100k-ranker-dpo-MARCO-FULL"
EMBEDDING_OUTPUT_DIR="/data/user_data/jmcoelho/embeddings/marco_docs/"
POOLING=avg
NUM_NEGS=9
NUM_EPISODES=1 
MOMENTUM_MODEL=""
PREVIOUS_JOB_ID=""

# Loop over the number of episodes
for EPISODE in $(seq 1 $NUM_EPISODES); do
    # Set the final model name for this episode
    CURRENT_FINAL_MODEL_NAME="${FINAL_MODEL_NAME}${EPISODE}"

    # If this is not the first episode, wait for the previous episode to finish
    if [ -n "$PREVIOUS_JOB_ID" ]; then
        DEPENDENCY_OPTION="-d afterok:$PREVIOUS_JOB_ID"
    else
        DEPENDENCY_OPTION=""  # No dependency for the first episode
    fi

    # Submit the jobs for document and query inference for episode EPISODE
    JOB1_ID=$(sbatch $DEPENDENCY_OPTION experiments/qwen/inference_documents_sharded.sh $BASE_MODEL $POOLING | awk '{print $NF}')
    JOB5_ID=$(sbatch $DEPENDENCY_OPTION experiments/qwen/inference_queries_marco_train.sh $BASE_MODEL $POOLING | awk '{print $NF}')

    echo "Submitted batch job $JOB1_ID"
    echo "Submitted batch job $JOB5_ID"

    JOB6_ID=$(sbatch -d afterok:$JOB1_ID,$JOB5_ID experiments/qwen/search_marco_train.sh $BASE_MODEL | awk '{print $NF}')
    echo "Submitted batch job $JOB6_ID"

    # Conditional Momentum Model Handling
    if [ -z "$MOMENTUM_MODEL" ]; then
        # First episode does not use momentum
        JOB7_ID=$(sbatch -d afterok:$JOB6_ID experiments/qwen/sample_hns.sh /data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv $EMBEDDING_OUTPUT_DIR/$BASE_MODEL/run.train.txt $EMBEDDING_OUTPUT_DIR/$BASE_MODEL/negatives.train.txt $NUM_NEGS | awk '{print $NF}')
    else
        # Subsequent episodes use momentum
        JOB7_ID=$(sbatch -d afterok:$JOB6_ID experiments/qwen/sample_hns.sh /data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv $EMBEDDING_OUTPUT_DIR/$BASE_MODEL/run.train.txt $EMBEDDING_OUTPUT_DIR/$BASE_MODEL/negatives.train.txt $NUM_NEGS $EMBEDDING_OUTPUT_DIR/$MOMENTUM_MODEL/run.train.txt | awk '{print $NF}')
    fi

    echo "Submitted batch job $JOB7_ID"

    # Pre-tokenization for the current episode
    JOB8_ID=$(sbatch -d afterok:$JOB7_ID experiments/qwen/pretokenize.sh $BASE_MODEL $EMBEDDING_OUTPUT_DIR/$BASE_MODEL/pretokenized $EMBEDDING_OUTPUT_DIR/$BASE_MODEL/negatives.train.txt 0 $NUM_NEGS | awk '{print $NF}')
    
    # Training for the current episode
    JOB9_ID=$(sbatch -d afterok:$JOB8_ID experiments/qwen/train_qwen.sh $BASE_MODEL $CURRENT_FINAL_MODEL_NAME $EMBEDDING_OUTPUT_DIR/$BASE_MODEL/pretokenized/train.jsonl $NUM_NEGS $POOLING | awk '{print $NF}')
    echo "Submitted batch job $JOB9_ID"    

    # Prepare for evaluation inference
    JOB10_ID=$(sbatch -d afterok:$JOB9_ID experiments/qwen/inference_documents_sharded.sh $CURRENT_FINAL_MODEL_NAME $POOLING | awk '{print $NF}')
    JOB14_ID=$(sbatch -d afterok:$JOB9_ID experiments/qwen/inference_queries_marco_dev.sh $CURRENT_FINAL_MODEL_NAME $POOLING | awk '{print $NF}')

    echo "Submitted batch job $JOB10_ID"
    echo "Submitted batch job $JOB14_ID"

    JOB15_ID=$(sbatch -d afterok:$JOB10_ID,$JOB14_ID experiments/qwen/search_marco_dev.sh $CURRENT_FINAL_MODEL_NAME | awk '{print $NF}')
    
    echo "Submitted batch job $JOB15_ID - Episode $EPISODE test MRR"

    # Update the base model, momentum model, and set the previous job ID to wait for the next episode
    MOMENTUM_MODEL=$BASE_MODEL
    BASE_MODEL=$CURRENT_FINAL_MODEL_NAME
    PREVIOUS_JOB_ID=$JOB15_ID  # Make sure the next episode depends on this episode finishing
done