#!/bin/bash

#SBATCH --job-name=qwen-retriever-inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=2-00:00:00
#SBATCH --exclude=babel-13-13,babel-14-37,babel-6-9,babel-7-9,babel-3-21,babel-13-25,babel-13-1,babel-14-1,babel-12-9



eval "$(conda shell.bash hook)"
conda activate tevatron-xformers
module load cuda-12.4

trained_model_name=stella_en_400M_v5

EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs/
mkdir $EMBEDDING_OUTPUT_DIR/$trained_model_name

OUTPUT_FILE=$EMBEDDING_OUTPUT_DIR/$trained_model_name/run.gen.random.trec

if [ -f "$OUTPUT_FILE" ]; then
  echo "File $OUTPUT_FILE already exists. Skipping search."
else
  echo "Searching..."

    set -f && OMP_NUM_THREADS=24 python -m tevatron.retriever.driver.search \
        --query_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/query-gen-gumbel.pkl \
        --passage_reps $EMBEDDING_OUTPUT_DIR/$trained_model_name/corpus.cweb.*.pkl \
        --depth 100 \
        --batch_size 128 \
        --save_text \
        --save_ranking_to $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.gen.gumbel.txt


    # python src/tevatron/utils/format/convert_result_to_trec.py \
    #     --input $EMBEDDING_OUTPUT_DIR/$trained_model_name/run.gen.random.txt \
    #     --output $OUTPUT_FILE


    # qrels=/data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv
    # trec_run=$OUTPUT_FILE

    # python scripts/eval_trec.py $qrels $trec_run
    # python scripts/eval_trec.py -m mrr_cut.100 $qrels $trec_run
    
fi