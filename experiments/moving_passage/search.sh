#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=babel-4-28,babel-1-27,babel-8-11


eval "$(conda shell.bash hook)"
conda activate tevatron

module load cuda-11.8

trained_model_name=pythia-160m-marco-docs-bow-subset-3
EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs


python scripts/moving_passage_reduce_index.py $trained_model_name
python scripts/moving_passage_reduce_query.py $trained_model_name

for i in {0..9}; do

    cp $EMBEDDING_OUTPUT_DIR/moving_passage/$trained_model_name/corpus.$i.pkl $EMBEDDING_OUTPUT_DIR/moving_passage/$trained_model_name/reduced_index/corpus.1.pkl
    
    set -f && python -m tevatron.retriever.driver.search \
        --query_reps $EMBEDDING_OUTPUT_DIR/moving_passage/$trained_model_name/reduced_index/qry-dev-reduced.pkl \
        --passage_reps $EMBEDDING_OUTPUT_DIR/moving_passage/$trained_model_name/reduced_index/corpus*.pkl \
        --depth 1000 \
        --batch_size 128 \
        --save_text \
        --save_ranking_to $EMBEDDING_OUTPUT_DIR/moving_passage/$trained_model_name/reduced_index/run.dev.$i.txt

    python src/tevatron/utils/format/convert_result_to_trec.py \
        --input $EMBEDDING_OUTPUT_DIR/moving_passage/$trained_model_name/reduced_index/run.dev.$i.txt \
        --output $EMBEDDING_OUTPUT_DIR/moving_passage/$trained_model_name/reduced_index/run.dev.$i.trec

    qrels=/home/jmcoelho/tevatron/qrels/marco.docs.dev.move.passage.qrel.tsv
    trec_run=$EMBEDDING_OUTPUT_DIR/moving_passage/$trained_model_name/reduced_index/run.dev.$i.trec

    python scripts/evaluate.py $qrels $trec_run > $EMBEDDING_OUTPUT_DIR/moving_passage/$trained_model_name/reduced_index/results.$i.trec
    python scripts/evaluate.py -m mrr_cut.100 $qrels $trec_run

    rm $EMBEDDING_OUTPUT_DIR/moving_passage/$trained_model_name/reduced_index/corpus.1.pkl

    echo "####################"
done

echo "Default"
qrels=/home/jmcoelho/tevatron/qrels/marco.docs.dev.move.passage.qrel.tsv
trec_run=$EMBEDDING_OUTPUT_DIR/$trained_model_name/run.dev.trec
python scripts/evaluate.py -m mrr_cut.100 $qrels $trec_run