#!/bin/bash
#SBATCH --job-name=pyserini
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=48
#SBATCH --mem=150G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=babel-4-36,babel-8-3,babel-4-28


eval "$(conda shell.bash hook)"
conda activate base

java -version
# python -m pyserini.index.lucene \
#   --collection JsonCollection \
#   --input /data/user_data/jmcoelho/datasets/marco/documents/pyserini_raw \
#   --index /data/group_data/cx_group/lucene_indexes/marco_docs \
#   --generator DefaultLuceneDocumentGenerator \
#   --threads 24 \
#   --storePositions --storeDocvectors --storeRaw


python -m pyserini.search.lucene \
  --index /data/group_data/cx_group/lucene_indexes/marco_docs \
  --topics /data/user_data/jmcoelho/datasets/marco/documents/gen6.query.tsv \
  --output /data/user_data/jmcoelho/datasets/marco/documents/gen6.bm25.run.trec \
  --batch-size 48 \
  --threads 48 \
  --bm25