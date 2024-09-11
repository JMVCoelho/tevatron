#!/bin/bash
#SBATCH --job-name=pythia_dr_inference
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=00:30:00

EMBEDDING_OUTPUT_DIR=/data/user_data/jmcoelho/embeddings/marco_docs
MODELS_DIR=/data/user_data/jmcoelho/models/fine-tuned

trained_model_name=$1
save_pretok=$2

# remove embeddings
rm $EMBEDDING_OUTPUT_DIR/$trained_model_name/*.pkl

# remove pretokenized data
rm -r $save_pretok

#remove model
#rm -r $MODELS_DIR/$trained_model_name

echo "Done".