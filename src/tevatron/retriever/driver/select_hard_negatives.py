import logging
import os
import sys
from tevatron.retriever.data_selection import RandomHardNegatives, InDiHardNegatives, LESSHardNegativesOpacus, MetaHardNegatives

from transformers import (
    HfArgumentParser,
)

from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments
    
    
    if data_args.method == "random":
        sampler = RandomHardNegatives(qrels_path=data_args.train_qrels, 
                                      run_path=data_args.train_run_path)
        sampler.set_seed(training_args.seed)

    elif data_args.method == "indi":
        sampler = InDiHardNegatives(qrels_path=data_args.train_qrels, 
                                    run_path=data_args.train_run_path,
                                    embeddings_path=data_args.embedding_path)
        
    elif data_args.method == "less":
        sampler = LESSHardNegativesOpacus(qrels_path=data_args.train_qrels, 
                                    run_path=data_args.train_run_path,
                                    embeddings_path=data_args.embedding_path,
                                    model_args=model_args,
                                    data_args=data_args,
                                    training_args=training_args)
        sampler.set_seed(training_args.seed)

    elif data_args.method == "meta":
        sampler = MetaHardNegatives(qrels_path=data_args.train_qrels, 
                                    run_path=data_args.train_run_path,
                                    embeddings_path=data_args.embedding_path,
                                    model_args=model_args,
                                    data_args=data_args,
                                    training_args=training_args)
        sampler.set_seed(training_args.seed)
        
    else:
        raise ValueError(f"Not implemented sampler: {data_args.method}")
    
    sampler.sample_hard_negatives(n=data_args.number_of_negatives,
                                  outpath=data_args.negatives_out_file)


if __name__ == "__main__":
    main()
