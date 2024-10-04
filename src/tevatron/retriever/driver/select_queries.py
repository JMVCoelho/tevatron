import logging
import os
import sys
from tevatron.retriever.data_selection import MATESQueryAttributionPAIRS, GranNormQueryAttribution, MATESQueryAttribution, LESSQueryAttribution, GranVarianceQueryAttribution

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
    
    # sampler = MATESQueryAttribution(qrels_path=data_args.train_qrels, 
    #                             run_path=data_args.train_run_path,
    #                             valid_path=data_args.validation_set,
    #                             model_args=model_args,
    #                             data_args=data_args,
    #                             training_args=training_args)

    # sampler = MATESQueryAttributionPAIRS(qrels_path=data_args.train_qrels, 
    #                             run_path=data_args.train_run_path,
    #                             valid_path=data_args.validation_set,
    #                             model_args=model_args,
    #                             data_args=data_args,
    #                             training_args=training_args)
    

    sampler = LESSQueryAttribution(qrels_path=data_args.train_qrels, 
                                run_path=data_args.train_run_path,
                                valid_path=data_args.validation_set,
                                model_args=model_args,
                                data_args=data_args,
                                training_args=training_args,
                                embeddings_path=data_args.embedding_path)


    # sampler = GranNormQueryAttribution(qrels_path=data_args.train_qrels, 
    #                         run_path=data_args.train_run_path,
    #                         valid_path=data_args.validation_set,
    #                         model_args=model_args,
    #                         data_args=data_args,
    #                         training_args=training_args)

    # sampler = GranVarianceQueryAttribution(qrels_path=data_args.train_qrels, 
    #                         run_path=data_args.train_run_path,
    #                         valid_path=data_args.validation_set,
    #                         model_args=model_args,
    #                         data_args=data_args,
    #                         training_args=training_args)

    sampler.set_seed(training_args.seed)
        
    sampler.sample_hard_negatives(n=data_args.number_of_negatives,
                                  outpath=data_args.negatives_out_file)
    
    # sampler.get_initial_valid_loss(outpath=data_args.negatives_out_file)


if __name__ == "__main__":
    main()
