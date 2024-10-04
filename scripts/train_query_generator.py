from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, DistributedSampler
import argparse
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
import ast


class T5QueryGenerator():
    # This should actually work with any model that supports HF's generate call, not just t5s.
    def __init__(self, base_model="t5-base", max_tokens=512, device='cuda'):
        self.max_tokens = max_tokens
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_train = self.get_model_train(base_model, self.device)
        self.tokenizer = self.get_tokenizer(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def get_model_train(base_model, device):
        return AutoModelForSeq2SeqLM.from_pretrained(base_model).to(device)
    
    @staticmethod
    def get_tokenizer(base_model):
        return AutoTokenizer.from_pretrained(base_model)
    
    def tokenize_train(self, batch):
        texts = []
        labels = []
        for example in batch:
            document = example['doc']
            query = example['query']
            texts.append(f'Generate a query for this document: {document}.')
            labels.append(query)
            
        tokenized = self.tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=self.max_tokens)
        tokenized['labels'] = self.tokenizer(labels, return_tensors='pt', padding=True, truncation='longest_first')['input_ids']
        
        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)

        return tokenized
    
    def tokenize_inference(self, batch):
        texts = []
        for example in batch:
            document = example
            texts.append(f'Generate a query for this document: {document}.')
            
        tokenized = self.tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=self.max_tokens)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)

        return tokenized
    
    # documents: list of strings; each string a document.
    def inference(self, documents, batchsize=300, generation_config=None):
        model_eval = self.model_train.eval()
        generation_config = generation_config if generation_config is not None else model_eval.generation_config

        def batch(X, batch_size=1):
            l = len(X)
            for idx in range(0, l, batch_size):
                yield X[idx:min(idx + batch_size, l)]

        outputs = []
        for sample in tqdm(batch(documents, batchsize)):
            inputs = self.tokenize_inference(sample)
            sample_outputs = model_eval.generate(**inputs, generation_config=generation_config)
            outputs.extend(sample_outputs)

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    


class NonShuffleSeq2SeqTrainer(Seq2SeqTrainer):
    # file should be preshuffled
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        
        # TODO: check world-size and add an if statement here.
        train_sampler = SequentialSampler(self.train_dataset)
        #train_sampler = DistributedSampler(self.train_dataset)
        
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle = False
        )
        
class QueryGenerationDatasetMemory(Dataset):
    # This dataset expects a jsonl file 
    def __init__(self, filename):
        self._filename = filename
        self._total_data = 0
        self.lines = None
        with open(filename, "r") as file:
            self.lines = [line for line in file]

        self._total_data = int(len(self.lines)-1)

    def __getitem__(self, idx):
        line = self.lines[idx]
        data = ast.literal_eval(line)
        query = data['query'] if 'query' in data else data['user_query']

        if 'text' in data:
            doc = data['text']
        elif 'positive_document' in data:
            doc = data['positive_document']
        else:
            doc = data['metadata']['doc_a']
        
        return {'query': query, 'doc': doc}

    def __len__(self):
        return self._total_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default='EleutherAI/pile-t5-large', type=str, required=False,
                        help="Base model to fine tune.")
    parser.add_argument("--max_tokens", default=512, type=int, required=False,
                        help="tokenizer max tokens")
    parser.add_argument("--train_pairs_path", type=str, required=True,
                        help="Triples.tsv path")
    parser.add_argument("--eval_pairs_path", type=str, required=True,
                        help="Triples.tsv path")                   
    parser.add_argument("--output_model_path", default=None, type=str, required=True,
                        help="Path for trained model and checkpoints.")
    parser.add_argument("--save_every_n_steps", default=0, type=int, required=False,
                        help="Save every N steps. (recommended 10000)")
    parser.add_argument("--logging_steps", default=1, type=int, required=False,
                        help="Logging steps parameter.")
    parser.add_argument("--per_device_train_batch_size", default=16, type=int, required=False,
                        help="Per device batch size parameter.")
    parser.add_argument("--per_device_eval_batch_size", default=100, type=int, required=False,
                        help="Per device batch size parameter.")
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int, required=False,
                        help="Gradient accumulation parameter.")
    parser.add_argument("--learning_rate", default=3e-4, type=float, required=False,
                        help="Learning rate parameter.")
    parser.add_argument("--epochs", default=10, type=int, required=False,
                        help="Number of epochs to train")
    parser.add_argument("--warmup_steps", default=250, type=int, required=False,
                        help="Number of warmup steps")
    parser.add_argument("--wandb_run_name", default=None, type=str, required=True,
                        help="WandB run name")
    parser.add_argument("--dataloader_num_workers", default=0, type=int, required=False,
                        help="Num workers for dataloader")

    device = 'cuda'
    torch.manual_seed(123)
    args = parser.parse_args()

    query_generator = T5QueryGenerator(base_model=args.base_model, max_tokens=args.max_tokens, device=device)


    dataset_train = QueryGenerationDatasetMemory(args.train_pairs_path)
    dataset_eval = QueryGenerationDatasetMemory(args.eval_pairs_path)

    if args.save_every_n_steps:
        steps = args.save_every_n_steps
        strategy = 'steps'
    else:
        steps = 1
        strategy = 'epoch'

    def compute_metrics(eval_preds):        
        metric_bleu = evaluate.load("bleu")
        metric_rouge = evaluate.load("rouge")
        logits, labels = eval_preds

        # Ignore tokens with -100:

        for logit in logits:
            for i in range(len(logit)):
                if logit[i] < 0:
                    logit[i] = 0

        for label in labels:
            for i in range(len(label)):
                if label[i] < 0:
                    label[i] = 0

        # Decode to string:

        str_labels = [query_generator.tokenizer.decode(k) for k in labels]
        str_preds = [query_generator.tokenizer.decode(k) for k in logits]


        bleu = metric_bleu.compute(predictions=str_preds, references=str_labels)['bleu']
        rouge = metric_rouge.compute(predictions=str_preds, references=str_labels)

        reported_metrics = {'bleu': bleu}
        reported_metrics.update(rouge)
        return reported_metrics

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_model_path,
        do_train=True,
        eval_steps=750,
        evaluation_strategy = "steps",
        save_strategy=strategy,
        save_steps =steps, 
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=5e-5,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        adafactor=True,
        seed=1,
        disable_tqdm=False,
        load_best_model_at_end=False,
        predict_with_generate=True,
        dataloader_pin_memory=False,
        dataloader_num_workers = args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=args.wandb_run_name,
    )

    trainer = NonShuffleSeq2SeqTrainer(
        model=query_generator.model_train,
        args=train_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        tokenizer=query_generator.tokenizer,
        data_collator=query_generator.tokenize_train,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

    trainer.save_model(args.output_model_path)
    trainer.save_state()

if __name__ == '__main__':
    main()