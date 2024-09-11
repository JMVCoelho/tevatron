from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

class T5QueryGenerator():
    def __init__(self, base_model="t5-base", max_tokens=512, device='cuda'):
        self.max_tokens = max_tokens
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_train = self.get_model_train(base_model, self.device)
        self.tokenizer = self.get_tokenizer(base_model)

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
            texts.append(f'Generate query: {document}. Query:')
            labels.append(query)
            
        tokenized = self.tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=self.max_tokens)
        tokenized['labels'] = self.tokenizer(labels, return_tensors='pt', padding=True, truncation='longest_first')['input_ids']
        
        # Force "Query:<eos>" to be at the end of the prompt, if it gets truncated.
        for example in tokenized['input_ids']:
            example[-4:] = torch.LongTensor([3, 27569, 10, 1])
        for example in tokenized['attention_mask']:
            example[-4:] = torch.LongTensor([1, 1, 1, 1])

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)

        return tokenized
    
    def tokenize_inference(self, batch):
        texts = []
        for example in batch:
            document = example
            texts.append(f'Generate query: {document}. Query:')
            
        tokenized = self.tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=self.max_tokens)
        
        # Force "Query:<eos>" to be at the end of the prompt, if it gets truncated.
        for example in tokenized['input_ids']:
            example[-4:] = torch.LongTensor([3, 27569, 10, 1])
        for example in tokenized['attention_mask']:
            example[-4:] = torch.LongTensor([1, 1, 1, 1])

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
    


documents = [
    """UK-Israeli mother wounded in West Bank attack dies
    The mother from a British-Israeli family who was wounded in a suspected Palestinian gun attack on Friday which killed two of her daughters has died.
    Lucy Dee, 45, had been in a coma since the attack in the occupied West Bank.

    Her daughters Rina, 15, and Maia, 20, were buried on Sunday in the settlement of Kfar Etzion in front of their father and three siblings.

    The family moved to Israel nine years ago from the UK, where Lucy's husband had served as a rabbi in north London.

    Thousands of mourners attended the emotionally charged funeral of the sisters. In his eulogy Rabbi Leo Dee asked: "How will I explain to Lucy what has happened to our two precious gifts, Maia and Rina, when she wakes up from her coma?"

    Ein Kerem Hospital in Jerusalem announced that Lucy (who was also known by her Hebrew name, Leah) Dee had died on Monday morning "despite g
reat and constant efforts".

    Lucy, Rina and Maia were shot at as they were driving in the Jordan Valley in the northern West Bank on their way to a family holiday. Thei
r vehicle crashed and the gunmen went up to the car and opened fire on the women at close range, Israeli media quoted investigators as saying.
    """,

    """Lasse Wellander: Abba pay tribute to guitarist's 'musical brilliance'
    Abba have paid tribute to long-serving guitarist Lasse Wellander, saying his "musical brilliance" played "an integral role in the Abba stor
y".

    Wellander first worked with the Swedish quartet as a session musician on their self-titled 1975 album and became the main guitarist on thei
r subsequent LPs.

    He can be heard on hits such as Knowing Me, Knowing You, Thank You for the Music and The Winner Takes It All.

    "Lasse was a dear friend, a fun guy and a superb guitarist," Abba said.

    He died on Friday at the age of 70.
    """,
]


#sample 8k docs that are not in qrels

dids_in_qrels = set()
with open("/data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv", 'r') as h:
    for line in h:
        qid, q0, did, rel = line.strip().split("\t")
        dids_in_qrels.add(did)

with open("/data/user_data/jmcoelho/datasets/marco/documents/qrels.dev.tsv", 'r') as h:
    for line in h:
        qid, q0, did, rel = line.strip().split("\t")
        dids_in_qrels.add(did)

corpus = {}
good_doc_ids = []
with open("/data/user_data/jmcoelho/datasets/marco/documents/corpus_firstp_2048.tsv", 'r') as h:
    for line in h:
        did, title, text = line.split("\t")
        if did not in dids_in_qrels:
            good_doc_ids.append(did)
            corpus[did] = f"{title} {text}"
            if len(good_doc_ids) == 300000:
                break

documents = [corpus[doc_id] for doc_id in good_doc_ids]

longp = T5QueryGenerator(base_model='jmvcoelho/t5-base-msmarco-squad-query-generation-longp-v2', max_tokens=1536, device='cuda')
longp_queries = longp.inference(documents, batchsize=150)

q_id = 0
with open("/data/user_data/jmcoelho/datasets/marco/documents/gen2.query.tsv", 'w', encoding='utf-8') as out_1, open("/data/user_data/jmcoelho/datasets/marco/documents/qrels.gen2.tsv", 'w', encoding='utf-8') as out_2:
    for doc_id, query in zip(good_doc_ids, longp_queries):
        query = query.replace('\t', '').replace('\n', '').replace('\r', '')
        try:
            out_1.write(f"{q_id}\t{query}\n")
        except:
            continue
        out_2.write(f"{q_id}\tQ0\t{doc_id}\t1\n")
        q_id += 1

print("Done")






#pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-random-negs-top100