import pickle
import torch
import io

N_DENOISE = 3
TOTAL_NEGS = 9

PATH_DOTPRODS = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-20pc-sample-less-negs-triplet-topk/20pc-sample-run-splits/less-opacus-triplet"
PATH_FULL_RUN = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-20pc-sample-less-negs-triplet-topk/run.train.20pc.sample.txt"
out=f"/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs128-20pc-sample-less-negs-triplet-topk/20pc-sample-run-splits/less-opacus-triplet/single_training_examples/test" 


class CPU_Unpickler(pickle.Unpickler):
    # if pickled tensor that was in CUDA but want to load in CPU
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def load_qrels(path):
    qid2pos = {}
    with open(path, 'r') as h:
        for line in h:
            qid, q0, did, rel = line.strip().split("\t")
            
            if qid not in qid2pos:
                qid2pos[qid] = []
            
            qid2pos[qid].append(did)

    return qid2pos

qid2pos = load_qrels("/data/user_data/jmcoelho/datasets/marco/documents/qrels.train.tsv")

def load_run(path):
    qid2negs = {}
    with open(path, 'r') as h:

        for line in h:
            qid, did, score = line.strip().split()

            if qid not in qid2negs:
                qid2negs[qid] = []

            if did not in qid2pos[qid]:
                qid2negs[qid].append(did) 

    return qid2negs

qid2negs = load_run(PATH_FULL_RUN)

dot_prods = {}
for i in range(4):
    with open(f"{PATH_DOTPRODS}/hardnegs_less_opacus.20.pc.{i}.txt_dotprods.pkl", 'rb') as h:
        dot_prods.update(CPU_Unpickler(h).load())

# def remove_highest_three(lst):
#     sorted_lst = sorted(lst, reverse=False)
#     return sorted_lst[N_DENOISE:]

# with open(out, 'w') as h:
#     for qid in qid2negs:
#         _, idxs = torch.topk(dot_prods[qid], TOTAL_NEGS+N_DENOISE)

#         denoised = remove_highest_three(idxs.tolist())
        
#         neg_ids = [qid2negs[qid][idx] for idx in denoised]

#         h.write(f"{qid}\t{','.join(neg_ids)}\n")


for qid in qid2negs:
    _, idxs = torch.topk(dot_prods[qid], TOTAL_NEGS+N_DENOISE)

    if 0 in idxs:
        print(_, idxs)
        for idx in idxs[:9]:
            with open(f"{out}_{idx}", "w") as h:
                h.write(f"{qid}\t{qid2negs[qid][idx]}\n")

        exit()
                
        
