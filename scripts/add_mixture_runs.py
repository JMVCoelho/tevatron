PATH_LESS="/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs64-10pc-sample-less-negs/less_train_run_splits/less_grad_bs64_temperature_top100"
PATH_RANDOM="/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs64-10pc-sample-less-negs/random_train_run_splits/random/"
import os

def head(infile, lines):
    with open(infile, 'r') as fin:
        head_lines = fin.readlines()[:lines]
        return head_lines


def tail(infile, lines):
    with open(infile, 'r') as fin:
        last_lines = fin.readlines()[-lines:]
        return last_lines


TOTAL_LINES = 36700

mixture = [70, 30]

# run_1_path = f"{PATH_RANDOM}/10pc.train.random.momentum.txt"
# run_2_path = f"{PATH_LESS}/hardnegs_less_opacus_10.pc.full.sample.t1.momentum"
# out_path = f"{PATH_LESS}/mixtures_{mixture[0]}{mixture[1]}"
# out_file = "/mixture.t1.10pc.momentum.txt"


run_1_path = f"{PATH_RANDOM}/10pc.train.random.momentum.1headrep.txt"
run_2_path = f"{PATH_LESS}/hardnegs_less_opacus_10.pc.full.sample.t1.momentum-1headrep"
out_path = f"{PATH_LESS}/mixtures_{mixture[0]}{mixture[1]}"
out_file = "/mixture.t1.10pc.momentum.1headrep.txt"

if not os.path.exists(out_path):
    os.makedirs(out_path)

all_lines = head(run_1_path, int(mixture[0]/100 * TOTAL_LINES)) + tail(run_2_path, int(mixture[1]/100 * TOTAL_LINES))

assert len(all_lines) == TOTAL_LINES

with open(out_path+out_file, 'w') as fout:
     fout.writelines(all_lines)