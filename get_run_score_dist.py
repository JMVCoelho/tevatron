import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    x = np.array(x)
    exp_x = np.exp(x - np.max(x)) 
    return exp_x / exp_x.sum(axis=0)


def normalize_to_prob_distribution(array):
    # Ensure the input is a NumPy array
    array = np.array(array[:50])
    # Calculate the sum of all elements in the array
    array_sum = np.sum(array)
    # Normalize the array by dividing each element by the sum
    prob_distribution = array / array_sum
    return prob_distribution

qid2score={}

with open("/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs64-10pc-sample-less-negs/run.train.10pc.sample.txt", 'r') as h:

    for line in h:
        qid, did, score = line.strip().split()

        if qid not in qid2score:
            qid2score[qid] = []

        qid2score[qid].append(float(score)/0.01)


qid2softmax = {k:softmax(v) for k,v in qid2score.items()}

prob = normalize_to_prob_distribution(np.sum(list(qid2softmax.values()), axis=0))

probabilities = {idx:p for idx,p in enumerate(prob)}

plt.bar(probabilities.keys(), probabilities.values())
plt.xlabel('Index')
plt.ylabel('Probabilities')
plt.title('2nd Episode: Negative index probability (softmax, T=0.01)')
plt.savefig('temp.png')
