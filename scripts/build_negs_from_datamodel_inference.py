import random

random.seed(17121998)


MODEL = "pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1"

out = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T0.1/datamodel_negs/full.queries.train+val.datamodel.top100.sampleT01.txt"


mapper = {}
with open(f"/data/user_data/jmcoelho/embeddings/marco_docs/{MODEL}/group_level_10000_two_valid_orcale_momentum_600_query_2k/datamodel_test_independency_w_preds_full.tsv") as h:
    for line in h:
        qid, neg, score = line.strip().split()

        if qid not in mapper:
            mapper[qid] = []

        mapper[qid].append((neg, (1/float(score)) / 0.1))


# for key in mapper:
#     mapper[key].sort(key=lambda x: x[1])
#     mapper[key] = [x[0] for x in mapper[key][:9]]


# with open(out, 'w') as f:
#     for key in mapper:
#         f.write(f"{key}\t{','.join(mapper[key])}\n")


def normalize_scores(tuples_list):
    total_score = sum(score for _, score in tuples_list)
    normalized_list = [(subid, score / total_score) for subid, score in tuples_list]
    return normalized_list

def sample_elements(mapper, num_samples=9):
    sampled_mapper = {}
    for id, tuples_list in mapper.items():
        normalized_list = normalize_scores(tuples_list)
        subids, normalized_scores = zip(*normalized_list)
        sampled_subids = random.choices(subids, weights=normalized_scores, k=num_samples)
        sampled_mapper[id] = sampled_subids
    return sampled_mapper

mapper = sample_elements(mapper)

with open(out, 'w') as f:
    for key in mapper:
        f.write(f"{key}\t{','.join(mapper[key])}\n")
