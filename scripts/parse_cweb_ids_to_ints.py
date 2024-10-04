

did = 0

cweb2int = {}

with open("/data/group_data/cx_group/query_generation_data/GPT4/bm25-negatives/corpus.tsv") as h, \
    open("/data/group_data/cx_group/query_generation_data/GPT4/bm25-negatives/corpus_int.tsv", 'w') as out:
    for line in h:
        cwebid, text = line.strip().split("\t")
        cweb2int[cwebid] = did
        out.write(f"{did}\t{text}\n")
        did += 1

with open("/data/group_data/cx_group/query_generation_data/GPT4/bm25-negatives/negs.tsv") as h, \
    open("/data/group_data/cx_group/query_generation_data/GPT4/bm25-negatives/negs_int.tsv", 'w') as out:

    for line in h:
        qid, negs = line.strip().split("\t")
        negs = negs.split(",")

        out.write(f"{qid}\t{','.join([str(cweb2int[neg]) for neg in negs])}\n")


with open("/data/group_data/cx_group/query_generation_data/GPT4/bm25-negatives/qrels.gen7.tsv") as h, \
    open("/data/group_data/cx_group/query_generation_data/GPT4/bm25-negatives/qrels.gen7.int.tsv", 'w') as out:

    for line in h:
        qid, q0, cwebid, rel = line.strip().split("\t")

        out.write(f"{qid}\t{q0}\t{cweb2int[cwebid]}\t{rel}\n")