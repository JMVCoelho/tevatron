import csv
import json


def tsv_to_jsonl(tsv_path, jsonl_path):
    with open(tsv_path, "r", encoding="utf-8") as tsv_file, open(
        jsonl_path, "w", encoding="utf-8"
    ) as jsonl_file:
        reader = csv.reader(tsv_file, delimiter="\t")

        for query_id, row in enumerate(reader):
            query = row[0]  # first column is the query text
            json_obj = {"query_id": str(query_id), "query": query}
            jsonl_file.write(json.dumps(json_obj) + "\n")


# Example usage

in_file = "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-random-20k-1gpu/mates_neg_cache_loss_100_2/mates_valid_loss.tsv"
out_file = "/data/user_data/jmcoelho/embeddings/marco_docs/Qwen2.5-0.5B-bidirectional-attn-wavg-pool-mntp-minicpmembed-random-20k-1gpu/mates_neg_cache_loss_100_2/all_queries.tsv"
tsv_to_jsonl(in_file, out_file)
