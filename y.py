import json
import pickle
from tqdm import tqdm


# Define file paths
jsonl_input_path = "/data/group_data/cx_group/query_generation_data/cweb_subset/clueweb_subset_4500000_extended.jsonl"
pickle_input_path = (
    "/data/group_data/cx_group/query_generation_data/cweb_subset/id_mapper_extended.pkl"
)
jsonl_output_path = "/data/group_data/cx_group/query_generation_data/cweb_subset/clueweb_subset_4500000_extended_with_old_docid.jsonl"

# Load the pickle file
with open(pickle_input_path, "rb") as pickle_file:
    id_mapper = pickle.load(pickle_file)
    rev = {int(v): k for k, v in id_mapper.items()}

# Process the JSONL file and add the "old_docid"
with open(jsonl_input_path, "r") as jsonl_file, open(
    jsonl_output_path, "w"
) as jsonl_output_file:
    for line in tqdm(jsonl_file):
        record = json.loads(line)
        docid = record.get("docid")
        record["old_docid"] = rev[docid]
        jsonl_output_file.write(json.dumps(record) + "\n")

print(f"Output written to {jsonl_output_path}")
