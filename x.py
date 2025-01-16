import json
from tqdm import tqdm

# Input and output file paths
input_file = (
    "/data/group_data/cx_group/query_generation_data/query_generation_llama.jsonl"
)
output_file = (
    "/data/group_data/cx_group/query_generation_data/cweb_subset/original_pos_neg.jsonl"
)
# Initialize mappings
id_to_seqint = {}
seqint_to_text = {}
current_seqint = 0

# Process the JSONL file
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for idx, line in tqdm(enumerate(infile)):
        # Parse the line
        data = json.loads(line)
        doc_a_id = data["positive_document_id"]
        doc_a_text = data["positive_document"]
        doc_b_id = data["negative_document_id"]
        doc_b_text = data["negative_document"]

        # Map doc_a
        if doc_a_id not in id_to_seqint:
            id_to_seqint[doc_a_id] = current_seqint
            seqint_to_text[current_seqint] = (doc_a_id, doc_a_text)
            current_seqint += 1

        # Map doc_b
        if doc_b_id not in id_to_seqint:
            id_to_seqint[doc_b_id] = current_seqint
            seqint_to_text[current_seqint] = (doc_b_id, doc_b_text)
            current_seqint += 1

    # Write the new JSONL
    for seqint, z in seqint_to_text.items():
        new_entry = {"docid": seqint, "old_docid": z[0], "text": z[1], "title": ""}
        outfile.write(json.dumps(new_entry) + "\n")
