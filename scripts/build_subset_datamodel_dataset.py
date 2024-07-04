import numpy as np

# Step 1: Read the dataset
data = []
with open("/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T1/group_level_10000_two_valid_orcale/datamodel_test_no_std.tsv", 'r') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            x1 = parts[0]
            x2 = parts[1]
            label = float(parts[2])
            data.append((x1, x2, label))

# Step 2: Extract labels
labels = np.array([item[2] for item in data])

# Step 3: Calculate percentiles
lowest_threshold = np.percentile(labels, 10)
highest_threshold = np.percentile(labels, 90)

# Step 4: Create binary labels (lowest 1, highest 0)
binary_labels = []
for label in labels:
    if label <= lowest_threshold:
        binary_labels.append(1)  # Lowest 10%
    elif label >= highest_threshold:
        binary_labels.append(0)  # Highest 10%
    else:
        binary_labels.append(None)  # Not used in subset

# Step 5: Create subset based on binary labels
subset_data = []
for i in range(len(data)):
    if binary_labels[i] is not None:
        x1, x2, label = data[i]
        subset_data.append((x1, x2, binary_labels[i]))

# Print or further process subset_data
print("Subset size:", len(subset_data))

with open("/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-all-queries-10k2-valid-5-group-level-T1/group_level_10000_two_valid_orcale/datamodel_test_binary_easy_hard_separation.tsv", 'w') as file:
    for q, docs, label in subset_data:
        file.write(f"{q}\t{docs}\t{label}\n")
