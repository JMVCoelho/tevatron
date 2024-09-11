import pickle
import os
import numpy as np
from tqdm import tqdm

# Path to the directory containing the gradient files
grad_dir = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision/valid_grads_bs64/"

# List to store all gradient vectors
all_gradients = []

# Loop over all files in the directory
for filename in tqdm(os.listdir(grad_dir)):
    if filename.endswith(".pkl"):
        filepath = os.path.join(grad_dir, filename)
        
        # Load each gradient file
        with open(filepath, 'rb') as f:
            grad_vector = pickle.load(f)
            all_gradients.append(grad_vector)

# Compute the average gradient vector
average_gradient = np.mean(all_gradients, axis=0)

# Save the average gradient vector
output_file = os.path.join(grad_dir, "average_gradient.pkl")
with open(output_file, 'wb') as h:
    pickle.dump(average_gradient, h, protocol=pickle.HIGHEST_PROTOCOL)

print(average_gradient.shape)

print(f"Average gradient vector saved to {output_file}")