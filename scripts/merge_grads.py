import pickle
import os
import numpy as np
from tqdm import tqdm

# Path to the directory containing the gradient files
grad_dir = "/data/user_data/jmcoelho/embeddings/marco_docs/pythia-160m-marco-docs-bow-ct-pretrain-bs256-small-supervision-1gpu/valid_grads_bs64_with_mom/"

# List to store all gradient vectors
all_gradients = []

# Loop over all files in the directory
for filename in tqdm(os.listdir(grad_dir)):
    if filename.startswith("grad_step_"):
        filepath = os.path.join(grad_dir, filename)
        
        # Load each gradient file
        with open(filepath, 'rb') as f:
            grad_vector = pickle.load(f).numpy()
            print(np.linalg.norm(grad_vector))
            all_gradients.append(grad_vector)

print("#########")
# Compute the average gradient vector
average_gradient = np.mean(all_gradients, axis=0)
print(np.linalg.norm(average_gradient))

normalized_avg_grad = average_gradient / np.linalg.norm(average_gradient)

print(np.linalg.norm(normalized_avg_grad))
# Save the average gradient vector
output_file = os.path.join(grad_dir, "normalized_average_gradients.pkl")
with open(output_file, 'wb') as h:
    pickle.dump(normalized_avg_grad, h, protocol=pickle.HIGHEST_PROTOCOL)

print(normalized_avg_grad.shape)

print(f"Normalized average gradient vector saved to {output_file}")