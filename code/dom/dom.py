# given the activations, this finds normalized directions for utility/safety
# it also shows a plot of their (safety/utility) cosine similarities across
# the different layers of the model

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def dom_cosine_similarities(base, group_a, group_b):
    '''
    base is what you compare from
    group_a is the direction of which we try to figure
    group_b is the other direction of which we try to figure
    it outputs an array containing cosine similarities of the
    directions per each layer
    '''
    num_layers = base.shape[1]
    cosine_sims = []
    for layer in range(num_layers):
        base_mean = base[:, layer, :].mean(axis=0)
        a_mean = group_a[:, layer, :].mean(axis=0)
        b_mean = group_b[:, layer, :].mean(axis=0)

        direction_a = a_mean - base_mean
        direction_b = b_mean - base_mean

        direction_a /= np.linalg.norm(direction_a)
        direction_b /= np.linalg.norm(direction_b)
        cosine_sims.append(np.dot(direction_a, direction_b))
    return np.array(cosine_sims)


def dom_permutation_test(base, group_a, group_b, n_iterations=1000):
    '''
    Runs a permutation test to build a null distribution of cosine similarities.
    base, group_a, group_b should each have shape (N, num_layers, hidden_dim).
    Returns (mean_null, lower_bound, upper_bound) each of shape (num_layers,).
    '''
    N = base.shape[0]
    num_layers = base.shape[1]

    all_activations = np.concatenate([group_a, base, group_b], axis=0)
    total_samples = all_activations.shape[0]

    null_cosine_similarities = np.zeros((n_iterations, num_layers))

    for i in tqdm(range(n_iterations), desc="Permutation Test"):
        shuffled_idx = np.random.permutation(total_samples)

        idx_a    = shuffled_idx[:N]
        idx_base = shuffled_idx[N:2*N]
        idx_b    = shuffled_idx[2*N:]

        for layer in range(num_layers):
            a_mean_pseudo    = all_activations[idx_a,    layer, :].mean(axis=0)
            base_mean_pseudo = all_activations[idx_base, layer, :].mean(axis=0)
            b_mean_pseudo    = all_activations[idx_b,    layer, :].mean(axis=0)

            dir_a = a_mean_pseudo - base_mean_pseudo
            dir_b = b_mean_pseudo - base_mean_pseudo

            dir_a /= np.linalg.norm(dir_a)
            dir_b /= np.linalg.norm(dir_b)

            null_cosine_similarities[i, layer] = np.dot(dir_a, dir_b)

    mean_null   = np.mean(null_cosine_similarities, axis=0)
    lower_bound = np.percentile(null_cosine_similarities, 2.5,  axis=0)
    upper_bound = np.percentile(null_cosine_similarities, 97.5, axis=0)

    return mean_null, lower_bound, upper_bound


def dom_plot(cosine_similarities, mean_null, lower_bound, upper_bound):
    '''
    Plots actual cosine similarities against the permutation null distribution.
    '''
    num_layers = len(cosine_similarities)

    plt.figure(figsize=(12, 6))

    plt.plot(range(num_layers), cosine_similarities, marker='o', linewidth=2,
             label="Actual Cosine Similarity", color='blue')

    plt.plot(range(num_layers), mean_null, color='red', linestyle='--',
             label="Null Distribution Mean")
    plt.fill_between(range(num_layers), lower_bound, upper_bound,
                     color='red', alpha=0.2, label="95% Confidence Interval (Null)")

    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("Layer number (0 is embedding layer)")
    plt.ylabel("Cosine similarity")
    plt.title("Actual Cosine Similarity vs. Permutation Null Distribution")
    plt.xticks(range(0, num_layers, 1))
    plt.yticks(np.arange(-1, 1.1, 0.1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
