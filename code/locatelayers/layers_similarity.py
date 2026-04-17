# locatelayers/layers_similarity.py
# this is an implementation of the locating safety layers paper
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

R = 500

def cosine_similarity(A, B, eps=1e-12):
    A = np.asarray(A)
    B = np.asarray(B)
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    if denom < eps:
        return 0.0
    return float(np.dot(A, B) / denom)


def layer_cosine_sim(act1, act2, eps=1e-12):
    act1 = np.asarray(act1)
    act2 = np.asarray(act2)
    dots = np.sum(act1 * act2, axis=1)
    norms = np.linalg.norm(act1, axis=1) * np.linalg.norm(act2, axis=1)
    return dots / np.maximum(norms, eps)

# this is for section 3.2 of the paper

def analysis(acts1, acts2, r=R):
    """
    acts1, acts2: shape = (N, num_layers, hidden_dim)
    return: mean, std for each layer
    """
    same_object = acts1 is acts2
    acts1 = np.asarray(acts1)
    acts2 = np.asarray(acts2)

    results = []

    for _ in range(r):
        i = np.random.randint(len(acts1))
        j = np.random.randint(len(acts2))
        if same_object:
            while j == i:
                j = np.random.randint(len(acts2))
        results.append(layer_cosine_sim(acts1[i], acts2[j]))

    results = np.asarray(results)
    return results.mean(axis=0), results.std(axis=0)

def get_similarity_plot(normal_acts, malicious_acts, r=R, combined=False, save_path=None):
    nn_mean, nn_std = analysis(normal_acts, normal_acts, r)
    mm_mean, mm_std = analysis(malicious_acts, malicious_acts, r)
    nm_mean, nm_std = analysis(normal_acts, malicious_acts, r)

    layers = np.arange(len(nn_mean))
    layer_labels = [str(i) for i in layers]

    configs = [
        ("Normal vs Normal", nn_mean, nn_std, "tab:green"),
        ("Malicious vs Malicious", mm_mean, mm_std, "tab:red"),
        ("Normal vs Malicious", nm_mean, nm_std, "tab:brown"),
    ]

    if combined:
        fig, ax = plt.subplots(figsize=(12, 5))
        for title, mean, std, color in configs:
            ax.plot(layers, mean, color=color, linewidth=2, label=title)
        ax.set_xlabel("Layer (Layer 0 is the embedding layer)")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Layer-wise Cosine Similarity Analysis")
        ax.set_xticks(layers)
        ax.set_xticklabels(layer_labels, rotation=90, fontsize=7)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for ax, (title, mean, std, color) in zip(axes, configs):
            ax.plot(layers, mean, color=color, linewidth=2, label=title)
            ax.fill_between(layers, mean - std, mean + std, color=color, alpha=0.2)
            ax.set_xlabel("Layer (Layer 0 is the embedding layer)")
            ax.set_ylabel("Cosine Similarity")
            ax.set_title(title)
            ax.set_xticks(layers)
            ax.set_xticklabels(layer_labels, rotation=90, fontsize=7)
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.suptitle("Layer-wise Cosine Similarity Analysis", fontsize=14, y=1.02)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return nn_mean, nn_std, mm_mean, mm_std, nm_mean, nm_std


# TODO this is for section 3.3 of the paper

# this is for section 3.4 (Not implemented yet, figure this out later), 
# it seems a little more complicated and some things would probably need to be 
# written within the experiments

