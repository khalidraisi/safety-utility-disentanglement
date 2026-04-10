# NOTE that for now at least this is only on safety layers, we have not figured out the utility layers yet

import numpy as np
import matplotlib.pyplot as plt

#We follow the safety layers in aligned models convention of 500 repetitions
R = 500

def cosine_similarity(A, B):
    dot = np.dot(A, B)
    norm = np.linalg.norm(A) * np.linalg.norm(B)
    return dot / norm

def layer_cosine_sim(act1, act2):
    return [cosine_similarity(act1[k], act2[k]) for k in range(len(act1))]

def analysis(acts1, acts2, r=R):
    results = []
    for _ in range(r):
        i = np.random.randint(len(acts1))
        j = np.random.randint(len(acts2))
        if acts1 is acts2:
            while j == i:
                j = np.random.randint(len(acts1))
        results.append(layer_cosine_sim(acts1[i], acts2[j]))
    results = np.array(results)
    return results.mean(axis=0), results.std(axis=0)

def get_similarity_plot(normal_acts, malicious_acts, r=R):
    nn_mean, nn_std = analysis(normal_acts, normal_acts, r)
    mm_mean, mm_std = analysis(malicious_acts, malicious_acts, r)
    nm_mean, nm_std = analysis(normal_acts, malicious_acts, r)

    layers = np.arange(len(nn_mean))
    layer_labels = [str(i) for i in layers]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    configs = [
        ("Normal vs Normal", nn_mean, nn_std, "tab:green"),
        ("Malicious vs Malicious", mm_mean, mm_std, "tab:red"),
        ("Normal vs Malicious", nm_mean, nm_std, "tab:brown"),
    ]

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

    plt.suptitle("How good is the model at disinguishing different types of prompts?", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()