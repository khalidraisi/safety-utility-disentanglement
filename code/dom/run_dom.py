import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dom import dom_cosine_similarities, dom_permutation_test

MODELS = {
    "gemma3_4b":  "../activations/gemma3acts/4b/harmless_4b_n500_activations.npy,"
                  "../activations/gemma3acts/4b/harmful_4b_n500_activations.npy,"
                  "../activations/gemma3acts/4b/utility_4b_n500_activations.npy",
    "gemma4_E2B": "../activations/gemma4acts/E2B/harmless_E2B_activations.npy,"
                  "../activations/gemma4acts/E2B/harmful_E2B_activations.npy,"
                  "../activations/gemma4acts/E2B/utility_E2B_activations.npy",
    "llama3_3b":  "../activations/llama3acts/harmless_3b_activations.npy,"
                  "../activations/llama3acts/harmful_3b_activations.npy,"
                  "../activations/llama3acts/utility_3b_activations.npy",
    "qwen35_4b":  "../activations/qwen35acts/harmless_4b_activations.npy,"
                  "../activations/qwen35acts/harmful_4b_activations.npy,"
                  "../activations/qwen35acts/utility_4b_activations.npy",
}

def run(name, paths, n_iter):
    base_p, safe_p, util_p = paths.split(",")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {name} ===")
    base = np.load(base_p)
    safe = np.load(safe_p)
    util = np.load(util_p)
    print(f"  shapes: base={base.shape} safety={safe.shape} utility={util.shape}")

    cs = dom_cosine_similarities(base, safe, util)
    mean_null, lo, hi = dom_permutation_test(base, safe, util, n_iterations=n_iter)

    np.save(os.path.join(out_dir, "cosine_similarities.npy"), cs)
    np.save(os.path.join(out_dir, "null_mean.npy"), mean_null)
    np.save(os.path.join(out_dir, "null_lower.npy"), lo)
    np.save(os.path.join(out_dir, "null_upper.npy"), hi)

    num_layers = len(cs)
    plt.figure(figsize=(12, 6))
    plt.plot(range(num_layers), cs, marker="o", linewidth=2,
             label="Actual Cosine Similarity", color="blue")
    plt.plot(range(num_layers), mean_null, color="red", linestyle="--",
             label="Null Distribution Mean")
    plt.fill_between(range(num_layers), lo, hi, color="red", alpha=0.2,
                     label="95% CI (Null)")
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Layer number (0 is embedding layer)")
    plt.ylabel("Cosine similarity")
    plt.title(f"Safety vs. Utility direction cosine similarity — {name}")
    plt.xticks(range(0, num_layers, 1))
    plt.yticks(np.arange(-1, 1.1, 0.1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cosine_similarity.png"), dpi=150)
    plt.close()
    print(f"  saved -> {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=1000)
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    args = ap.parse_args()
    for m in args.models:
        run(m, MODELS[m], args.n_iter)
