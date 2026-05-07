# plot_csvd.py
#
# Same plot shape as the original ActSVD overlap plot, but uses the
# CONTRASTIVE safety subspace instead of the ActSVD-Top one.
# Answers: "if we extract safety via paired (harmful - harmless) diffs
# and SVD that, is it orthogonal to utility?"
#
# Reads:
#   svd/<tag>/contrast_subspaces_<tag>.npy
#   svd/<tag>/utility_subspaces_<tag>.npy
#
# Writes:
#   svd/<tag>/csvd_overlap_<tag>.png
#
# Example:
#   python3 plot_csvd.py --svd-dir svd/gemma-3-4b-it --model-tag gemma-3-4b-it

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def principal_angle_cosines(U1, U2):
    _, s, _ = np.linalg.svd(U1.T @ U2, full_matrices=False)
    return np.clip(s, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svd-dir", required=True)
    ap.add_argument("--model-tag", required=True)
    args = ap.parse_args()

    d = Path(args.svd_dir)
    tag = args.model_tag

    contrast = np.load(d / f"contrast_subspaces_{tag}.npy")  # (L, d, r)
    utility  = np.load(d / f"utility_subspaces_{tag}.npy")

    overlap, top_cos = [], []
    for U_c, U_u in zip(contrast, utility):
        cos = principal_angle_cosines(U_c, U_u)
        overlap.append(float(np.mean(cos ** 2)))
        top_cos.append(float(cos[0]))
    overlap = np.array(overlap)
    top_cos = np.array(top_cos)

    layers = np.arange(len(overlap))
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].plot(layers, overlap, marker="o", linewidth=2, color="purple")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Subspace overlap score")
    axes[0].set_title(f"Contrastive Safety vs Utility — {tag}")
    axes[0].set_xticks(layers)
    axes[0].axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(layers, top_cos, marker="o", linewidth=2, color="darkorange")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Top principal-angle cosine")
    axes[1].set_title(f"Top contrastive-safety / utility alignment — {tag}")
    axes[1].set_xticks(layers)
    axes[1].axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = d / f"csvd_overlap_{tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out}")
    print(f"mean overlap = {overlap.mean():.4f}")
    print(f"mean top cos = {top_cos.mean():.4f}")


if __name__ == "__main__":
    main()
