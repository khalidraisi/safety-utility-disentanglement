#   python3 run_paired_csvd.py \
#       --activations-dir activations/gemma3acts \
#       --suffix 4b \
#       --model-tag gemma-3-4b-it \
#       --r 10
 
import argparse
import json
from pathlib import Path
 
import numpy as np
import matplotlib.pyplot as plt
 
 
def compute_topr_sv(activations, r):
    X = activations.T
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :r]
 

def per_layer_contrast(target, contrast, r):
    assert target.shape == contrast.shape, \
        f"shape mismatch: target={target.shape}, contrast={contrast.shape}"
    n_layers = target.shape[1]
    return [compute_topr_sv(target[:, i, :] - contrast[:, i, :], r) for i in range(n_layers)]
 
 
def principal_angle_cosines(U1, U2):
    _, s, _ = np.linalg.svd(U1.T @ U2, full_matrices=False)
    return np.clip(s, 0.0, 1.0)
 
 
def per_layer_overlap(subs_A, subs_B):
    overlap, top_cos = [], []
    for U_a, U_b in zip(subs_A, subs_B):
        cos = principal_angle_cosines(U_a, U_b)
        overlap.append(float(np.mean(cos ** 2)))
        top_cos.append(float(cos[0]))
    return np.array(overlap), np.array(top_cos)
 
 
def plot_overlap(overlap, top_cos, out_path, title_top, title_bot, color):
    layers = np.arange(len(overlap))
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
 
    axes[0].plot(layers, overlap, marker="o", linewidth=2, color=color)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Subspace overlap score")
    axes[0].set_title(title_top)
    axes[0].set_xticks(layers)
    axes[0].axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
    axes[0].grid(True, alpha=0.3)
 
    axes[1].plot(layers, top_cos, marker="o", linewidth=2, color="darkorange")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Top principal-angle cosine")
    axes[1].set_title(title_bot)
    axes[1].set_xticks(layers)
    axes[1].axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
    axes[1].grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
def truncate_pair(A, B):
    n = min(A.shape[0], B.shape[0])
    if n != A.shape[0] or n != B.shape[0]:
        print(f"  truncating to n={n} for index-alignment "
              f"(A had {A.shape[0]}, B had {B.shape[0]})")
    return A[:n], B[:n]
 
 
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations-dir", required=True)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--r", type=int, default=10)
    ap.add_argument("--out-root", default="svd")
    return ap.parse_args()
 
 
def main():
    args = parse_args()
    act_dir = Path(args.activations_dir)
    out_dir = Path(args.out_root) / args.model_tag
    tag = args.model_tag
 
    #load activations
    harmful         = np.load(act_dir / f"harmful_{args.suffix}_activations.npy")
    utility         = np.load(act_dir / f"utility_{args.suffix}_activations.npy")
    harmless_paired = np.load(act_dir / f"harmless_paired_{args.suffix}_activations.npy")
    nonutility      = np.load(act_dir / f"nonutility_{args.suffix}_activations.npy")
 
    #load existing subspaces
    safety_subs  = np.load(out_dir / f"safety_subspaces_{tag}.npy")
    utility_subs = np.load(out_dir / f"utility_subspaces_{tag}.npy")
 
    # contrastive safety properly paired via LLM
    print("Safety contrast (paired):")
    h_pair, s_pair = truncate_pair(harmful, harmless_paired)
    r_safety = min(args.r, h_pair.shape[2], h_pair.shape[0])
    contrast_safety = per_layer_contrast(h_pair, s_pair, r_safety)
    np.save(out_dir / f"contrast_safety_subspaces_{tag}.npy", np.stack(contrast_safety))
 
    #helpfulness=4 vs =0
    print("Utility contrast (index-aligned):")
    u_pair, n_pair = truncate_pair(utility, nonutility)
    r_utility = min(args.r, u_pair.shape[2], u_pair.shape[0])
    contrast_utility = per_layer_contrast(u_pair, n_pair, r_utility)
    np.save(out_dir / f"contrast_utility_subspaces_{tag}.npy", np.stack(contrast_utility))
 
    # overlap with the mean-centered subspaces from run_actsvd.py
    # Truncate the mean-centered bases to the same r so overlaps are between
    # subspaces of equal dimension.
    utility_subs_r = [U[:, :r_safety] for U in utility_subs]
    safety_subs_r  = [U[:, :r_utility] for U in safety_subs]
 
    safety_overlap, safety_topc = per_layer_overlap(contrast_safety, utility_subs_r)
    utility_overlap, utility_topc = per_layer_overlap(contrast_utility, safety_subs_r)
 
    plot_overlap(
        safety_overlap, safety_topc,
        out_dir / f"csvd_safety_overlap_{tag}.png",
        f"Paired contrastive safety vs utility — {tag} (r={r_safety})",
        f"Top paired-safety / utility alignment — {tag} (r={r_safety})",
        color="purple",
    )
    plot_overlap(
        utility_overlap, utility_topc,
        out_dir / f"csvd_utility_overlap_{tag}.png",
        f"Contrastive utility vs safety — {tag} (r={r_utility})",
        f"Top utility-contrast / safety alignment — {tag} (r={r_utility})",
        color="seagreen",
    )
 
    metrics = {
        "model_tag": tag,
        "r_safety": r_safety,
        "r_utility": r_utility,
        "safety_contrast_vs_utility": {
            "overlap_scores": safety_overlap.tolist(),
            "top_cos":        safety_topc.tolist(),
            "mean_overlap":   float(safety_overlap.mean()),
            "mean_top_cos":   float(safety_topc.mean()),
        },
        "utility_contrast_vs_safety": {
            "overlap_scores": utility_overlap.tolist(),
            "top_cos":        utility_topc.tolist(),
            "mean_overlap":   float(utility_overlap.mean()),
            "mean_top_cos":   float(utility_topc.mean()),
        },
    }
    (out_dir / f"csvd_paired_metrics_{tag}.json").write_text(
        json.dumps(metrics, indent=2)
    )
 
    print(f"\nWrote outputs to {out_dir}")
    print(f"  paired safety contrast vs utility:    mean overlap = {safety_overlap.mean():.4f}")
    print(f"  utility contrast vs safety (mean-c.): mean overlap = {utility_overlap.mean():.4f}")
 
 
if __name__ == "__main__":
    main()