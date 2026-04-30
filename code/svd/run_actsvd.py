# Run ActSVD analysis for a given model's activations.
#
# Self-contained: no actsvd.py import needed. Loads pre-extracted activations
# (harmful, harmless, utility) from a directory, computes safety/utility
# subspaces using ActSVD (Top) and ActSVD (Orthogonal Projection), measures
# overlap, and saves all results + plots into svd/<model_tag>/.
#
# Example:
#   python3 run_actsvd.py \
#       --activations-dir activations/gemma3acts \
#       --suffix 4b \
#       --model-tag gemma-3-4b-it \
#       --r 10
#
# Output goes to: svd/gemma-3-4b-it/

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# ActSVD core
# ---------------------------------------------------------------------------

def compute_topr_sv(activations, r):
    '''
    Top-r left singular vectors of the activations matrix.
    Input: (n_prompts, hidden_dim). Output: (hidden_dim, r) orthonormal cols.
    '''
    X = activations.T  # (hidden_dim, n_prompts)
    U, S, V = np.linalg.svd(X, full_matrices=False)
    return U[:, :r]


def actsvd_top(activations, r):
    '''
    ActSVD (Top): per-layer top-r singular subspaces.
    Input: (n_prompts, n_layers, hidden_dim). Output: list of (hidden_dim, r).
    '''
    num_layers = activations.shape[1]
    return [compute_topr_sv(activations[:, i, :], r) for i in range(num_layers)]


def actsvd_op(subspace_A, subspace_B):
    '''
    ActSVD (Orthogonal Projection): remove the part of subspace_A that
    overlaps with subspace_B. Returns the isolated basis plus per-column
    survival norms.

    survival_norm[j] = || (I - Pi_B) u_j ||  where u_j is a column of U_A.
      ~1.0 => safety direction was nearly orthogonal to utility (kept intact)
      ~0.0 => safety direction lived inside utility subspace (killed by proj)
    '''
    num_layers = len(subspace_A)
    isolated = []
    survival_norms = []

    for i in range(num_layers):
        U_A = subspace_A[i]
        U_B = subspace_B[i]

        # projection matrix onto subspace_B, applied to U_A
        Pi_B_U_A = U_B @ (U_B.T @ U_A)
        U_A_isolated = U_A - Pi_B_U_A

        col_norms = np.linalg.norm(U_A_isolated, axis=0)
        survival_norms.append(col_norms)

        # orthonormalize what's left
        Q, R = np.linalg.qr(U_A_isolated)
        isolated.append(Q[:, :U_A.shape[1]])

    return isolated, survival_norms


def contrastive_svd(tgt_acts, baseline_acts, r):
    '''
    Top-r directions distinguishing two paired prompt sets.
    Inputs must have matching shape (n, n_layers, hidden_dim).
    '''
    assert tgt_acts.shape == baseline_acts.shape, \
        "target and baseline have different shapes"

    num_layers = tgt_acts.shape[1]
    subspaces = []
    for i in range(num_layers):
        differences = tgt_acts[:, i, :] - baseline_acts[:, i, :]
        subspaces.append(compute_topr_sv(differences, r))
    return subspaces


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_activations(activations_dir: Path, suffix: str):
    '''
    Load the three activation arrays produced by extract_activations.py.
    Returns (harmful, harmless, utility), each (n, n_layers, hidden_dim).
    '''
    harmful  = np.load(activations_dir / f"harmful_{suffix}_activations.npy")
    harmless = np.load(activations_dir / f"harmless_{suffix}_activations.npy")
    utility  = np.load(activations_dir / f"utility_{suffix}_activations.npy")

    if not (harmful.shape[1:] == harmless.shape[1:] == utility.shape[1:]):
        raise ValueError(
            f"Layer/hidden dims disagree: "
            f"harmful={harmful.shape}, harmless={harmless.shape}, "
            f"utility={utility.shape}"
        )

    print(f"harmful : {harmful.shape}")
    print(f"harmless: {harmless.shape}")
    print(f"utility : {utility.shape}")
    return harmful, harmless, utility


# ---------------------------------------------------------------------------
# Subspace overlap metrics
# ---------------------------------------------------------------------------

def principal_angle_cosines(U1, U2):
    '''
    Cosines of principal angles between two orthonormal bases.
    Returns array of length min(r1, r2), each in [0, 1].
    '''
    _, s, _ = np.linalg.svd(U1.T @ U2, full_matrices=False)
    return np.clip(s, 0.0, 1.0)


def subspace_overlap_score(U1, U2):
    '''Mean squared cosine of principal angles. 0 = orthogonal, 1 = identical.'''
    s = principal_angle_cosines(U1, U2)
    return float(np.mean(s ** 2))


def per_layer_overlap(safety_subs, utility_subs):
    '''Per-layer (overlap_score, top_principal_cosine).'''
    overlap = []
    top_cos = []
    for U_s, U_u in zip(safety_subs, utility_subs):
        cos = principal_angle_cosines(U_s, U_u)
        overlap.append(float(np.mean(cos ** 2)))
        top_cos.append(float(cos[0]))
    return np.array(overlap), np.array(top_cos)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overlap(overlap_scores, top_cos, out_path: Path, r: int, model_tag: str):
    '''
    Two-panel plot: subspace overlap score and top principal-angle cosine
    per layer.
    '''
    layers = np.arange(len(overlap_scores))
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].plot(layers, overlap_scores, marker="o", linewidth=2, color="steelblue")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Subspace overlap score")
    axes[0].set_title(f"Safety vs Utility subspace overlap — {model_tag} (r={r})")
    axes[0].set_xticks(layers)
    axes[0].axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(layers, top_cos, marker="o", linewidth=2, color="darkorange")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Top principal-angle cosine")
    axes[1].set_title(f"Top safety/utility alignment — {model_tag} (r={r})")
    axes[1].set_xticks(layers)
    axes[1].axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_survival(survival_norms, out_path: Path, model_tag: str):
    '''
    Per-column survival norms, log y-axis. Each dot is one safety direction
    at one layer. Norm near 1.0 = direction survived projection (orthogonal
    to utility). Norm near 0 = killed by projection (entangled with utility).
    '''
    num_layers = len(survival_norms)
    fig = plt.figure(figsize=(12, 6))
    rng = np.random.default_rng(0)

    for layer_idx, norms in enumerate(survival_norms):
        xs = layer_idx + rng.uniform(-0.15, 0.15, size=len(norms))
        ys = np.clip(norms, 1e-18, None)
        plt.scatter(
            xs, ys, s=25, alpha=0.6, edgecolor="none", color="blue",
            label="Column survival norm" if layer_idx == 0 else None,
        )

    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    plt.yscale("log")
    plt.xlabel("Layer (0 = embedding)")
    plt.ylabel(r"Survival norm  $\| (I - \Pi_B)\, u \|$")
    plt.title(f"Survival norms after orthogonal projection — {model_tag}")
    plt.xticks(range(0, num_layers, 1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layer_groups(overlap_scores, out_path: Path, model_tag: str):
    '''Mean overlap split into early / middle / late thirds of the layers.'''
    n = len(overlap_scores)
    third = n // 3
    early = overlap_scores[:third]
    mid   = overlap_scores[third:2 * third]
    late  = overlap_scores[2 * third:]

    means = [early.mean(), mid.mean(), late.mean()]
    labels = [f"early (0–{third-1})",
              f"middle ({third}–{2*third-1})",
              f"late ({2*third}–{n-1})"]

    fig = plt.figure(figsize=(8, 5))
    plt.bar(labels, means, color=["#4c72b0", "#dd8452", "#55a868"])
    plt.ylabel("Mean subspace overlap score")
    plt.title(f"Overlap by layer group — {model_tag}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Subspace pipeline
# ---------------------------------------------------------------------------

def run_pipeline(harmful, harmless, utility, r: int):
    '''
    Build per-layer safety and utility subspaces, then compute overlap
    metrics and the orthogonal-projection isolation of safety from utility.

    Convention (matches the original ActSVD notebook):
      D_safety  = harmful  - mean(harmless)
      D_utility = utility  - mean(harmless)
    so harmless prompts act as the shared baseline.
    '''
    n_layers = harmful.shape[1]

    harmless_mean = harmless.mean(axis=0, keepdims=True)  # (1, L, d)
    D_safety  = harmful  - harmless_mean
    D_utility = utility  - harmless_mean

    safety_subs  = actsvd_top(D_safety,  r)
    utility_subs = actsvd_top(D_utility, r)

    overlap_scores, top_cos = per_layer_overlap(safety_subs, utility_subs)

    safety_orth, survival_norms = actsvd_op(safety_subs, utility_subs)

    # Contrastive subspace from paired (harmful, harmless) diffs.
    # Truncate to the smaller set when counts differ.
    n_pair = min(harmful.shape[0], harmless.shape[0])
    contrast_subs = contrastive_svd(
        harmful[:n_pair], harmless[:n_pair], r
    )

    return {
        "n_layers":       n_layers,
        "safety_subs":    safety_subs,
        "utility_subs":   utility_subs,
        "safety_orth":    safety_orth,
        "survival_norms": survival_norms,
        "contrast_subs":  contrast_subs,
        "overlap_scores": overlap_scores,
        "top_cos":        top_cos,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_summary(result, model_tag: str, r: int, out_path: Path):
    '''Plain-text summary mirroring the printout style of the notebook.'''
    overlap = result["overlap_scores"]
    top_cos = result["top_cos"]
    n_layers = result["n_layers"]

    third = n_layers // 3
    early = overlap[:third].mean()
    mid   = overlap[third:2 * third].mean()
    late  = overlap[2 * third:].mean()

    best  = np.argsort(overlap)[:5].tolist()
    worst = np.argsort(overlap)[-5:].tolist()

    mean_overlap = float(np.mean(overlap))
    if mean_overlap < 0.2:
        verdict = "Safety and utility are largely ORTHOGONAL (independent)."
    elif mean_overlap < 0.5:
        verdict = "Safety and utility show MODERATE overlap (partial entanglement)."
    else:
        verdict = "Safety and utility are strongly ENTANGLED."

    # mean survival norm per layer — handy companion stat for the verdict
    surv_means = np.array([np.mean(s) for s in result["survival_norms"]])
    surv_global = float(surv_means.mean())

    lines = [
        f"ActSVD analysis — {model_tag}",
        f"r = {r}, n_layers = {n_layers}",
        "",
        "=== OVERALL OVERLAP SCORES ===",
        f"Min overlap : {overlap.min():.4f} (layer {int(np.argmin(overlap))})",
        f"Max overlap : {overlap.max():.4f} (layer {int(np.argmax(overlap))})",
        f"Mean overlap: {mean_overlap:.4f}",
        "",
        "=== TOP PRINCIPAL-ANGLE COSINE ===",
        f"Min: {top_cos.min():.4f} (layer {int(np.argmin(top_cos))})",
        f"Max: {top_cos.max():.4f} (layer {int(np.argmax(top_cos))})",
        "",
        "=== SURVIVAL NORMS (after removing utility from safety) ===",
        f"Mean across layers: {surv_global:.4f}  "
        f"(1.0 = fully orthogonal, 0.0 = fully entangled)",
        f"Min layer mean    : {surv_means.min():.4f} "
        f"(layer {int(np.argmin(surv_means))})",
        f"Max layer mean    : {surv_means.max():.4f} "
        f"(layer {int(np.argmax(surv_means))})",
        "",
        "=== LAYER GROUPS ===",
        f"early  (0–{third-1})       : mean overlap = {early:.4f}",
        f"middle ({third}–{2*third-1}): mean overlap = {mid:.4f}",
        f"late   ({2*third}–{n_layers-1}): mean overlap = {late:.4f}",
        "",
        "=== MOST SEPARATED LAYERS ===",
        *[f"  layer {l}: overlap = {overlap[l]:.4f}" for l in best],
        "",
        "=== MOST OVERLAPPING LAYERS ===",
        *[f"  layer {l}: overlap = {overlap[l]:.4f}" for l in worst],
        "",
        "=== INTERPRETATION ===",
        verdict,
        "",
    ]
    out_path.write_text("\n".join(lines))


def save_arrays(result, out_dir: Path, model_tag: str):
    '''
    Persists per-layer subspaces and metrics so downstream intervention
    experiments can reuse them without re-running SVD.
    '''
    safety_stack   = np.stack(result["safety_subs"])
    utility_stack  = np.stack(result["utility_subs"])
    contrast_stack = np.stack(result["contrast_subs"])

    # safety_orth columns may differ in count under degenerate ranks
    orth_shapes = {U.shape for U in result["safety_orth"]}
    if len(orth_shapes) == 1:
        safety_orth_stack = np.stack(result["safety_orth"])
        np.save(out_dir / f"safety_orth_{model_tag}.npy", safety_orth_stack)
    else:
        safety_orth_obj = np.empty(len(result["safety_orth"]), dtype=object)
        for i, U in enumerate(result["safety_orth"]):
            safety_orth_obj[i] = U
        np.save(out_dir / f"safety_orth_{model_tag}.npy",
                safety_orth_obj, allow_pickle=True)

    np.save(out_dir / f"safety_subspaces_{model_tag}.npy",   safety_stack)
    np.save(out_dir / f"utility_subspaces_{model_tag}.npy",  utility_stack)
    np.save(out_dir / f"contrast_subspaces_{model_tag}.npy", contrast_stack)

    # survival norms — one 1d vector per layer, all the same length here
    sv_stack = np.stack(result["survival_norms"])  # (n_layers, r)
    np.save(out_dir / f"survival_norms_{model_tag}.npy", sv_stack)

    np.savez(
        out_dir / f"overlap_metrics_{model_tag}.npz",
        overlap_scores=result["overlap_scores"],
        top_cos=result["top_cos"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run ActSVD on pre-extracted activations and save results."
    )
    p.add_argument("--activations-dir", required=True,
                   help="Directory containing harmful/harmless/utility .npy files.")
    p.add_argument("--suffix", required=True,
                   help="Filename suffix used during extraction (e.g. '4b').")
    p.add_argument("--model-tag", required=True,
                   help="Subdirectory name under svd/ for this model's outputs.")
    p.add_argument("--r", type=int, default=10,
                   help="Subspace rank (top-r singular vectors). Default: 10.")
    p.add_argument("--out-root", default="svd",
                   help="Root output directory. Default: svd")
    return p.parse_args()


def main():
    args = parse_args()
    activations_dir = Path(args.activations_dir)
    out_dir = Path(args.out_root) / args.model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading activations from {activations_dir} (suffix '{args.suffix}')")
    harmful, harmless, utility = load_activations(activations_dir, args.suffix)

    # cap r so SVD never asks for more components than data supports
    hidden_dim = harmful.shape[2]
    n_min = min(harmful.shape[0], harmless.shape[0], utility.shape[0])
    r = min(args.r, hidden_dim, n_min)
    if r != args.r:
        print(f"Reducing r from {args.r} to {r} based on data dimensions.")

    print(f"\nRunning ActSVD pipeline (r={r})...")
    result = run_pipeline(harmful, harmless, utility, r)
    print("Done.\n")

    save_arrays(result, out_dir, args.model_tag)

    metrics = {
        "model_tag":         args.model_tag,
        "r":                 r,
        "n_layers":          result["n_layers"],
        "overlap_scores":    result["overlap_scores"].tolist(),
        "top_cos":           result["top_cos"].tolist(),
        "mean_overlap":      float(result["overlap_scores"].mean()),
        "min_overlap":       float(result["overlap_scores"].min()),
        "max_overlap":       float(result["overlap_scores"].max()),
        "min_overlap_layer": int(np.argmin(result["overlap_scores"])),
        "max_overlap_layer": int(np.argmax(result["overlap_scores"])),
    }
    (out_dir / f"metrics_{args.model_tag}.json").write_text(
        json.dumps(metrics, indent=2)
    )

    plot_overlap(
        result["overlap_scores"], result["top_cos"],
        out_dir / f"overlap_{args.model_tag}.png", r, args.model_tag,
    )
    plot_survival(
        result["survival_norms"],
        out_dir / f"survival_{args.model_tag}.png", args.model_tag,
    )
    plot_layer_groups(
        result["overlap_scores"],
        out_dir / f"layer_groups_{args.model_tag}.png", args.model_tag,
    )

    write_summary(result, args.model_tag, r,
                  out_dir / f"summary_{args.model_tag}.txt")

    print(f"All outputs written to: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
