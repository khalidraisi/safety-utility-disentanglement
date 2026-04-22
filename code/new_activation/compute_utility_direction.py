# compute_utility_direction.py
#
# Computes V_utility from the 4 activation groups and evaluates how strong
# the utility direction is at each layer of the model.
#
# V_utility = mean(helpful activations) - mean(evasive activations)
#
# Output:
#   - prints per-layer AUROC, best layer, cosine similarity with V_safety
#   - saves plots to <activations_dir>/:
#       utility_auroc.png      - how well V_utility separates helpful/evasive
#       utility_projection.png - distribution of helpful vs evasive projections
#       direction_norms.png    - magnitude of V_safety and V_utility per layer
#       safety_vs_utility.png  - cosine similarity between the two directions
#
# HOW TO RUN:
#   python3 compute_utility_direction.py \
#     --activations-dir test_activations/gemma3_1b \
#     --suffix gemma3_1b

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def load_activations(activations_dir, suffix):
    # load all 4 activation groups
    # each has shape (n_prompts, n_layers+1, hidden_dim)
    d = Path(activations_dir)
    harmful  = np.load(d / f"harmful_{suffix}_activations.npy")
    harmless = np.load(d / f"harmless_{suffix}_activations.npy")
    helpful  = np.load(d / f"helpful_{suffix}_activations.npy")
    evasive  = np.load(d / f"evasive_{suffix}_activations.npy")
    print(f"Loaded activations:")
    print(f"  harmful  {harmful.shape}")
    print(f"  harmless {harmless.shape}")
    print(f"  helpful  {helpful.shape}")
    print(f"  evasive  {evasive.shape}")
    return harmful, harmless, helpful, evasive


def compute_dom_direction(group_on, group_off):
    # DoM: mean of "concept on" minus mean of "concept off"
    # input shapes:  (n_prompts, n_layers, hidden_dim)
    # output shape:  (n_layers, hidden_dim)
    return group_on.mean(axis=0) - group_off.mean(axis=0)


def auroc_per_layer(group_on, group_off, V):
    # for each layer, project both groups onto V[layer] and compute AUROC
    # AUROC = 1.0 means perfect separation
    # AUROC = 0.5 means the direction doesn't separate the two groups
    n_layers = V.shape[0]
    aurocs = np.zeros(n_layers)
    for l in range(n_layers):
        v = V[l] / (np.linalg.norm(V[l]) + 1e-12)  # unit vector
        scores_on  = group_on[:,  l, :] @ v
        scores_off = group_off[:, l, :] @ v
        scores = np.concatenate([scores_on, scores_off])
        labels = np.concatenate([np.ones(len(scores_on)), np.zeros(len(scores_off))])
        aurocs[l] = roc_auc_score(labels, scores)
    return aurocs


def project_onto_direction(group, V, layer):
    # project every activation in `group` at `layer` onto V[layer]
    # returns one score per prompt -> shape (n_prompts,)
    v = V[layer] / (np.linalg.norm(V[layer]) + 1e-12)
    return group[:, layer, :] @ v


def cosine(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--activations-dir", required=True,
                   help="Directory containing the 4 .npy activation files")
    p.add_argument("--suffix", required=True,
                   help="Filename suffix used during extraction, e.g. gemma3_1b")
    args = p.parse_args()

    out_dir = Path(args.activations_dir)

    # 1. load activations
    harmful, harmless, helpful, evasive = load_activations(
        args.activations_dir, args.suffix
    )
    n_layers = helpful.shape[1]

    # 2. compute DoM directions
    V_safety  = compute_dom_direction(harmful, harmless)
    V_utility = compute_dom_direction(helpful, evasive)
    print(f"\nV_safety  shape: {V_safety.shape}")
    print(f"V_utility shape: {V_utility.shape}")

    # 3. AUROC per layer for both directions
    print("\nComputing AUROC per layer...")
    safety_auroc  = auroc_per_layer(harmful, harmless, V_safety)
    utility_auroc = auroc_per_layer(helpful, evasive, V_utility)

    best_safety_layer  = int(np.argmax(safety_auroc))
    best_utility_layer = int(np.argmax(utility_auroc))

    print(f"\nBest safety layer:  {best_safety_layer:2d}  "
          f"AUROC={safety_auroc[best_safety_layer]:.3f}")
    print(f"Best utility layer: {best_utility_layer:2d}  "
          f"AUROC={utility_auroc[best_utility_layer]:.3f}")

    # how to interpret AUROC:
    # 0.5     -> direction is useless
    # 0.6-0.7 -> weak signal
    # 0.8-0.9 -> strong signal
    # >0.95   -> very strong, direction captures the concept well

    # 4. cosine similarity between safety and utility at every layer
    cos_per_layer = np.array([
        cosine(V_safety[l], V_utility[l]) for l in range(n_layers)
    ])
    cos_best = cosine(V_safety[best_safety_layer], V_utility[best_utility_layer])
    print(f"\ncos(V_safety[best], V_utility[best]) = {cos_best:.3f}")
    print("(close to 0 = orthogonal / independent concepts)")
    print("(close to 1 or -1 = highly aligned / entangled concepts)")

    # 5. direction norms per layer
    # bigger norm = stronger systematic difference between the two groups
    safety_norms  = np.linalg.norm(V_safety,  axis=1)
    utility_norms = np.linalg.norm(V_utility, axis=1)

    # --------------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------------

    # PLOT 1: AUROC per layer (is the direction strong?)
    plt.figure(figsize=(8, 5))
    plt.plot(safety_auroc,  "o-", label="safety  (harmful vs harmless)")
    plt.plot(utility_auroc, "s-", label="utility (helpful vs evasive)")
    plt.axhline(0.5, color="gray", linestyle="--", label="chance (0.5)")
    plt.axvline(best_utility_layer, color="orange", alpha=0.3,
                label=f"best utility layer ({best_utility_layer})")
    plt.xlabel("layer")
    plt.ylabel("AUROC")
    plt.title(f"Direction strength per layer ({args.suffix})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "utility_auroc.png", dpi=150)
    plt.close()
    print(f"\nSaved {out_dir / 'utility_auroc.png'}")

    # PLOT 2: projection distributions at best utility layer
    # if V_utility is strong, helpful and evasive histograms should separate cleanly
    helpful_scores = project_onto_direction(helpful, V_utility, best_utility_layer)
    evasive_scores = project_onto_direction(evasive, V_utility, best_utility_layer)

    plt.figure(figsize=(8, 5))
    plt.hist(helpful_scores, bins=30, alpha=0.6, label="helpful", color="green")
    plt.hist(evasive_scores, bins=30, alpha=0.6, label="evasive", color="orange")
    plt.xlabel(f"projection onto V_utility (layer {best_utility_layer})")
    plt.ylabel("count")
    plt.title(f"Utility direction projections ({args.suffix})\n"
              f"AUROC={utility_auroc[best_utility_layer]:.3f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "utility_projection.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'utility_projection.png'}")

    # PLOT 3: direction norms per layer (magnitude of the signal)
    plt.figure(figsize=(8, 5))
    plt.plot(safety_norms,  "o-", label="||V_safety||")
    plt.plot(utility_norms, "s-", label="||V_utility||")
    plt.xlabel("layer")
    plt.ylabel("L2 norm")
    plt.title(f"Direction magnitudes per layer ({args.suffix})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "direction_norms.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'direction_norms.png'}")

    # PLOT 4: cosine similarity between safety and utility per layer
    # this is the main orthogonality result of the project
    plt.figure(figsize=(8, 5))
    plt.plot(cos_per_layer, "o-", color="purple")
    plt.axhline(0, color="gray", linestyle="--", label="orthogonal (cos=0)")
    plt.xlabel("layer")
    plt.ylabel("cos(V_safety, V_utility)")
    plt.title(f"Safety vs utility cosine similarity per layer ({args.suffix})\n"
              f"at best layers: cos = {cos_best:.3f}")
    plt.ylim(-1, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "safety_vs_utility.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'safety_vs_utility.png'}")

    # 6. save the directions themselves for later use (e.g. intervention)
    np.save(out_dir / f"V_safety_{args.suffix}.npy",  V_safety)
    np.save(out_dir / f"V_utility_{args.suffix}.npy", V_utility)
    print(f"\nSaved V_safety and V_utility to {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
