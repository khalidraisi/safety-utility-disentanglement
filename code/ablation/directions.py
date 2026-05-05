# Compute DoM directions (safety, utility) from cached activations and pick
# per-direction layers by AUROC of class separability:
#   safety layer  = argmax_L AUROC(harmful vs harmless | proj onto V_safety[L])
#   utility layer = argmax_L AUROC(utility vs harmless | proj onto V_utility[L])

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def _unit(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v / max(n, eps)


def compute_dom_directions(harmless, harmful, utility):
    """
    Each input has shape (N, n_layers, D) where index 0 is the embedding.
    Returns v_safety, v_utility each shape (n_layers, D), unit-normalized.
    """
    n_layers = harmless.shape[1]
    D = harmless.shape[2]
    v_safety = np.zeros((n_layers, D), dtype=np.float32)
    v_utility = np.zeros((n_layers, D), dtype=np.float32)
    for L in range(n_layers):
        base_mu = harmless[:, L, :].mean(axis=0)
        v_safety[L]  = _unit(harmful[:, L, :].mean(axis=0)  - base_mu)
        v_utility[L] = _unit(utility[:, L, :].mean(axis=0) - base_mu)
    return v_safety, v_utility


def auroc_per_layer(group_pos, group_neg, V):
    """
    AUROC of (activations @ V[L]) separating group_pos from group_neg, per layer.
    group_*: (N, n_layers, D); V: (n_layers, D). Returns (n_layers,) array.
    """
    n_layers = V.shape[0]
    aurocs = np.zeros(n_layers)
    for L in range(n_layers):
        v = V[L] / (np.linalg.norm(V[L]) + 1e-12)
        s_pos = group_pos[:, L, :] @ v
        s_neg = group_neg[:, L, :] @ v
        scores = np.concatenate([s_pos, s_neg])
        labels = np.concatenate([np.ones(len(s_pos)), np.zeros(len(s_neg))])
        aurocs[L] = roc_auc_score(labels, scores)
    return aurocs


def pick_auroc_layer(aurocs, exclude=("first", "last")):
    """
    Return (L*, auroc_at_L*). Excludes embedding (idx 0) and final layer
    by default to avoid degenerate picks.
    """
    n = len(aurocs)
    mask = np.ones(n, dtype=bool)
    if "first" in exclude:
        mask[0] = False
    if "last" in exclude:
        mask[-1] = False
    pool = np.where(mask)[0]
    best = int(pool[np.argmax(aurocs[pool])])
    return best, float(aurocs[best])


def load_all(activations_dir, suffix, dom_dir=None):
    """
    dom_dir is accepted for backwards-compat but not required for AUROC layer pick.
    If provided, the cos_sim/null_lower at L_safety and L_utility are recorded.
    """
    activations_dir = Path(activations_dir)

    harmless = np.load(activations_dir / f"harmless_{suffix}_activations.npy")
    harmful  = np.load(activations_dir / f"harmful_{suffix}_activations.npy")
    utility  = np.load(activations_dir / f"utility_{suffix}_activations.npy")

    v_safety_all, v_utility_all = compute_dom_directions(harmless, harmful, utility)

    safety_aurocs  = auroc_per_layer(harmful, harmless, v_safety_all)
    utility_aurocs = auroc_per_layer(utility, harmless, v_utility_all)

    L_safety,  safety_auroc  = pick_auroc_layer(safety_aurocs)
    L_utility, utility_auroc = pick_auroc_layer(utility_aurocs)

    info = {
        "v_safety_all": v_safety_all,
        "v_utility_all": v_utility_all,
        "v_safety": v_safety_all[L_safety],
        "v_utility": v_utility_all[L_utility],
        "L_safety": L_safety,
        "L_utility": L_utility,
        "safety_auroc": safety_auroc,
        "utility_auroc": utility_auroc,
        "safety_aurocs": safety_aurocs,
        "utility_aurocs": utility_aurocs,
        "hidden_dim": int(harmless.shape[-1]),
        "n_layers": int(harmless.shape[1]),
    }

    if dom_dir is not None:
        dom_dir = Path(dom_dir)
        try:
            cos_sim    = np.load(dom_dir / "cosine_similarities.npy")
            null_lower = np.load(dom_dir / "null_lower.npy")
            info["cos_sim_at_L_safety"]    = float(cos_sim[L_safety])
            info["null_lower_at_L_safety"] = float(null_lower[L_safety])
            info["cos_sim_at_L_utility"]    = float(cos_sim[L_utility])
            info["null_lower_at_L_utility"] = float(null_lower[L_utility])
        except FileNotFoundError:
            pass

    return info


def random_unit_direction(dim, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return _unit(v)
