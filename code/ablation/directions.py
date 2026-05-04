# Compute DoM directions (safety, utility) from cached activations and pick
# the layer L* where they are most disentangled per the DoM permutation test.

from pathlib import Path

import numpy as np


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


def pick_disentangled_layer(cos_sim, null_lower, exclude=("first", "last")):
    """
    Return L* = argmax of (null_lower - cos_sim) over layers where
    cos_sim < null_lower. Excludes the embedding layer (idx 0) and the
    final layer by default. Returns (L*, score, was_significant).
    """
    n = len(cos_sim)
    mask = np.ones(n, dtype=bool)
    if "first" in exclude:
        mask[0] = False
    if "last" in exclude:
        mask[-1] = False
    sig = cos_sim < null_lower
    candidates = np.where(mask & sig)[0]
    if len(candidates) == 0:
        # fall back: most-negative cos_sim among non-excluded layers
        idx_pool = np.where(mask)[0]
        best = idx_pool[np.argmin(cos_sim[idx_pool])]
        return int(best), float(null_lower[best] - cos_sim[best]), False
    scores = null_lower[candidates] - cos_sim[candidates]
    best = candidates[int(np.argmax(scores))]
    return int(best), float(null_lower[best] - cos_sim[best]), True


def load_all(activations_dir, suffix, dom_dir):
    activations_dir = Path(activations_dir)
    dom_dir = Path(dom_dir)

    harmless = np.load(activations_dir / f"harmless_{suffix}_activations.npy")
    harmful  = np.load(activations_dir / f"harmful_{suffix}_activations.npy")
    utility  = np.load(activations_dir / f"utility_{suffix}_activations.npy")

    cos_sim    = np.load(dom_dir / "cosine_similarities.npy")
    null_lower = np.load(dom_dir / "null_lower.npy")

    v_safety, v_utility = compute_dom_directions(harmless, harmful, utility)
    L_star, score, sig = pick_disentangled_layer(cos_sim, null_lower)

    return {
        "v_safety_all": v_safety,
        "v_utility_all": v_utility,
        "v_safety": v_safety[L_star],
        "v_utility": v_utility[L_star],
        "L_star": L_star,
        "score": score,
        "significant": sig,
        "cos_sim_at_L": float(cos_sim[L_star]),
        "null_lower_at_L": float(null_lower[L_star]),
        "hidden_dim": int(harmless.shape[-1]),
        "n_layers": int(harmless.shape[1]),
    }


def random_unit_direction(dim, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return _unit(v)
