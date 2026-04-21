# locatelayers/layers_similarity.py
# this is an implementation of the locating safety layers paper
# but we try to modify this a little bit so that we try to locate
#utility layers

# for our cases
# anchor = Alpaca always (for now at least)
# contrast = Helpsteer/Advbench for unsafe and helpful respectively
# use codes N-M (normal/malicious) and N-H (normal/helpful)

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

R = 500

def layer_cosine_sim(act1, act2, eps=1e-12):
    '''
    returns the cosine similarity of activations from two layers
    '''
    act1 = np.asarray(act1)
    act2 = np.asarray(act2)
    dots = np.sum(act1 * act2, axis=1)
    norms = np.linalg.norm(act1, axis=1) * np.linalg.norm(act2, axis=1)
    return dots / np.maximum(norms, eps)

# this is for section 3.2 of the paper

def analysis(acts1, acts2, r=R):
    '''
    finds the mean and std of layer-wise cos sim on
    r pairs
    acts1, acts2: shape = (N, num_layers, hidden_dim)
    return: mean, std for each layer
    '''
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

def get_similarity_plot(
    anchor_acts,
    contrast_acts,
    r=R,
    combined=False,
    anchor_label="anchor",
    contrast_label="contrast",
    anchor_code=None,
    contrast_code=None,
    model_name=None,
    save_path=None,
):
    '''
    plots a layer wise cosine similarity, shows plots of every possible pair
    of types that is anchor and anchor, anchor and contrast and contrast and
    contrast returns the mean and std of the cosine similarities of the
    possible pairs
    '''
    a_mean, a_std = analysis(anchor_acts,   anchor_acts,   r)
    c_mean, c_std = analysis(contrast_acts, contrast_acts, r)
    x_mean, x_std = analysis(anchor_acts,   contrast_acts, r)

    # if no codes for anchor/contrast then make them first letter of each,
    # this isn't ideal at all though so do use codes for this
    a_code = anchor_code or anchor_label[0].upper()
    c_code = contrast_code or contrast_label[0].upper()
    if a_code == c_code and anchor_code is None and contrast_code is None:
        a_code = anchor_label[:2].capitalize()
        c_code = contrast_label[:2].capitalize()

    layers = np.arange(len(a_mean))
    layer_labels = [str(i) for i in layers]

    configs = [
        (f"{anchor_label} vs {anchor_label} ({a_code}-{a_code})",
         a_mean, a_std, "tab:green"),
        (f"{contrast_label} vs {contrast_label} ({c_code}-{c_code})",
         c_mean, c_std, "tab:red"),
        (f"{anchor_label} vs {contrast_label} ({a_code}-{c_code})",
         x_mean, x_std, "tab:brown"),
    ]

    suptitle = f"Layer-wise Cosine Similarity: {anchor_label} vs {contrast_label}"
    if model_name:
        suptitle = f"{model_name} — {suptitle}"

    if combined:
        fig, ax = plt.subplots(figsize=(12, 5))
        for title, mean, std, color in configs:
            ax.plot(layers, mean, color=color, linewidth=2, label=title)
        ax.set_xlabel("Layer (Layer 0 is the embedding layer)")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(suptitle)
        ax.set_xticks(layers)
        ax.set_xticklabels(layer_labels, rotation=90, fontsize=7)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for ax, (title, mean, std, color) in zip(axes, configs):
            ax.plot(layers, mean, color=color, linewidth=2, label=title)
            ax.fill_between(layers, mean - std, mean + std,
                            color=color, alpha=0.2)
            ax.set_xlabel("Layer (Layer 0 is the embedding layer)")
            ax.set_ylabel("Cosine Similarity")
            ax.set_title(title)
            ax.set_xticks(layers)
            ax.set_xticklabels(layer_labels, rotation=90, fontsize=7)
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.suptitle(suptitle, fontsize=14, y=1.02)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return a_mean, a_std, c_mean, c_std, x_mean, x_std


# this is for section 3.3 of the paper

def layer_angles(acts1, acts2, eps=1e-12):
    '''
    the change in cos sim is not linear which makes it a little harder to interpret
    this is why it is converted to the angles instead
    '''
    cos_sim = layer_cosine_sim(acts1, acts2, eps=eps)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return np.arccos(cos_sim)

def angular_diffs_analysis(
    anchor_acts,
    contrast_acts,
    r=R,
    anchor_label="anchor",
    contrast_label="contrast",
):
    '''
    this measures the angle between the anchor and contrast activations 
    in the activation space relative to how different are the anchor
    activations amongst themselves
    '''

    #get size of both activations
    n_a = len(anchor_acts)
    n_c = len(contrast_acts)

    p_idx = np.random.randint(n_a, size=r)
    q_idx = np.random.randint(n_c, size=r)

    pp_raw = np.random.randint(n_a - 1, size=r)
    p_prime_idx = pp_raw + (pp_raw >= p_idx).astype(int)

    desc = f"angular-diff trials ({anchor_label} vs {contrast_label})"
    gaps = []
    for p, p_prime, q in tqdm(
        zip(p_idx, p_prime_idx, q_idx), total=r, desc=desc
    ):
        angle_contrast = layer_angles(anchor_acts[p], contrast_acts[q])
        angle_within   = layer_angles(anchor_acts[p], anchor_acts[p_prime])
        gaps.append(angle_contrast - angle_within)

    gaps = np.asarray(gaps)
    gaps = np.degrees(gaps)

    return gaps.mean(axis=0), gaps.std(axis=0)

# plot



# this is for section 3.4 (Not implemented yet, figure this out later), 
# it seems a little more complicated and some things would probably need to be 
# written within the experiments