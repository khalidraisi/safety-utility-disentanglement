# Find the mean of helpfulness for method 2 of obtaining 
# useful behaviour direction
# for the working done to get what is done here refer to part A of the 
# appendix
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# How to use (EXAMPLE)
# python3 find_helpfulness_mean.py \
#     --model google/gemma-3-4b-it \
#     --template gemma-3 \
#     --n-pilot 100 \
#     --epsilon 1.0 \
#     --delta 0.05 \
#     --save-means means_{MODEL_NAME}.npy
#

import numpy as np
import torch
import argparse
import gc
import os
from tqdm import tqdm
from datasets import load_dataset
from extract_activations import load_model, get_template, TEMPLATES

GROUP_COUNTS = {0: 1223, 1: 2295, 2: 7310, 3: 16452, 4: 8051}


def load_helpsteer_by_helpfulness(n_per_group):
    '''
    Load n_per_group prompts for each helpfulness rating 0-4
    we do not consider the proportions until later
    '''
    helpsteer = load_dataset("nvidia/HelpSteer")
    by_group = {k: [] for k in range(5)}
    seen = {k: set() for k in range(5)}
    
    for entry in helpsteer['train']:
        h = entry['helpfulness']
        prompt = entry['prompt']
        if prompt not in seen[h] and len(by_group[h]) < n_per_group:
            by_group[h].append(prompt)
            seen[h].add(prompt)
        if all(len(v) >= n_per_group for v in by_group.values()):
            break
    
    return by_group


def extract_activations(prompts, model, tok_or_proc, template_fn, is_processor=False):
    '''
    Extract last-token activations.
    Returns: ndarray of shape (n_prompts, n_layers, hidden_dim)
    '''
    def _get_one(prompt):
        formatted = template_fn(prompt, tok_or_proc)
        if is_processor:
            inputs = tok_or_proc(
                text=formatted, return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)
        else:
            inputs = tok_or_proc(
                formatted, return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        return [h[:, -1, :].float().cpu().numpy()[0] for h in outputs.hidden_states]
    
    return np.array([_get_one(p) for p in tqdm(prompts, desc="activations")])


def per_layer_means(activations):
    '''
    Compute the mean activation at each layer for a single group.
    
    Args:
        activations: ndarray of shape (n_prompts, n_layers, hidden_dim)
    Returns:
        ndarray of shape (n_layers, hidden_dim) where index l is the mean
        activation vector at layer l, averaged over all prompts.
    '''
    return activations.mean(axis=0)


def per_layer_means_by_group(activations_by_group):
    '''
    Per-layer means for each group.
    
    Args:
        activations_by_group: dict {k: ndarray of shape (n_k, n_layers, hidden_dim)}
    Returns:
        dict {k: ndarray of shape (n_layers, hidden_dim)}
    '''
    return {k: per_layer_means(acts) for k, acts in activations_by_group.items()}


def estimator_per_layer(activations_by_group, p):
    '''
    Compute mu_hat = sum_k p_k * mean(activations_k) at each layer.
    
    Args:
        activations_by_group: dict {k: ndarray of shape (n_k, n_layers, hidden_dim)}
        p: array of group proportions, length 5
    Returns:
        ndarray of shape (n_layers, hidden_dim) — mu_hat at each layer.
    '''
    group_means = per_layer_means_by_group(activations_by_group)
    stacked = np.stack([group_means[k] for k in sorted(group_means.keys())])
    return (p[:, None, None] * stacked).sum(axis=0)


def per_layer_trace_sigma(activations):
    '''
    Compute tr(Sigma_k) at each layer for a single group.
    
    Args:
        activations: ndarray of shape (n_prompts, n_layers, hidden_dim)
    Returns:
        ndarray of shape (n_layers,) — trace of within-group covariance per layer.
    '''
    return activations.var(axis=0, ddof=1).sum(axis=-1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--template", default="none", choices=sorted(TEMPLATES.keys()))
    p.add_argument("--n-pilot", type=int, default=50,
                   help="Pilot samples per group")
    p.add_argument("--epsilon", type=float, required=True,
                   help="L2 tolerance on the mean estimate")
    p.add_argument("--delta", type=float, default=0.05,
                   help="Failure probability (default: 0.05)")
    p.add_argument("--save-means", default=None,
                   help=" Path to save per-layer mu_hat estimator (.npy)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    #build an np.array with contents proportional to the contents in HelpSteer
    total = sum(GROUP_COUNTS.values())
    p = np.array([GROUP_COUNTS[k] / total for k in range(5)])
    
    template_fn = get_template(args.template)
    use_processor = args.template == "gemma-4"
    model, tok_or_proc = load_model(args.model, use_processor=use_processor)
    
    prompts_by_group = load_helpsteer_by_helpfulness(args.n_pilot)
    
    #get the activations (num input n for now to get a good estimate of each group's mean)
    activations_per_group = {}
    for k in range(5):
        print(f"Group {k} (helpfulness={k}, p_k={p[k]:.4f})")
        acts = extract_activations(
            prompts_by_group[k], model, tok_or_proc,
            template_fn, is_processor=use_processor,
        )
        activations_per_group[k] = acts
    
    n_layers = activations_per_group[0].shape[1]
    
    mu_hat_per_layer = estimator_per_layer(activations_per_group, p)
    print(f"mu_hat shape: {mu_hat_per_layer.shape}  (n_layers, hidden_dim)\n")
    
    trace_per_group_per_layer = np.zeros((5, n_layers))
    for k in range(5):
        trace_per_group_per_layer[k] = per_layer_trace_sigma(activations_per_group[k])
    
    # tr(Sigma_bar) at each layer = sum_k p_k * tr(Sigma_k)
    trace_sigma_bar_per_layer = (p[:, None] * trace_per_group_per_layer).sum(axis=0)
    
    N_per_layer = np.ceil(
        trace_sigma_bar_per_layer / (args.delta * args.epsilon ** 2)
    ).astype(int)
    
    print(f"{'='*70}")
    print(f"Per-layer lower bounds  (epsilon={args.epsilon}, delta={args.delta})")
    print(f"{'='*70}")
    print(f"{'Layer':>6} {'tr(Sigma_bar)':>15} {'N_lower':>12}")
    print("-" * 35)
    for layer in range(n_layers):
        print(f"{layer:>6} {trace_sigma_bar_per_layer[layer]:>15.2f} "
              f"{N_per_layer[layer]:>12}")
    
    # Conservative bound: max over layers
    N_max = int(N_per_layer.max())
    worst_layer = int(N_per_layer.argmax())
    n_per_group = np.ceil(p * N_max).astype(int)
    
    print(f"\n{'='*70}")
    print(f"Conservative bound (covers all layers): N >= {N_max}")
    print(f"  (driven by layer {worst_layer})")
    print(f"\nPer-group sample sizes (proportional allocation):")
    for k in range(5):
        flag = "  WARNING: exceeds available" if n_per_group[k] > GROUP_COUNTS[k] else ""
        print(f"  Group {k}: n_{k} = {n_per_group[k]}  "
              f"(available: {GROUP_COUNTS[k]}){flag}")

    #just to be safe
    if args.save_means:
        os.makedirs("means", exist_ok=True)
        save_path = os.path.join("means", args.save_means)
        np.save(save_path, mu_hat_per_layer)
        print(f"Per-layer mu_hat saved to {save_path}")
    
    del model, tok_or_proc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()