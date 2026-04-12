import os
import sys
import gc
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.append(str(Path("locatelayers").resolve()))
import layers_similarity

ACT_DIR = Path("activations")
OUT_DIR = Path("output")
ACT_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

MODELS = [
    ("google/gemma-3-4b-it", "4b"),
    ("google/gemma-3-12b-it", "12b"),
]

N = 100
MAX_LENGTH = 512

def load_prompts():
    print("loading (AdvBench, Alpaca, HelpSteer)...")

    advbench = load_dataset("walledai/AdvBench", split="train")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    helpsteer = load_dataset("nvidia/HelpSteer", split="train")

    harmful_prompts = list(advbench["prompt"])[:N]
    harmless_prompts = [x["instruction"] for x in alpaca][:N]
    utility_prompts = [x["prompt"] for x in helpsteer][:N]

    print(f"loaded all each have {len(harmful_prompts)} prompts。")
    return harmful_prompts, harmless_prompts, utility_prompts

def get_activations(prompt, model, tokenizer):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    # hidden_states: tuple[embedding + every layer]
    acts = [h[:, -1, :].float().cpu().numpy()[0] for h in outputs.hidden_states]
    return np.asarray(acts, dtype=np.float32)

def build_and_save_activations(model_name, suffix, harmful_prompts, harmless_prompts, utility_prompts):
    print("\n" + "=" * 60)
    print(f"start model: {model_name}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto"
    )
    model.eval()

    harmful_acts = np.asarray([
        get_activations(p, model, tokenizer)
        for p in tqdm(harmful_prompts, desc=f"Harmful ({suffix})")
    ], dtype=np.float32)

    harmless_acts = np.asarray([
        get_activations(p, model, tokenizer)
        for p in tqdm(harmless_prompts, desc=f"Harmless ({suffix})")
    ], dtype=np.float32)

    utility_acts = np.asarray([
        get_activations(p, model, tokenizer)
        for p in tqdm(utility_prompts, desc=f"Utility ({suffix})")
    ], dtype=np.float32)

    np.save(ACT_DIR / f"harmful_{suffix}_activations.npy", harmful_acts)
    np.save(ACT_DIR / f"harmless_{suffix}_activations.npy", harmless_acts)
    np.save(ACT_DIR / f"utility_{suffix}_activations.npy", utility_acts)

    print(f"save activation {ACT_DIR}/")

    # ===== DoM cosine similarity =====
    num_layers = harmful_acts.shape[1]
    cosine_similarities = []

    for layer in range(num_layers):
        harm_mean = harmful_acts[:, layer, :].mean(axis=0)
        help_mean = utility_acts[:, layer, :].mean(axis=0)
        alp_mean = harmless_acts[:, layer, :].mean(axis=0)

        refusal_direction = harm_mean - alp_mean
        utility_direction = help_mean - alp_mean

        r_norm = np.linalg.norm(refusal_direction)
        u_norm = np.linalg.norm(utility_direction)

        if r_norm < 1e-12 or u_norm < 1e-12:
            cosine_similarities.append(0.0)
        else:
            refusal_direction = refusal_direction / r_norm
            utility_direction = utility_direction / u_norm
            cosine_similarities.append(float(np.dot(refusal_direction, utility_direction)))

    plt.figure(figsize=(10, 5))
    plt.plot(range(num_layers), cosine_similarities, marker="o", linewidth=2)
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Layer number (0 is embedding layer)")
    plt.ylabel("Cosine similarity")
    plt.title(f"Safety vs Utility Cosine Similarity ({model_name})")
    plt.xticks(range(0, num_layers, max(1, num_layers // 20)))
    plt.yticks(np.arange(-1, 1.1, 0.1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"dom_cosine_sim_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"DoM saved graph {OUT_DIR / f'dom_cosine_sim_{suffix}.png'}")

    # ===== layers similarity =====
    print(f"use layers_similarity to eval {suffix} ...")
    r = 500

    nn_mean, nn_std = layers_similarity.analysis(harmless_acts, harmless_acts, r)
    mm_mean, mm_std = layers_similarity.analysis(harmful_acts, harmful_acts, r)
    nm_mean, nm_std = layers_similarity.analysis(harmless_acts, harmful_acts, r)

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
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(title)
        ax.set_xticks(layers)
        ax.set_xticklabels(layer_labels, rotation=90, fontsize=7)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Layer-wise Similarity Analysis ({model_name})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"layer_similarity_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"saved graph {OUT_DIR / f'layer_similarity_{suffix}.png'}")

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"model {model_name} completed\n")

if __name__ == "__main__":
    harmful_prompts, harmless_prompts, utility_prompts = load_prompts()

    for model_name, suffix in MODELS:
        build_and_save_activations(
            model_name=model_name,
            suffix=suffix,
            harmful_prompts=harmful_prompts,
            harmless_prompts=harmless_prompts,
            utility_prompts=utility_prompts,
        )

    print("all task succeed！")
