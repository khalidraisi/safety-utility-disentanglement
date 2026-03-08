#I wanted to try separability in terms of utility, tried PCA 2D and 3D with neither seem seperable
#We may also try SAE? 

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os
import random
from datasets import load_dataset
import plotly.graph_objects as go

N = 10 #this is num of pronmts to test with
random.seed(42)

device = torch.device("cpu")
model_name = "google/gemma-3-1b-it" #it has 26 layers
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load datasets and pick 10 prompts
# safety

# Harmful: AdvBench harmful behaviours 
advbench = load_dataset("walledai/AdvBench", split="train")
harmful_prompts = random.sample(list(advbench["prompt"]), N)

# Harmless: Alpaca instructions (general helpful tasks)
alpaca = load_dataset("tatsu-lab/alpaca", split="train")
# Filter to entries with no input (standalone instructions) and reasonable length
alpaca_clean = [ex["instruction"] for ex in alpaca if ex["input"].strip() == "" and 10 < len(ex["instruction"]) < 200]
harmless_prompts = random.sample(alpaca_clean, N)

#utility
helpsteer = load_dataset("nvidia/HelpSteer2", split="train")

# High utility: prompts whose responses were rated highly helpful (helpfulness >= 3)
high_util_candidates = [ex["prompt"] for ex in helpsteer if ex["helpfulness"] >= 3 and 10 < len(ex["prompt"]) < 300]
high_utility_prompts = random.sample(high_util_candidates, N)
# Low utility: prompts whose responses were rated low helpfulness (helpfulness <= 1)
low_util_candidates = [ex["prompt"] for ex in helpsteer if ex["helpfulness"] <= 1 and 10 < len(ex["prompt"]) < 300]
low_utility_prompts = random.sample(low_util_candidates, N)

def get_activation(prompt, layer=-1):
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer][0, -1, :].cpu().numpy()
    return hidden


def extract_all(prompts, label, layer):
    activations = []
    for i, p in enumerate(prompts):
        print(f"  [{label}] ({i+1}/{len(prompts)}) {p[:60]}...")
        activations.append(get_activation(p, layer=layer))
    return np.array(activations)


#safety shown to be separable in last layer using PCA, test other layers for utility (26 total layers)
acts_harmful   = extract_all(harmful_prompts,      "harmful", layer=-1)
acts_harmless  = extract_all(harmless_prompts,     "harmless", layer=-1) 
acts_high_util = extract_all(high_utility_prompts, "high-util", layer=-2)
acts_low_util  = extract_all(low_utility_prompts,  "low-util", layer=-2)

all_acts = np.concatenate([acts_harmful, acts_harmless, acts_high_util, acts_low_util])
pca = PCA(n_components=3)
reduced = pca.fit_transform(all_acts)

n1 = len(acts_harmful)
n2 = n1 + len(acts_harmless)
n3 = n2 + len(acts_high_util)

red_harmful   = reduced[:n1]
red_harmless  = reduced[n1:n2]
red_high_util = reduced[n2:n3]
red_low_util  = reduced[n3:]


#this was done with help from claude lol, nice to see by moving it around
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=red_high_util[:, 0], y=red_high_util[:, 1], z=red_high_util[:, 2],
    mode='markers', marker=dict(size=6, color='green', symbol='diamond'),
    name='High utility'
))
fig.add_trace(go.Scatter3d(
    x=red_low_util[:, 0], y=red_low_util[:, 1], z=red_low_util[:, 2],
    mode='markers', marker=dict(size=6, color='orange', symbol='square'),
    name='Low utility'
))
fig.update_layout(
    title="Utility PCA — 3D",
    scene=dict(
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
        zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)",
    )
)
fig.show()