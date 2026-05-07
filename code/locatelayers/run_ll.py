import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layers_similarity import get_similarity_plot, get_angular_diffs_plot
 
MODELS = {
    "gemma3_1b":  "../activations/gemma3acts/1b/harmless_1b_n500_activations.npy,"
                  "../activations/gemma3acts/1b/harmful_1b_n500_activations.npy,"
                  "../activations/gemma3acts/1b/utility_1b_n500_activations.npy",
    "gemma3_4b":  "../activations/gemma3acts/4b/harmless_4b_n500_activations.npy,"
                  "../activations/gemma3acts/4b/harmful_4b_n500_activations.npy,"
                  "../activations/gemma3acts/4b/utility_4b_n500_activations.npy",
    "gemma4_E2B": "../activations/gemma4acts/E2B/harmless_E2B_activations.npy,"
                  "../activations/gemma4acts/E2B/harmful_E2B_activations.npy,"
                  "../activations/gemma4acts/E2B/utility_E2B_activations.npy",
    "llama3_3b":  "../activations/llama3acts/harmless_3b_activations.npy,"
                  "../activations/llama3acts/harmful_3b_activations.npy,"
                  "../activations/llama3acts/utility_3b_activations.npy",
    "qwen35_4b":  "../activations/qwen35acts/harmless_4b_activations.npy,"
                  "../activations/qwen35acts/harmful_4b_activations.npy,"
                  "../activations/qwen35acts/utility_4b_activations.npy",
}
 
# anchor = baseline (Alpaca, "N"); contrasts = malicious ("M"), utility ("U")
CONTRASTS = [
    ("Malicious", "M", "N-M"),
    ("Useful", "U", "N-U"),
]
 
def run(name, paths, r):
    base_p, safe_p, util_p = paths.split(",")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    os.makedirs(out_dir, exist_ok=True)
 
    print(f"\n=== {name} ===")
    base = np.load(base_p)
    safe = np.load(safe_p)
    util = np.load(util_p)
    print(f"  shapes: base={base.shape} safety={safe.shape} utility={util.shape}")
 
    contrasts_data = {"Malicious": safe, "Useful": util}
 
    for contrast_label, contrast_code, tag in CONTRASTS:
        contrast = contrasts_data[contrast_label]
 
        a_mean, a_std, c_mean, c_std, x_mean, x_std = get_similarity_plot(
            anchor_acts=base, contrast_acts=contrast, r=r, combined=False,
            anchor_label="Normal", contrast_label=contrast_label,
            anchor_code="N", contrast_code=contrast_code, model_name=name,
            save_path=os.path.join(out_dir, f"similarity_{tag}.png"),
        )
        plt.close("all")
 
        ang_mean, ang_std = get_angular_diffs_plot(
            anchor_acts=base, contrast_acts=contrast, r=r,
            anchor_label="Normal", contrast_label=contrast_label,
            anchor_code="N", contrast_code=contrast_code, model_name=name,
            save_path=os.path.join(out_dir, f"angular_{tag}.png"),
        )
        plt.close("all")
 
    print(f"  saved -> {out_dir}")
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=1000)
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    args = ap.parse_args()
    for m in args.models:
        run(m, MODELS[m], args.r)