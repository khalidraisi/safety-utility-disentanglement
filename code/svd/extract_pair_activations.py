#   python3 svd/extract_pair_activations.py \
#       --model google/gemma-3-4b-it \
#       --suffix 4b_n500 \
#       --target-dir activations/gemma3acts/4b \
#       --template gemma-3 \
#       --paired-prompts-json svd/prompts/advbench_paired_harmless.json
 
import argparse
import gc
import json
import sys
from pathlib import Path
 
# Make code/extract_activations.py importable regardless of CWD.
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
 
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
 
from extract_activations import get_template, load_model, TEMPLATES, N as DEFAULT_N
 
 
def extract_for_prompts(prompts, label, model, tok_or_proc, template_fn, is_processor):
    def get_activations(prompt):
        formatted = template_fn(prompt, tok_or_proc)
        if is_processor:
            inputs = tok_or_proc(
                text=formatted, return_tensors="pt", add_special_tokens=False,
            ).to(model.device)
        else:
            inputs = tok_or_proc(
                formatted, return_tensors="pt", add_special_tokens=False,
            ).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        return [h[:, -1, :].float().cpu().numpy()[0] for h in outputs.hidden_states]
 
    return np.array([get_activations(p) for p in tqdm(prompts, desc=label)])
 
 
def load_paired_harmless(json_path):
    data = json.loads(Path(json_path).read_text())
    return [entry["harmless_paired"] for entry in data["pairs"]]
 
 
def load_nonutility(n):
    '''
    HelpSteer entries with helpfulness=0 (response was rated unhelpful).
    Used as the index-aligned contrast against helpfulness=4 utility prompts.
    Note: the rating is on the response, not the prompt — pairing is index-only.
    '''
    helpsteer = load_dataset("nvidia/HelpSteer")
    prompts = [
        entry["prompt"]
        for entry in helpsteer["train"]
        if entry["helpfulness"] == 0
    ]
    prompts = list(dict.fromkeys(prompts))
    if len(prompts) < n:
        print(f"Warning: only {len(prompts)} helpfulness=0 prompts available "
              f"(requested {n}). Using all of them.")
    return prompts[:n]
 
 
def parse_args():
    ap = argparse.ArgumentParser(
        description="Extract activations for paired harmless + nonutility contrasts.",
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--template", default="none", choices=sorted(TEMPLATES.keys()))
    ap.add_argument("--paired-prompts-json", required=True,
                    help="Output of generate_pairs.py")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help=f"Cap for nonutility prompts (default: {DEFAULT_N})")
    return ap.parse_args()
 
 
def main():
    args = parse_args()
    template_fn = get_template(args.template)
    use_processor = args.template in {"gemma-4", "qwen-3-5"}
 
    out_dir = Path(args.target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    paired_harmless = load_paired_harmless(args.paired_prompts_json)
    nonutility = load_nonutility(args.n)
    print(f"Loaded {len(paired_harmless)} paired harmless prompts and "
          f"{len(nonutility)} nonutility (helpfulness=0) prompts.")
 
    model, tok_or_proc = load_model(args.model, use_processor=use_processor)
 
    paired_acts = extract_for_prompts(
        paired_harmless, "harmless_paired",
        model, tok_or_proc, template_fn, use_processor,
    )
    np.save(out_dir / f"harmless_paired_{args.suffix}_activations.npy", paired_acts)
    print(f"  -> harmless_paired_{args.suffix}_activations.npy  shape={paired_acts.shape}")
 
    nonutil_acts = extract_for_prompts(
        nonutility, "nonutility",
        model, tok_or_proc, template_fn, use_processor,
    )
    np.save(out_dir / f"nonutility_{args.suffix}_activations.npy", nonutil_acts)
    print(f"  -> nonutility_{args.suffix}_activations.npy  shape={nonutil_acts.shape}")
 
    del model, tok_or_proc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
 
if __name__ == "__main__":
    main()