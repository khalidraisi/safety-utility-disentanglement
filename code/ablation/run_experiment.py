# Driver: causal-independence ablation.
#
# Conditions: baseline, ablate-safety, ablate-utility, ablate-both, ablate-random.
# Benchmarks: AdvBench (refusal rate), HelpSteer (1-5 helpfulness via GPT judge).
#
# Layer pick: AUROC. Per-direction.
#   L_safety  = argmax_L AUROC( harmful  vs harmless | proj onto V_safety[L]  )
#   L_utility = argmax_L AUROC( utility vs harmless | proj onto V_utility[L] )
# Ablation hook is installed at one block (L_safety or L_utility) per direction.
#
# Example:
#   python3 run_experiment.py \
#     --model meta-llama/Llama-3.2-3B-Instruct \
#     --activations-dir ../activations/llama3acts \
#     --suffix 3b \
#     --out-dir /content/results/llama3_3b_causal \
#     --advbench-n 200 --helpsteer-n 150

import argparse
import gc
import json
import os
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from directions import load_all, random_unit_direction
from hooks import projection_ablation
import bench_advbench
import bench_helpsteer


CONDITIONS = ["baseline", "ablate_safety", "ablate_utility", "ablate_both", "ablate_random"]


def make_ablation(condition, info, model, random_seed):
    """
    Return a context manager that installs the appropriate hook(s) for the
    given condition, or a no-op context for baseline.
    """
    stack = ExitStack()

    if condition == "baseline":
        return stack

    if condition == "ablate_safety":
        stack.enter_context(projection_ablation(
            model, np.stack([info["v_safety"]], axis=0), block_idx=info["L_safety"]))
        return stack

    if condition == "ablate_utility":
        stack.enter_context(projection_ablation(
            model, np.stack([info["v_utility"]], axis=0), block_idx=info["L_utility"]))
        return stack

    if condition == "ablate_both":
        stack.enter_context(projection_ablation(
            model, np.stack([info["v_safety"]], axis=0), block_idx=info["L_safety"]))
        stack.enter_context(projection_ablation(
            model, np.stack([info["v_utility"]], axis=0), block_idx=info["L_utility"]))
        return stack

    if condition == "ablate_random":
        # Random control mirrors the safety scope: same layer, same rank.
        v = random_unit_direction(info["hidden_dim"], seed=random_seed)
        stack.enter_context(projection_ablation(
            model, np.stack([v], axis=0), block_idx=info["L_safety"]))
        return stack

    raise ValueError(condition)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--activations-dir", required=True)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--dom-dir", default=None,
                    help="optional; if given, cos_sim/null_lower at picked layers are recorded")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--advbench-n", type=int, default=None,
                    help="subsample AdvBench (default: full 520)")
    ap.add_argument("--helpsteer-n", type=int, default=150,
                    help="number of HelpSteer val prompts to evaluate per condition")
    ap.add_argument("--advbench-batch-size", type=int, default=4)
    ap.add_argument("--helpsteer-batch-size", type=int, default=4)
    ap.add_argument("--helpsteer-max-new-tokens", type=int, default=256)
    ap.add_argument("--judge-model", default="gpt-5-nano")
    ap.add_argument("--judge-concurrency", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64,
                    help="max_new_tokens for AdvBench generation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--L-utility", type=int, default=None,
                    help="override AUROC pick for utility layer")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set; required for HelpSteer judge.")

    # 1) load DoM directions and pick layers by AUROC
    info = load_all(args.activations_dir, args.suffix, args.dom_dir)
    if args.L_utility is not None:
        info["L_utility"] = args.L_utility
        info["v_utility"] = info["v_utility_all"][args.L_utility]
        info["utility_auroc"] = float(info["utility_aurocs"][args.L_utility])
        print(f"[override] L_utility = {args.L_utility}")
    print(f"L_safety  = {info['L_safety']}   AUROC = {info['safety_auroc']:.4f}")
    print(f"L_utility = {info['L_utility']}  AUROC = {info['utility_auroc']:.4f}")

    # 2) load model + tokenizer
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"loading model {args.model} on {device} ({args.dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device,
    )

    # 3) prepare benchmark inputs once
    print("loading benchmarks...")
    advbench_prompts = bench_advbench.load_advbench(n=args.advbench_n, seed=args.seed)
    helpsteer_prompts = bench_helpsteer.load_helpsteer(n=args.helpsteer_n, seed=args.seed)
    print(f"  AdvBench:  {len(advbench_prompts)} prompts")
    print(f"  HelpSteer: {len(helpsteer_prompts)} prompts (judge={args.judge_model})")

    # 4) loop conditions x benchmarks
    results = {
        "model": args.model,
        "L_safety": info["L_safety"],
        "L_utility": info["L_utility"],
        "safety_auroc": info["safety_auroc"],
        "utility_auroc": info["utility_auroc"],
        "n_layers": info["n_layers"],
        "advbench_n": len(advbench_prompts),
        "helpsteer_n": len(helpsteer_prompts),
        "judge_model": args.judge_model,
        "conditions": {},
    }
    for k in ("cos_sim_at_L_safety", "null_lower_at_L_safety",
              "cos_sim_at_L_utility", "null_lower_at_L_utility"):
        if k in info:
            results[k] = info[k]

    for cond in CONDITIONS:
        print(f"\n=== {cond} ===")

        with make_ablation(cond, info, model, random_seed=args.seed):
            adv = bench_advbench.evaluate(
                model, tokenizer, advbench_prompts,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.advbench_batch_size,
                device=device,
            )
            hs = bench_helpsteer.evaluate(
                model, tokenizer, helpsteer_prompts,
                judge_model=args.judge_model,
                max_new_tokens=args.helpsteer_max_new_tokens,
                batch_size=args.helpsteer_batch_size,
                judge_concurrency=args.judge_concurrency,
                device=device,
            )

        results["conditions"][cond] = {
            "advbench_refusal_rate": adv["refusal_rate"],
            "helpsteer_score": hs["mean_score"],
            "helpsteer_n_scored": hs["n_scored"],
        }
        print(f"  AdvBench refusal: {adv['refusal_rate']:.3f}")
        print(f"  HelpSteer score:  {hs['mean_score']:.3f}  ({hs['n_scored']}/{hs['n']} scored)")

        raw = {"advbench_rows": adv["rows"], "helpsteer_rows": hs["rows"]}
        with open(out_dir / f"raw_{cond}.json", "w") as f:
            json.dump(raw, f)

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    out_file = out_dir / "causal_ablation.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_file}")


if __name__ == "__main__":
    main()
