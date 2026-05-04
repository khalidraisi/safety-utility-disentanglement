# Driver: causal-independence ablation for gemma-3-4b-it.
#
# Conditions: baseline, ablate-safety, ablate-utility, ablate-both, ablate-random.
# Benchmarks: AdvBench (refusal rate), MMLU subsample (accuracy).
# Same disentangled layer L* used for all DoM ablations; hooks attach at every
# block index >= L*.
#
# Example:
#   python3 run_experiment.py \
#     --model google/gemma-3-4b-it \
#     --activations-dir ../activations/gemma3acts/4b \
#     --suffix 4b_n500 \
#     --dom-dir ../dom/gemma3_4b \
#     --out-dir ../../results/gemma3_4b_causal \
#     --advbench-n 200 --mmlu-per-subject 5

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from directions import load_all, random_unit_direction
from hooks import projection_ablation
import bench_advbench
import bench_mmlu


CONDITIONS = ["baseline", "ablate_safety", "ablate_utility", "ablate_both", "ablate_random"]


def directions_for(condition, info, random_seed):
    if condition == "baseline":
        return None
    if condition == "ablate_safety":
        return np.stack([info["v_safety"]], axis=0)
    if condition == "ablate_utility":
        return np.stack([info["v_utility"]], axis=0)
    if condition == "ablate_both":
        return np.stack([info["v_safety"], info["v_utility"]], axis=0)
    if condition == "ablate_random":
        v = random_unit_direction(info["hidden_dim"], seed=random_seed)
        return np.stack([v], axis=0)
    raise ValueError(condition)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--activations-dir", required=True)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--dom-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--advbench-n", type=int, default=None,
                    help="subsample AdvBench (default: full 520)")
    ap.add_argument("--mmlu-per-subject", type=int, default=5)
    ap.add_argument("--advbench-batch-size", type=int, default=4)
    ap.add_argument("--mmlu-batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load DoM directions and pick L*
    info = load_all(args.activations_dir, args.suffix, args.dom_dir)
    print(f"L* = {info['L_star']}  (cos_sim={info['cos_sim_at_L']:.4f}, "
          f"null_lower={info['null_lower_at_L']:.4f}, "
          f"score={info['score']:.4f}, significant={info['significant']})")

    # 2) load model + tokenizer
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"loading model {args.model} on {device} ({args.dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # needed for generate() with batch
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device,
    )

    # 3) prepare benchmark inputs once
    print("loading benchmarks...")
    advbench_prompts = bench_advbench.load_advbench(n=args.advbench_n, seed=args.seed)
    mmlu_rows = bench_mmlu.load_mmlu_subsample(per_subject=args.mmlu_per_subject, seed=args.seed)
    print(f"  AdvBench: {len(advbench_prompts)} prompts")
    print(f"  MMLU:     {len(mmlu_rows)} questions across "
          f"{len(set(r['subject'] for r in mmlu_rows))} subjects")

    # 4) loop conditions x benchmarks
    results = {
        "model": args.model,
        "L_star": info["L_star"],
        "cos_sim_at_L": info["cos_sim_at_L"],
        "null_lower_at_L": info["null_lower_at_L"],
        "score": info["score"],
        "significant": info["significant"],
        "n_layers": info["n_layers"],
        "advbench_n": len(advbench_prompts),
        "mmlu_n": len(mmlu_rows),
        "conditions": {},
    }

    for cond in CONDITIONS:
        print(f"\n=== {cond} ===")
        dirs = directions_for(cond, info, random_seed=args.seed)

        with projection_ablation(model, dirs, start_block_idx=info["L_star"]):
            adv = bench_advbench.evaluate(
                model, tokenizer, advbench_prompts,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.advbench_batch_size,
                device=device,
            )
            mml = bench_mmlu.evaluate(
                model, tokenizer, mmlu_rows,
                batch_size=args.mmlu_batch_size,
                device=device,
            )

        results["conditions"][cond] = {
            "advbench_refusal_rate": adv["refusal_rate"],
            "mmlu_accuracy": mml["accuracy"],
            "mmlu_by_subject": mml["by_subject"],
        }
        print(f"  AdvBench refusal: {adv['refusal_rate']:.3f}")
        print(f"  MMLU accuracy:    {mml['accuracy']:.3f}")

        # save raw rows separately so the main JSON stays small
        raw = {"advbench_rows": adv["rows"], "mmlu_details": mml["details"]}
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
