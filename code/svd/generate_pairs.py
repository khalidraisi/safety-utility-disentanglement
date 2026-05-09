# Generate paired harmless rewrites of AdvBench prompts using OpenAI's API
# (GPT-5 nano by default), with concurrent async requests.
#
# For each AdvBench prompt, generate a benign rewrite that:
#   - uses the same imperative verb (Write/Give/Provide/Teach/Create/Develop/...)
#   - uses the same meta-task (tutorial, instructions, strategy, script, ...)
#   - has approximately the same length (within ~20% token count)
#   - preserves grammatical structure and register
#   - swaps ONLY the harmful target for a benign one of comparable specificity
#
# Saves to a JSON file aligned by AdvBench train index, so harmful_i and
# harmless_paired_i are matched pairs by construction.
#
# Requires:  pip install openai datasets tqdm
# Env var :  OPENAI_API_KEY
#
# Example:
#   python3 generate_paired_harmless.py
#   python3 generate_paired_harmless.py --resume        # if interrupted
#   python3 generate_paired_harmless.py --concurrency 16
 
import argparse
import asyncio
import json
import os
import random
from pathlib import Path
 
from datasets import load_dataset
from tqdm import tqdm
 
 
_REWRITE_SYSTEM = """You rewrite harmful instructions into benign ones for AI safety research.
 
Given a harmful instruction, produce a paired harmless version that:
1. Uses the SAME imperative verb (Write/Give/Provide/Teach/Create/Develop/etc.)
2. Uses the SAME meta-task (tutorial, instructions, strategy, script, website, etc.)
3. Has APPROXIMATELY the same length (within 20% token count)
4. Preserves the SAME grammatical structure and register
5. Swaps ONLY the harmful target for a benign target of comparable specificity
 
The harmless version should be a coherent, realistic instruction that someone might genuinely send to an assistant.
 
Do NOT add hedging, politeness markers, or ethical caveats that aren't in the original.
Do NOT switch from imperative to question form.
Do NOT make the rewrite vague or trivially short.
 
Respond ONLY with the rewritten instruction. No preamble, no quotes, no explanation."""
 
_REWRITE_USER_TEMPLATE = "Harmful instruction:\n{prompt}\n\nPaired harmless rewrite:"
 
 
def _clean(text):
    text = text.strip()
    if len(text) >= 2 and text[0] in '"\u201c' and text[-1] in '"\u201d':
        text = text[1:-1].strip()
    return text
 
 
async def _rewrite_one(client, model, prompt, semaphore, max_retries=4):
    delay = 1.0
    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _REWRITE_SYSTEM},
                        {"role": "user", "content": _REWRITE_USER_TEMPLATE.format(prompt=prompt)},
                    ],
                )
                return _clean(resp.choices[0].message.content)
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[rewrite] giving up after {max_retries} attempts: {e}")
                    return None
                await asyncio.sleep(delay)
                delay *= 2
    return None
 
 
def _save(pairs, path):
    sorted_pairs = sorted(pairs.values(), key=lambda x: x["idx"])
    path.write_text(json.dumps(
        {"n": len(sorted_pairs), "pairs": sorted_pairs}, indent=2,
    ))
 
 
async def _rewrite_all(prompts, existing, model, concurrency, out_path):
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)
 
    todo = [(i, p) for i, p in enumerate(prompts) if i not in existing]
    pairs = dict(existing)
    pbar = tqdm(total=len(todo), desc="rewriting")
 
    async def _wrapped(idx, prompt):
        result = await _rewrite_one(client, model, prompt, semaphore)
        pbar.update(1)
        return idx, prompt, result
 
    tasks = [_wrapped(i, p) for i, p in todo]
 
    try:
        completed = 0
        for coro in asyncio.as_completed(tasks):
            idx, prompt, harmless = await coro
            if harmless is not None:
                pairs[idx] = {"idx": idx, "harmful": prompt, "harmless_paired": harmless}
            completed += 1
            if completed % 25 == 0:
                _save(pairs, out_path)
    finally:
        pbar.close()
        _save(pairs, out_path)
 
    return pairs
 
 
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500,
                    help="Number of AdvBench prompts (matches extract_activations.py default)")
    ap.add_argument("--model", default="gpt-5-nano",
                    help="OpenAI model for rewriting")
    ap.add_argument("--out", default="prompts/advbench_paired_harmless.json")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from existing output file (skip already-rewritten indices)")
    ap.add_argument("--concurrency", type=int, default=8)
    return ap.parse_args()
 
 
def main():
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
 
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
 
    print("Loading AdvBench...")
    advbench = load_dataset("walledai/AdvBench")
    prompts = advbench["train"]["prompt"][:args.n]
 
    existing = {}
    if args.resume and out_path.exists():
        data = json.loads(out_path.read_text())
        for entry in data["pairs"]:
            existing[entry["idx"]] = entry
        print(f"Resuming with {len(existing)} pairs already in file.")
 
    pairs = asyncio.run(
        _rewrite_all(prompts, existing, args.model, args.concurrency, out_path)
    )
 
    print(f"\nDone. {len(pairs)} pairs written to {out_path}")
    sample = random.sample(list(pairs.values()), min(8, len(pairs)))
    print("\n=== SPOT CHECK ===")
    for p in sample:
        print(f"\n[{p['idx']}]")
        print(f"  HARMFUL : {p['harmful']}")
        print(f"  HARMLESS: {p['harmless_paired']}")
 
 
if __name__ == "__main__":
    main()