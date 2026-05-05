# HelpSteer utility benchmark, scored by GPT-5-nano as helpfulness judge.
# Generates a completion for each prompt under the active ablation context,
# then asks the judge to rate it 1 (not helpful) to 5 (very helpful).

import asyncio
import os
import re

import torch
from datasets import load_dataset
from tqdm import tqdm


_JUDGE_SYSTEM = (
    "You are an impartial judge rating the helpfulness of an AI assistant's "
    "response to a user's prompt. Rate on a scale of 1 (not helpful) to "
    "5 (very helpful). Respond with only a single integer between 1 and 5."
)

_JUDGE_USER_TEMPLATE = (
    "User prompt:\n{prompt}\n\n"
    "Assistant response:\n{completion}\n\n"
    "Rate the helpfulness (1-5). Respond with only the integer."
)

_SCORE_RE = re.compile(r"[1-5]")


def load_helpsteer(n=None, seed=0):
    ds = load_dataset("nvidia/HelpSteer", split="validation")
    prompts = list(dict.fromkeys(ds["prompt"]))
    if n is not None and n < len(prompts):
        import random
        rng = random.Random(seed)
        prompts = rng.sample(prompts, n)
    return prompts


@torch.no_grad()
def generate_completions(model, tokenizer, prompts, max_new_tokens=256, batch_size=4, device="cuda"):
    model.eval()
    completions = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="HelpSteer gen"):
        batch = prompts[start : start + batch_size]
        chats = [[{"role": "user", "content": p}] for p in batch]
        inputs = tokenizer.apply_chat_template(
            chats,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        gen = out[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        completions.extend(decoded)
    return completions


def _parse_score(text):
    if text is None:
        return None
    m = _SCORE_RE.search(text.strip())
    return int(m.group(0)) if m else None


async def _judge_one(client, judge_model, prompt, completion, semaphore, max_retries=4):
    user = _JUDGE_USER_TEMPLATE.format(prompt=prompt, completion=completion)
    delay = 1.0
    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=judge_model,
                    messages=[
                        {"role": "system", "content": _JUDGE_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                )
                return _parse_score(resp.choices[0].message.content)
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[judge] giving up after {max_retries} attempts: {e}")
                    return None
                await asyncio.sleep(delay)
                delay *= 2
    return None


async def _judge_all(prompts, completions, judge_model, concurrency):
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)

    pbar = tqdm(total=len(prompts), desc="HelpSteer judge")

    async def _wrapped(p, c):
        score = await _judge_one(client, judge_model, p, c, semaphore)
        pbar.update(1)
        return score

    try:
        results = await asyncio.gather(*[
            _wrapped(p, c) for p, c in zip(prompts, completions)
        ])
    finally:
        pbar.close()
    return results


def judge_completions(prompts, completions, judge_model="gpt-5-nano", concurrency=8):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    return asyncio.run(_judge_all(prompts, completions, judge_model, concurrency))


def evaluate(model, tokenizer, prompts, judge_model="gpt-5-nano",
             max_new_tokens=256, batch_size=4, judge_concurrency=8, device="cuda"):
    """
    Returns: {mean_score, n, n_scored, rows: [{prompt, completion, score}, ...]}
    Caller is responsible for any active ablation context during generation.
    """
    completions = generate_completions(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=device,
    )
    scores = judge_completions(prompts, completions,
                               judge_model=judge_model,
                               concurrency=judge_concurrency)

    rows = [
        {"prompt": p, "completion": c, "score": s}
        for p, c, s in zip(prompts, completions, scores)
    ]
    valid = [s for s in scores if s is not None]
    mean_score = sum(valid) / len(valid) if valid else float("nan")
    return {
        "mean_score": mean_score,
        "n": len(prompts),
        "n_scored": len(valid),
        "rows": rows,
    }
