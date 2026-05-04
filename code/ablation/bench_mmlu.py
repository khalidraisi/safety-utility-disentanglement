# MMLU subsample benchmark, scored by log-prob of " A"/" B"/" C"/" D"
# continuations after a 0-shot prompt. Default subsample = 5 per subject (285).

from collections import defaultdict
import random

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm


_PROMPT_TEMPLATE = (
    "The following is a multiple choice question. "
    "Respond with only the letter of the correct answer.\n\n"
    "{question}\n"
    "A. {a}\n"
    "B. {b}\n"
    "C. {c}\n"
    "D. {d}\n"
    "Answer:"
)

_LETTERS = ["A", "B", "C", "D"]


def load_mmlu_subsample(per_subject=5, seed=0):
    ds = load_dataset("cais/mmlu", "all", split="test")
    by_subj = defaultdict(list)
    for row in ds:
        by_subj[row["subject"]].append(row)
    rng = random.Random(seed)
    out = []
    for subj, rows in by_subj.items():
        rng.shuffle(rows)
        out.extend(rows[:per_subject])
    return out


def _letter_token_ids(tokenizer):
    # Token id for " A", " B", " C", " D" (leading space) — what the model
    # would emit immediately after "Answer:" with no further whitespace.
    ids = []
    for L in _LETTERS:
        toks = tokenizer.encode(" " + L, add_special_tokens=False)
        ids.append(toks[0])
    return ids


@torch.no_grad()
def evaluate(model, tokenizer, rows, batch_size=8, device="cuda"):
    model.eval()
    letter_ids = _letter_token_ids(tokenizer)

    correct = 0
    n = 0
    by_subject = defaultdict(lambda: [0, 0])  # subject -> [correct, n]
    details = []

    for start in tqdm(range(0, len(rows), batch_size), desc="MMLU"):
        batch = rows[start : start + batch_size]
        prompts = [
            _PROMPT_TEMPLATE.format(
                question=r["question"],
                a=r["choices"][0], b=r["choices"][1],
                c=r["choices"][2], d=r["choices"][3],
            )
            for r in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        out = model(**enc)
        logits = out.logits  # (B, T, V)

        # last non-pad position per row
        attn = enc["attention_mask"]
        last_idx = attn.sum(dim=1) - 1  # (B,)
        last_logits = logits[torch.arange(logits.shape[0]), last_idx]  # (B, V)
        letter_logits = last_logits[:, letter_ids]  # (B, 4)
        log_probs = F.log_softmax(letter_logits, dim=-1)
        pred = log_probs.argmax(dim=-1).cpu().tolist()

        for r, p, lp in zip(batch, pred, log_probs.cpu().tolist()):
            ok = int(p == r["answer"])
            correct += ok
            n += 1
            by_subject[r["subject"]][0] += ok
            by_subject[r["subject"]][1] += 1
            details.append({
                "subject": r["subject"],
                "answer": int(r["answer"]),
                "pred": int(p),
                "correct": bool(ok),
                "log_probs": lp,
            })

    return {
        "accuracy": correct / max(n, 1),
        "n": n,
        "by_subject": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in by_subject.items()},
        "details": details,
    }
