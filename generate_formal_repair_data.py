"""Generate a dense CPU repair corpus for the existing 117M checkpoint.

This is not a broad pretraining dataset. It is a small TS-style relaxation set:
high-signal formal/code completions plus replay of arithmetic/transitivity
items that the current checkpoint already wins on. The goal is to move the raw
next-token trajectory without rebuilding the full model.

Example:
    python generate_formal_repair_data.py \
      --tokenizer checkpoints/117m-curriculum/tokenizer.json \
      --out_dir data/formal-repair-117m \
      --target_tokens 2000000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
from tokenizers import Tokenizer

import formal_eval


CODE_REPAIR = [
    ("In Python, len([1, 2, 3]) returns", "3"),
    ("In Python, bool([]) evaluates to", "False"),
    ("In Python, range(3) produces 0, 1, and", "2"),
    ("A function with no explicit return in Python returns", "None"),
    ("In Python, {'a': 1}['a'] evaluates to", "1"),
    ("The time complexity of binary search on a sorted list is", "O(log n)"),
    ("A stack removes the most recently added item first, also called", "LIFO"),
    ("A queue removes the earliest added item first, also called", "FIFO"),
    ("A parse error happens before a program successfully", "runs"),
    ("If a loop invariant is true before and after each iteration, it helps prove", "correctness"),
    ("In Python, bool([0]) evaluates to", "True"),
    ("In Python, int('3') returns", "3"),
    ("In Python, list(range(2)) is", "[0, 1]"),
    ("In Python, 'abc'[0] evaluates to", "a"),
    ("In Python, {'x': 4}.get('x') returns", "4"),
    ("A Python list preserves insertion", "order"),
    ("A Python set removes duplicate", "values"),
    ("A Python dictionary maps keys to", "values"),
    ("A syntax error is detected before runtime, during", "parsing"),
    ("A recursive function needs a base case to", "terminate"),
    ("The worst-case time complexity of linear search is", "O(n)"),
    ("The average time complexity of dictionary lookup in Python is", "O(1)"),
    ("A variable assigned inside a Python function is local by", "default"),
    ("The Python keyword used to handle exceptions is", "try"),
    ("The Python keyword used to define a function is", "def"),
]

ARITHMETIC_REPAIR = [
    ("3 plus 5 equals", "8"),
    ("9 minus 4 equals", "5"),
    ("6 multiplied by 7 equals", "42"),
    ("12 divided by 3 equals", "4"),
    ("15 plus 27 equals", "42"),
    ("100 minus 37 equals", "63"),
    ("11 multiplied by 11 equals", "121"),
    ("81 divided by 9 equals", "9"),
    ("The square of 12 is", "144"),
    ("The cube of 3 is", "27"),
    ("Half of 18 is", "9"),
    ("One quarter of 20 is", "5"),
    ("2 to the power of 5 equals", "32"),
    ("The greatest common divisor of 12 and 18 is", "6"),
    ("The least common multiple of 4 and 6 is", "12"),
    ("The remainder when 17 is divided by 5 is", "2"),
]

TRANSITIVITY_REPAIR = [
    ("If A implies B and B implies C then A implies", "C"),
    ("If P implies Q and Q implies R then P implies", "R"),
    ("A is greater than B. B is greater than C. Therefore A is greater than", "C"),
    ("John is taller than Mary. Mary is taller than Sam. Therefore John is taller than", "Sam"),
    ("If X is a subset of Y and Y is a subset of Z then X is a subset of", "Z"),
    ("Alice is older than Ben. Ben is older than Cara. Therefore Alice is older than", "Cara"),
    ("5 is less than 9. 9 is less than 12. Therefore 5 is less than", "12"),
    ("A depends on B. B depends on C. Therefore A depends on", "C"),
    ("x equals y. y equals z. Therefore x equals", "z"),
]


def canonical_eval_items() -> list[tuple[str, str, str]]:
    rows = []
    for item in formal_eval.BENCHMARK:
        answer = item["accept"][0]
        rows.append((item["category"], item["prompt"], answer))
    return rows


def load_excluded_prompts(paths: list[str]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        payload = json.loads(Path(path).read_text())
        items = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
        for item in items:
            if isinstance(item, dict) and "prompt" in item:
                excluded.add(item["prompt"])
    return excluded


def build_pool(*, include_canonical_eval: bool = True,
               excluded_prompts: set[str] | None = None) -> list[tuple[str, str, str, float]]:
    excluded_prompts = excluded_prompts or set()
    pool: list[tuple[str, str, str, float]] = []
    if include_canonical_eval:
        for cat, prompt, answer in canonical_eval_items():
            weight = 1.0
            if cat == "code_reasoning":
                weight = 6.0
            elif cat in {"arithmetic", "transitivity"}:
                weight = 3.0
            elif cat in {"definition", "contradiction", "induction"}:
                weight = 2.0
            pool.append((cat, prompt, answer, weight))
    pool += [("code_reasoning", p, a, 8.0) for p, a in CODE_REPAIR]
    pool += [("arithmetic", p, a, 3.0) for p, a in ARITHMETIC_REPAIR]
    pool += [("transitivity", p, a, 3.0) for p, a in TRANSITIVITY_REPAIR]
    return [row for row in pool if row[1] not in excluded_prompts]


def render_example(category: str, prompt: str, answer: str, rng: random.Random,
                   *, answer_prefix_only: bool = False) -> str:
    # Keep answer-prefix pressure high: the completion starts immediately after
    # the prompt, matching `formal_eval.py` rather than instruction tuning.
    variants = [
        f"{prompt} {answer}.",
        f"{prompt} {answer}\n",
    ]
    if not answer_prefix_only:
        variants.append(f"Question: {prompt}\nAnswer: {answer}.\n")
    if category == "code_reasoning":
        variants.append(f"{prompt} {answer}.")
        variants.append(f"{prompt} {answer}\n")
    return rng.choice(variants)


def write_shard(path: Path, tokens: list[int]) -> dict:
    np.array(tokens, dtype=np.uint16).tofile(str(path))
    split = "val" if path.name.startswith("val_") else "train"
    return {"split": split, "path": str(path), "tokens": len(tokens)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="checkpoints/117m-curriculum/tokenizer.json")
    ap.add_argument("--out_dir", default="data/formal-repair-117m")
    ap.add_argument("--target_tokens", type=int, default=2_000_000)
    ap.add_argument("--shard_tokens", type=int, default=250_000)
    ap.add_argument("--val_tokens", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_prompts_json", action="append", default=[],
                    help="Benchmark JSON whose exact prompts must be excluded from the repair corpus.")
    ap.add_argument("--no_canonical_eval", action="store_true",
                    help="Do not include formal_eval.BENCHMARK prompts in the repair pool.")
    ap.add_argument("--shuffle_answers", action="store_true",
                    help="Negative control: keep prompts/categories but assign shuffled answers.")
    ap.add_argument("--shuffle_within_category", action="store_true",
                    help="Stronger negative control: shuffle answers only within each category, preserving category balance and answer frequency.")
    ap.add_argument("--answer_prefix_only", action="store_true",
                    help="Omit Question/Answer wrapper examples so completions start with the answer.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    sep = tokenizer.encode("\n\n").ids
    excluded_prompts = load_excluded_prompts(args.exclude_prompts_json)
    pool = build_pool(
        include_canonical_eval=not args.no_canonical_eval,
        excluded_prompts=excluded_prompts,
    )
    if not pool:
        raise ValueError("repair pool is empty after exclusions")
    prompts = [(cat, prompt, answer) for cat, prompt, answer, _ in pool]
    if args.shuffle_answers and args.shuffle_within_category:
        raise ValueError("choose only one of --shuffle_answers or --shuffle_within_category")
    if args.shuffle_within_category:
        grouped: dict[str, list[int]] = {}
        for i, (cat, _, _) in enumerate(prompts):
            grouped.setdefault(cat, []).append(i)
        next_prompts = list(prompts)
        for cat, indices in grouped.items():
            answers = [prompts[i][2] for i in indices]
            rng.shuffle(answers)
            if len(answers) > 1 and all(answers[j] == prompts[indices[j]][2] for j in range(len(indices))):
                answers = answers[1:] + answers[:1]
            for answer, idx in zip(answers, indices):
                old_cat, prompt, _ = next_prompts[idx]
                next_prompts[idx] = (old_cat, prompt, answer)
        prompts = next_prompts
    elif args.shuffle_answers:
        answers = [answer for _, _, answer in prompts]
        rng.shuffle(answers)
        prompts = [(cat, prompt, answers[i]) for i, (cat, prompt, _) in enumerate(prompts)]
    weights = [weight for _, _, _, weight in pool]

    train_budget = max(args.target_tokens - args.val_tokens, 1)
    shards = []
    buf: list[int] = []
    total = 0
    train_written = 0
    train_idx = 0
    val_idx = 0

    print(f"Generating formal repair corpus -> {out}")
    print(f"target={args.target_tokens:,} train={train_budget:,} val={args.val_tokens:,}")
    print(f"tokenizer vocab={tokenizer.get_vocab_size()} pool={len(pool)}")

    while total < args.target_tokens:
        category, prompt, answer = rng.choices(prompts, weights=weights, k=1)[0]
        text = render_example(category, prompt, answer, rng,
                              answer_prefix_only=args.answer_prefix_only)
        buf.extend(tokenizer.encode(text).ids + sep)

        while len(buf) >= args.shard_tokens:
            remaining = args.target_tokens - total
            if train_written < train_budget:
                chunk_n = min(args.shard_tokens, remaining, train_budget - train_written)
                path = out / f"train_{train_idx:04d}.bin"
                train_idx += 1
                train_written += chunk_n
            else:
                chunk_n = min(args.shard_tokens, remaining)
                path = out / f"val_{val_idx:04d}.bin"
                val_idx += 1
            chunk = buf[:chunk_n]
            buf = buf[chunk_n:]
            shards.append(write_shard(path, chunk))
            total += chunk_n
            print(f"  {path.name}: {chunk_n:,} tokens")
            if total >= args.target_tokens:
                break

    if total < args.target_tokens and buf:
        split_train = train_written < train_budget
        path = out / (f"train_{train_idx:04d}.bin" if split_train else f"val_{val_idx:04d}.bin")
        n = min(len(buf), args.target_tokens - total)
        shards.append(write_shard(path, buf[:n]))
        total += n
        if split_train:
            train_written += n
        print(f"  {path.name}: {n:,} tokens")

    if not any(s["split"] == "val" for s in shards):
        last = next(s for s in reversed(shards) if s["split"] == "train")
        old = Path(last["path"])
        new = old.parent / old.name.replace("train_", "val_")
        old.rename(new)
        last["path"] = str(new)
        last["split"] = "val"

    meta = {
        "dataset": "formal-repair-117m",
        "description": "Dense answer-prefix repair corpus for CPU top-layer tuning.",
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer": args.tokenizer,
        "total_tokens": sum(s["tokens"] for s in shards),
        "train_tokens": sum(s["tokens"] for s in shards if s["split"] == "train"),
        "val_tokens": sum(s["tokens"] for s in shards if s["split"] == "val"),
        "shards": shards,
        "categories": sorted({cat for cat, _, _, _ in pool}),
        "excluded_prompt_count": len(excluded_prompts),
        "include_canonical_eval": not args.no_canonical_eval,
        "shuffle_answers": args.shuffle_answers,
        "shuffle_within_category": args.shuffle_within_category,
        "answer_prefix_only": args.answer_prefix_only,
    }
    meta_path = out / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
