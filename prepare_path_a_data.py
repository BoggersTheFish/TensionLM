"""Prepare GPT-2-tokenized Path A training shards.

This is the data-side bridge for the next raw-model run:

1. export the GPT-2 tokenizer as a `tokenizers` JSON,
2. stream one or more formal/code datasets,
3. write uint16 train/val shards compatible with `train.py`.

The script intentionally keeps source names configurable. Dataset availability
on Hugging Face changes more often than the training loop should, so the launch
scripts pin the intended sources while this tool remains reusable.

Examples:
    # Tiny smoke set from inline examples.
    python prepare_path_a_data.py \
      --source smoke-formal \
      --out_dir data/path-a-smoke \
      --tokenizer data/gpt2-tokenizer/tokenizer.json \
      --max_tokens 50000 --shard_size 10000

    # Stage 2 formal-language target.
    python prepare_path_a_data.py \
      --source hf:hoskinson-center/proof-pile:text:train \
      --out_dir data/path-a-proofpile-gpt2 \
      --tokenizer data/gpt2-tokenizer/tokenizer.json \
      --max_tokens 500000000

    # Stage 3 mixed formal/code target.
    python prepare_path_a_data.py \
      --source hf:open-web-math/open-web-math:text:train \
      --source hf:bigcode/the-stack-smol:content:train \
      --out_dir data/path-a-math-code-gpt2 \
      --tokenizer data/gpt2-tokenizer/tokenizer.json \
      --max_tokens 5000000000
"""

from __future__ import annotations

import argparse
from itertools import cycle
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Iterator

import numpy as np
from tokenizers import Tokenizer


SMOKE_DOCS = [
    "If A implies B and B implies C, then A implies C.",
    "All squares are rectangles. All rectangles are quadrilaterals. Therefore all squares are quadrilaterals.",
    "2 plus 2 equals 4. 7 multiplied by 8 equals 56. The square root of 9 is 3.",
    "In Python, len([1, 2, 3]) returns 3. bool([]) evaluates to False.",
    "A stack is last in first out. A queue is first in first out.",
    "Proof by contradiction assumes the negation and derives a contradiction.",
    "The derivative of x squared is 2x. The integral of 2x dx is x squared.",
    "A loop invariant helps prove correctness because it remains true before and after each iteration.",
]


def parse_hf_source(spec: str) -> tuple[str, str | None, str, str]:
    # hf:dataset[:config]:field[:split]
    parts = spec.split(":")
    if len(parts) not in (4, 5) or parts[0] != "hf":
        raise ValueError(
            "HF source must be hf:dataset:field:split or hf:dataset:config:field:split"
        )
    if len(parts) == 4:
        _, dataset, field, split = parts
        return dataset, None, field, split
    _, dataset, config, field, split = parts
    return dataset, config or None, field, split


def stream_source(spec: str) -> Iterator[str]:
    if spec == "smoke-formal":
        while True:
            yield from SMOKE_DOCS
        return

    dataset, config, field, split = parse_hf_source(spec)
    from datasets import load_dataset

    kwargs = {"streaming": True, "split": split}
    if config is None:
        ds = load_dataset(dataset, **kwargs)
    else:
        ds = load_dataset(dataset, config, **kwargs)
    for row in ds:
        text = row.get(field)
        if isinstance(text, str) and text.strip():
            yield text


def interleave_sources(sources: list[str]) -> Iterator[str]:
    streams = [stream_source(src) for src in sources]
    for stream in cycle(streams):
        yield next(stream)


def flush_shard(tokens: list[int], out_dir: Path, split: str, idx: int) -> dict:
    path = out_dir / f"{split}_{idx:04d}.bin"
    np.array(tokens, dtype=np.uint16).tofile(str(path))
    return {"split": split, "path": str(path), "tokens": len(tokens)}


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--source", action="append", required=True,
                    help="Source spec. Use smoke-formal or hf:dataset[:config]:field:split.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tokenizer", default="data/gpt2-tokenizer/tokenizer.json")
    ap.add_argument("--max_tokens", type=int, required=True)
    ap.add_argument("--shard_size", type=int, default=100_000_000)
    ap.add_argument("--val_tokens", type=int, default=None,
                    help="Validation tokens to reserve at the end. Defaults to one shard or 5%%.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok_dst = out / "tokenizer.json"
    if Path(args.tokenizer).resolve() != tok_dst.resolve():
        shutil.copy(args.tokenizer, tok_dst)

    tokenizer = Tokenizer.from_file(str(tok_dst))
    if tokenizer.get_vocab_size() > 65535:
        raise ValueError("uint16 shards require vocab_size <= 65535")

    val_tokens = args.val_tokens
    if val_tokens is None:
        val_tokens = min(args.shard_size, max(args.max_tokens // 20, 1))
    train_budget = max(args.max_tokens - val_tokens, 1)

    sep = tokenizer.encode("\n\n").ids
    stream = interleave_sources(args.source)

    shards: list[dict] = []
    buf: list[int] = []
    total = 0
    train_written = 0
    idx = {"train": 0, "val": 0}

    print(f"Preparing {args.max_tokens:,} tokens -> {out}")
    print(f"Tokenizer vocab: {tokenizer.get_vocab_size()}")
    print(f"Sources: {args.source}")
    print(f"Train budget: {train_budget:,} | Val budget: {val_tokens:,}")

    for doc in stream:
        ids = tokenizer.encode(doc).ids + sep
        buf.extend(ids)

        while len(buf) >= args.shard_size:
            split = "train" if train_written < train_budget else "val"
            chunk = buf[:args.shard_size]
            buf = buf[args.shard_size:]
            meta = flush_shard(chunk, out, split, idx[split])
            shards.append(meta)
            idx[split] += 1
            total += meta["tokens"]
            if split == "train":
                train_written += meta["tokens"]
            print(f"  {Path(meta['path']).name}: {meta['tokens']:,} tokens")
            if total >= args.max_tokens:
                break
        if total >= args.max_tokens:
            break

    if total < args.max_tokens and buf:
        split = "train" if train_written < train_budget else "val"
        meta = flush_shard(buf[: args.max_tokens - total], out, split, idx[split])
        shards.append(meta)
        total += meta["tokens"]
        if split == "train":
            train_written += meta["tokens"]
        print(f"  {Path(meta['path']).name}: {meta['tokens']:,} tokens")

    train_tokens = sum(s["tokens"] for s in shards if s["split"] == "train")
    val_count = sum(1 for s in shards if s["split"] == "val")
    if val_count == 0:
        last_train = next(s for s in reversed(shards) if s["split"] == "train")
        old = Path(last_train["path"])
        new = old.parent / old.name.replace("train_", "val_")
        old.rename(new)
        last_train["path"] = str(new)
        last_train["split"] = "val"
        train_tokens -= last_train["tokens"]

    meta = {
        "dataset": "path-a-mixed",
        "sources": args.source,
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer": str(tok_dst),
        "total_tokens": sum(s["tokens"] for s in shards),
        "train_tokens": sum(s["tokens"] for s in shards if s["split"] == "train"),
        "val_tokens": sum(s["tokens"] for s in shards if s["split"] == "val"),
        "shards": shards,
    }
    meta_path = out / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {meta_path}")
    sys.stdout.flush()
    sys.stderr.flush()
    # Some HF streaming backends can abort during interpreter teardown after a
    # deliberately partial stream. All shard files and metadata are already
    # durable here, so exit directly once success is reached.
    os._exit(0)


if __name__ == "__main__":
    main()
