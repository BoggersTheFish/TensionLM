"""
Evaluate a learned field rescorer as a first-token selector.

This uses the trained ranker to choose among graph candidates + base top-k
tokens, then scores that selected first token. It does not yet generate a full
continuation through the LM; it isolates candidate selection quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import formal_eval  # noqa: E402
from ts_bridge.field_rescorer import (  # noqa: E402
    build_rows_for_item, eval_ranker, load_ranker,
)
from ts_bridge.rescore import RescoreConfig, score_output  # noqa: E402
from ts_bridge.smoke_test import load_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--ranker", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--category", action="append", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--negatives_k", type=int, default=12)
    ap.add_argument("--out", default="logs/eval/field_rescorer_generation_eval.json")
    args = ap.parse_args()

    lm, tokenizer, _ = load_model(args.checkpoint, args.device, args.tokenizer)
    ranker, norm = load_ranker(args.ranker)
    items = formal_eval.select_benchmark(args.category, args.limit)
    cfg = RescoreConfig(alpha=args.alpha, max_new=1)

    rows = []
    by_prompt = {}
    for i, item in enumerate(items):
        item_rows = build_rows_for_item(lm, tokenizer, item, i, cfg, args.device, args.negatives_k)
        rows.extend(item_rows)
        by_prompt[i] = item

    rank_metrics = eval_ranker(ranker, norm, rows)
    results = []
    for pick in rank_metrics["picks"]:
        item = by_prompt[pick["prompt_idx"]]
        token_text = pick["token"].replace("Ġ", " ").replace("▁", " ")
        scores = score_output(token_text, item["accept"], item["reject"])
        results.append({
            **pick,
            "accept": item["accept"],
            "reject": item["reject"],
            "selected_text": token_text,
            **scores,
        })

    prefix = sum(r["prefix_correct"] for r in results)
    substring = sum(r["substring_correct"] for r in results)
    token_label = sum(r["label"] for r in results)
    payload = {
        "ranker": args.ranker,
        "categories": args.category,
        "total": len(results),
        "prefix_correct": prefix,
        "substring_correct": substring,
        "token_label_correct": token_label,
        "prefix_accuracy": prefix / len(results) if results else 0.0,
        "substring_accuracy": substring / len(results) if results else 0.0,
        "token_label_accuracy": token_label / len(results) if results else 0.0,
        "rank_metrics": rank_metrics,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(
        f"field-rescorer selection token-label={token_label}/{len(results)} "
        f"prefix={prefix}/{len(results)} substring={substring}/{len(results)}"
    )
    print(f"Wrote eval: {out}")


if __name__ == "__main__":
    main()
