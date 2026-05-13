"""
Run formal_eval ablations for TS-assisted generation.

Modes:
  base    : standard generation
  tau     : graph -> tau-bias only
  surface : graph-supported first-token rescoring only
  both    : tau-bias + graph-supported first-token rescoring
  sequence: graph-supported answer-sequence selection
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import formal_eval  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--category", action="append", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_new", type=int, default=12)
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--surface_beta", type=float, default=4.0)
    ap.add_argument("--out", default="logs/eval/ablation_summary.json")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": args.checkpoint,
        "tokenizer": args.tokenizer,
        "category": args.category,
        "limit": args.limit,
        "max_new": args.max_new,
        "temp": args.temp,
        "top_p": args.top_p,
        "seed": args.seed,
        "alpha": args.alpha,
        "surface_beta": args.surface_beta,
        "modes": {},
    }

    for mode in ["base", "tau", "surface", "both", "sequence"]:
        mode_out = out_path.with_name(f"{out_path.stem}_{mode}.json")
        correct, total, results = formal_eval.run_eval(
            args.checkpoint,
            max_new=args.max_new,
            temp=args.temp,
            top_p=args.top_p,
            device=args.device,
            tokenizer_path=args.tokenizer,
            categories_filter=args.category,
            limit=args.limit,
            seed=args.seed,
            json_out=str(mode_out),
            ts_mode=mode,
            alpha=args.alpha,
            surface_beta=args.surface_beta,
        )
        prefix_correct = sum(r["prefix_correct"] for r in results)
        summary["modes"][mode] = {
            "json": str(mode_out),
            "substring_correct": correct,
            "prefix_correct": prefix_correct,
            "total": total,
            "substring_accuracy": correct / total if total else 0.0,
            "prefix_accuracy": prefix_correct / total if total else 0.0,
        }

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote ablation summary: {out_path}")
    for mode, data in summary["modes"].items():
        print(
            f"{mode:7s} prefix={data['prefix_correct']}/{data['total']} "
            f"substring={data['substring_correct']}/{data['total']}"
        )


if __name__ == "__main__":
    main()
