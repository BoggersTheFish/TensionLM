"""
Export TS field-rescorer training signals from formal_eval items.

This does not train a model. It creates supervised records that can later train
a small field rescorer or graph-candidate scorer:

  prompt, category, rule, inferred answers, candidate token ids, graph edge
  count, generated output, prefix/substring labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import formal_eval  # noqa: E402
from ts_bridge.rescore import (  # noqa: E402
    RescoreConfig, build_rule_graph, generate_with_rescore,
    mode_flags, score_output, set_seed,
)
from ts_bridge.smoke_test import load_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--category", action="append", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", choices=["base", "tau", "surface", "both"], default="both")
    ap.add_argument("--max_new", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--surface_beta", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="logs/eval/training_signals.jsonl")
    args = ap.parse_args()

    model, tokenizer, _ = load_model(args.checkpoint, args.device, args.tokenizer)
    items = formal_eval.select_benchmark(args.category, args.limit)
    cfg = RescoreConfig(
        alpha=args.alpha,
        surface_beta=args.surface_beta,
        max_new=args.max_new,
        rep_penalty=1.2,
    )
    use_tau, use_surface = mode_flags(args.mode)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i, item in enumerate(items):
            set_seed(args.seed + i)
            prompt_ids = tokenizer.encode(item["prompt"]).ids
            graph = build_rule_graph(item, prompt_ids, tokenizer)
            trace = generate_with_rescore(
                model, tokenizer, prompt_ids,
                graph_result=graph,
                config=cfg,
                use_tau_bias=use_tau,
                use_surface_rescore=use_surface,
                device=args.device,
            )
            scores = score_output(trace.generated_text, item["accept"], item["reject"])
            f.write(json.dumps({
                "prompt": item["prompt"],
                "category": item["category"],
                "accept": item["accept"],
                "reject": item["reject"],
                "mode": args.mode,
                "graph": graph.to_dict(),
                "output": trace.generated_text,
                "first_token": trace.first_token,
                "first_token_id": trace.first_token_id,
                "first_topk": trace.first_topk,
                **scores,
            }) + "\n")
    print(f"Wrote training-signal JSONL: {out_path} ({len(items)} rows)")


if __name__ == "__main__":
    main()
