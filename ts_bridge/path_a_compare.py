"""Compare raw model-level Path A eval JSON files.

Path A means TensionLM must beat GPT-2 as a model, not by relying on
external TS answer extraction. This script consumes `formal_eval.py` JSON
outputs and writes a compact comparison artifact.

Examples:
    python -m ts_bridge.path_a_compare \
      --run gpt2=logs/eval/gpt2_pathA_raw_tac_seed42.json \
      --run tension117m=logs/eval/tension117m_pathA_raw_tac_seed42.json \
      --out logs/eval/pathA_raw_compare_seed42.json

    python -m ts_bridge.path_a_compare \
      --run_glob 'gpt2=logs/eval/gpt2_pathA_raw_tac_seed*.json' \
      --run_glob 'tension117m=logs/eval/tension117m_pathA_raw_tac_seed*.json' \
      --out logs/eval/pathA_raw_compare_multiseed.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from glob import glob
import json
from pathlib import Path
from typing import Any


def load_eval(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if "results" not in payload:
        raise ValueError(f"{path} is not a formal_eval JSON file")
    return payload


def summarize_eval(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload["results"]
    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "prefix_correct": 0, "substring_correct": 0}
    )
    for row in results:
        cat = row["category"]
        by_category[cat]["total"] += 1
        by_category[cat]["prefix_correct"] += int(bool(row.get("prefix_correct")))
        by_category[cat]["substring_correct"] += int(bool(row.get("substring_correct")))

    total = len(results)
    prefix_correct = sum(int(bool(row.get("prefix_correct"))) for row in results)
    substring_correct = sum(int(bool(row.get("substring_correct"))) for row in results)
    categories = {
        cat: {
            **vals,
            "prefix_accuracy": vals["prefix_correct"] / vals["total"],
            "substring_accuracy": vals["substring_correct"] / vals["total"],
        }
        for cat, vals in sorted(by_category.items())
    }
    return {
        "checkpoint": payload.get("checkpoint"),
        "hf_model": payload.get("hf_model"),
        "seed": payload.get("seed"),
        "max_new": payload.get("max_new"),
        "temp": payload.get("temp"),
        "top_p": payload.get("top_p"),
        "ts_mode": payload.get("ts_mode"),
        "total": total,
        "prefix_correct": prefix_correct,
        "substring_correct": substring_correct,
        "prefix_accuracy": prefix_correct / total if total else 0.0,
        "substring_accuracy": substring_correct / total if total else 0.0,
        "categories": categories,
    }


def summarize_many(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        raise ValueError("no payloads to summarize")
    per_run = [summarize_eval(payload) for payload in payloads]
    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "prefix_correct": 0, "substring_correct": 0}
    )
    for summary in per_run:
        for cat, vals in summary["categories"].items():
            by_category[cat]["total"] += vals["total"]
            by_category[cat]["prefix_correct"] += vals["prefix_correct"]
            by_category[cat]["substring_correct"] += vals["substring_correct"]

    total = sum(item["total"] for item in per_run)
    prefix_correct = sum(item["prefix_correct"] for item in per_run)
    substring_correct = sum(item["substring_correct"] for item in per_run)
    categories = {
        cat: {
            **vals,
            "prefix_accuracy": vals["prefix_correct"] / vals["total"],
            "substring_accuracy": vals["substring_correct"] / vals["total"],
        }
        for cat, vals in sorted(by_category.items())
    }
    return {
        "checkpoint": per_run[0].get("checkpoint"),
        "hf_model": per_run[0].get("hf_model"),
        "seeds": [item.get("seed") for item in per_run],
        "runs": len(per_run),
        "max_new": per_run[0].get("max_new"),
        "temp": per_run[0].get("temp"),
        "top_p": per_run[0].get("top_p"),
        "ts_mode": per_run[0].get("ts_mode"),
        "total": total,
        "prefix_correct": prefix_correct,
        "substring_correct": substring_correct,
        "prefix_accuracy": prefix_correct / total if total else 0.0,
        "substring_accuracy": substring_correct / total if total else 0.0,
        "categories": categories,
        "per_run": per_run,
    }


def compare(primary_name: str, baseline_name: str, summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    primary = summaries[primary_name]
    baseline = summaries[baseline_name]
    cats = sorted(set(primary["categories"]) | set(baseline["categories"]))
    by_category = {}
    for cat in cats:
        p = primary["categories"].get(cat)
        b = baseline["categories"].get(cat)
        if not p or not b:
            continue
        by_category[cat] = {
            "prefix_delta": p["prefix_accuracy"] - b["prefix_accuracy"],
            "substring_delta": p["substring_accuracy"] - b["substring_accuracy"],
            "primary_prefix": p["prefix_correct"],
            "baseline_prefix": b["prefix_correct"],
            "total": p["total"],
        }
    return {
        "primary": primary_name,
        "baseline": baseline_name,
        "prefix_delta": primary["prefix_accuracy"] - baseline["prefix_accuracy"],
        "substring_delta": primary["substring_accuracy"] - baseline["substring_accuracy"],
        "primary_prefix": primary["prefix_correct"],
        "baseline_prefix": baseline["prefix_correct"],
        "total": primary["total"],
        "win": primary["prefix_accuracy"] > baseline["prefix_accuracy"],
        "by_category": by_category,
    }


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("run name must not be empty")
    return name, Path(path)


def parse_run_glob(value: str) -> tuple[str, list[Path]]:
    name, pattern = parse_run(value)
    matches = [Path(p) for p in sorted(glob(str(pattern)))]
    if not matches:
        raise argparse.ArgumentTypeError(f"glob matched no files: {pattern}")
    return name, matches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], type=parse_run)
    ap.add_argument("--run_glob", action="append", default=[], type=parse_run_glob)
    ap.add_argument("--primary", default="tension117m")
    ap.add_argument("--baseline", default="gpt2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    grouped: dict[str, list[Path]] = defaultdict(list)
    for name, path in args.run:
        grouped[name].append(path)
    for name, paths in args.run_glob:
        grouped[name].extend(paths)
    if not grouped:
        raise SystemExit("provide at least one --run or --run_glob")

    summaries = {
        name: summarize_many([load_eval(path) for path in paths])
        for name, paths in sorted(grouped.items())
    }
    if args.primary not in summaries:
        raise SystemExit(f"missing primary run {args.primary!r}")
    if args.baseline not in summaries:
        raise SystemExit(f"missing baseline run {args.baseline!r}")

    payload = {
        "comparison": compare(args.primary, args.baseline, summaries),
        "runs": summaries,
    }

    comp = payload["comparison"]
    print(
        f"{args.primary} vs {args.baseline}: "
        f"prefix {comp['primary_prefix']}/{comp['total']} vs "
        f"{comp['baseline_prefix']}/{comp['total']} "
        f"(delta {comp['prefix_delta']:+.1%})"
    )
    for cat, row in comp["by_category"].items():
        print(
            f"  {cat:14s}: {row['primary_prefix']}/{row['total']} vs "
            f"{row['baseline_prefix']}/{row['total']} "
            f"(delta {row['prefix_delta']:+.1%})"
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
