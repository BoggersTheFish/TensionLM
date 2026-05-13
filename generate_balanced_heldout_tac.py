"""Generate balanced held-out TAC benchmarks and matched controls.

TAC = transitivity + arithmetic + code_reasoning. The current held-out set is
small enough that prompt luck and answer-token frequency can dominate. This
script generates a larger deterministic held-out benchmark plus two diagnostic
control benchmarks:

- global-shuffled: preserves the full answer multiset across all prompts.
- category-shuffled: preserves the answer multiset inside each category.

The controls are not "wrong benchmarks" for model quality. They are receipts
for negative-control repair/eval runs: if a model improves on shuffled controls,
part of the gain is format/domain adaptation rather than answer-specific
learning.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Iterable

import formal_eval


def _case(category: str, prompt: str, accept: list[str], reject: list[str] | None = None) -> dict:
    return {
        "category": category,
        "prompt": prompt,
        "accept": accept,
        "reject": reject or [],
    }


def _num_word(n: int) -> str:
    small = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
        5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
        10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
        14: "fourteen", 15: "fifteen", 16: "sixteen",
        17: "seventeen", 18: "eighteen", 19: "nineteen",
        20: "twenty",
    }
    tens = {
        30: "thirty", 40: "forty", 50: "fifty", 60: "sixty",
        70: "seventy", 80: "eighty", 90: "ninety",
    }
    if n in small:
        return small[n]
    if n in tens:
        return tens[n]
    if 20 < n < 100:
        stem = "twenty" if n < 30 else tens[(n // 10) * 10]
        return f"{stem}-{small[n % 10]}"
    return str(n)


def _answer(val: int) -> list[str]:
    return [str(val), _num_word(val)]


def arithmetic_items(limit: int) -> list[dict]:
    rows: list[dict] = []
    for a in range(4, 80, 7):
        b = (a * 3 + 5) % 17 + 2
        rows.append(_case("arithmetic", f"{a} plus {b} equals", _answer(a + b)))
        rows.append(_case("arithmetic", f"{a + b} minus {b} equals", _answer(a)))
        rows.append(_case("arithmetic", f"{b} multiplied by {a % 9 + 2} equals", _answer(b * (a % 9 + 2))))
        divisor = a % 8 + 2
        quotient = b % 9 + 2
        rows.append(_case("arithmetic", f"{divisor * quotient} divided by {divisor} equals", _answer(quotient)))
        rows.append(_case("arithmetic", f"The square of {divisor} is", _answer(divisor * divisor)))
        rows.append(_case("arithmetic", f"The remainder when {a + b} is divided by {divisor} is", _answer((a + b) % divisor)))
    return rows[:limit]


def transitivity_items(limit: int) -> list[dict]:
    triples = [
        ("L", "M", "N"), ("R", "S", "T"), ("alpha", "beta", "gamma"),
        ("red", "blue", "green"), ("one", "two", "three"),
        ("A", "B", "C"), ("delta", "epsilon", "zeta"),
        ("north", "center", "south"), ("first", "second", "third"),
        ("oak", "pine", "cedar"), ("p", "q", "r"), ("x", "y", "z"),
    ]
    templates = [
        ("If {a} leads to {b} and {b} leads to {c}, then {a} leads to", "{c}", "{b}"),
        ("{a} is heavier than {b}. {b} is heavier than {c}. Therefore {a} is heavier than", "{c}", "{b}"),
        ("When {a} entails {b} and {b} entails {c}, {a} entails", "{c}", "{b}"),
        ("Box {a} contains box {b}. Box {b} contains box {c}. Box {a} contains box", "{c}", "{b}"),
        ("A path goes from city {a} to city {b}. It then goes from city {b} to city {c}. Starting at city {a} reaches city", "{c}", "{b}"),
        ("If service {a} requires service {b} and service {b} requires service {c}, service {a} requires service", "{c}", "{b}"),
        ("The {a} file imports the {b} file. The {b} file imports the {c} file. The {a} file indirectly imports the", "{c}", "{b}"),
        ("June is earlier than July. July is earlier than August. June is earlier than", "August", "July"),
        ("Token {a} maps to token {b}. Token {b} maps to token {c}. Token {a} maps eventually to token", "{c}", "{b}"),
        ("{a} equals {b}. {b} equals {c}. Therefore {a} equals", "{c}", "{b}"),
    ]
    rows: list[dict] = []
    for triple in triples:
        a, b, c = triple
        for prompt, accept, reject in templates:
            rows.append(_case(
                "transitivity",
                prompt.format(a=a, b=b, c=c),
                [accept.format(a=a, b=b, c=c)],
                [reject.format(a=a, b=b, c=c)],
            ))
    return rows[:limit]


def code_items(limit: int) -> list[dict]:
    rows = [
        _case("code_reasoning", "In Python, len((1, 2, 3, 4)) returns", ["4", "four"]),
        _case("code_reasoning", "In Python, bool({}) evaluates to", ["False", "false"], ["true"]),
        _case("code_reasoning", "In Python, list(range(4)) ends with", ["3", "three"], ["4", "four"]),
        _case("code_reasoning", "In Python, {'b': 2}['b'] evaluates to", ["2", "two"]),
        _case("code_reasoning", "A first-in first-out data structure is a", ["queue"], ["stack"]),
        _case("code_reasoning", "A last-in first-out data structure is a", ["stack"], ["queue"]),
        _case("code_reasoning", "A Python function begins with the keyword", ["def"]),
        _case("code_reasoning", "The operator used for equality comparison in Python is", ["==", "equals equals"], ["="]),
        _case("code_reasoning", "A loop that never reaches a stopping condition is called an", ["infinite loop", "infinite"]),
        _case("code_reasoning", "The time complexity of checking every item in a list once is", ["O(n)", "linear"], ["O(log n)", "logarithmic"]),
        _case("code_reasoning", "In Python, 'hello'.upper() returns", ["HELLO", "hello uppercase"]),
        _case("code_reasoning", "In Python, 'abc'[1] evaluates to", ["b"]),
        _case("code_reasoning", "In Python, [1, 2] + [3] evaluates to", ["[1, 2, 3]", "1, 2, 3"]),
        _case("code_reasoning", "In Python, 5 // 2 evaluates to", ["2", "two"]),
        _case("code_reasoning", "In Python, 5 % 2 evaluates to", ["1", "one"]),
        _case("code_reasoning", "A Python set is commonly used to remove duplicate", ["values", "items"]),
        _case("code_reasoning", "A dictionary lookup uses a key to retrieve a", ["value"]),
        _case("code_reasoning", "The keyword that exits a Python loop early is", ["break"]),
        _case("code_reasoning", "The keyword that skips to the next loop iteration is", ["continue"]),
        _case("code_reasoning", "The Python value representing nothing is", ["None", "none"]),
        _case("code_reasoning", "A parse error is detected before the program", ["runs", "executes"]),
        _case("code_reasoning", "A base case prevents recursive functions from recursing", ["forever", "infinitely"]),
        _case("code_reasoning", "Binary search requires the input list to be", ["sorted"]),
        _case("code_reasoning", "Appending to the end of a Python list uses the method", ["append"]),
        _case("code_reasoning", "The Python operator for exponentiation is", ["**", "star star"]),
        _case("code_reasoning", "A variable inside a function is local by", ["default"]),
        _case("code_reasoning", "The Python statement used to import a module is", ["import"]),
        _case("code_reasoning", "A try block handles errors with an except", ["block", "clause"]),
        _case("code_reasoning", "A hash table stores key value", ["pairs"]),
        _case("code_reasoning", "The worst-case time complexity of linear search is", ["O(n)", "linear"]),
    ]
    for n in range(2, 20):
        rows.append(_case("code_reasoning", f"In Python, len(list(range({n}))) returns", [str(n), _num_word(n)]))
        rows.append(_case("code_reasoning", f"In Python, list(range({n + 1})) ends with", [str(n), _num_word(n)], [str(n + 1)]))
    return rows[:limit]


def load_excluded_prompts(paths: Iterable[str], *, exclude_builtin: bool) -> set[str]:
    excluded = {item["prompt"] for item in formal_eval.BENCHMARK} if exclude_builtin else set()
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        items = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
        for item in items:
            if isinstance(item, dict) and "prompt" in item:
                excluded.add(item["prompt"])
    return excluded


def dedupe_and_filter(items: list[dict], excluded: set[str]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in items:
        prompt = item["prompt"]
        if prompt in excluded or prompt in seen:
            continue
        seen.add(prompt)
        out.append(item)
    return out


def rotate(values: list[list[str]], rng: random.Random) -> list[list[str]]:
    if len(values) < 2:
        return values
    idx = list(range(len(values)))
    # Keep trying until no answer value remains in place. Duplicate answer
    # values make this stricter than index derangement.
    for _ in range(100):
        rng.shuffle(idx)
        if all(i != j and values[i] != values[j] for i, j in enumerate(idx)):
            return [values[j] for j in idx]
    shifted = values[1:] + values[:1]
    return shifted if any(a != b for a, b in zip(values, shifted)) else values


def make_control(items: list[dict], *, mode: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = [dict(item) for item in items]

    def reduce_same_accepts(indices: list[int]) -> None:
        changed = True
        while changed:
            changed = False
            same = [i for i in indices if out[i]["accept"] == out[i]["original_accept"]]
            for i in same:
                for j in indices:
                    if i == j:
                        continue
                    ai, aj = out[i]["accept"], out[j]["accept"]
                    if aj != out[i]["original_accept"] and ai != out[j]["original_accept"]:
                        out[i]["accept"], out[j]["accept"] = aj, ai
                        changed = True
                        break
                if changed:
                    break

    if mode == "global":
        shuffled = rotate([item["accept"] for item in out], rng)
        for item, accept in zip(out, shuffled):
            original = item["accept"]
            item["original_accept"] = original
            item["accept"] = accept
            item["reject"] = sorted(set(item.get("reject", []) + original))
            item["control"] = "global_answer_frequency_matched"
        reduce_same_accepts(list(range(len(out))))
        return out

    grouped: dict[str, list[int]] = defaultdict(list)
    for i, item in enumerate(out):
        grouped[item["category"]].append(i)
    for cat, indices in grouped.items():
        shuffled = rotate([out[i]["accept"] for i in indices], rng)
        for i, accept in zip(indices, shuffled):
            original = out[i]["accept"]
            out[i]["original_accept"] = original
            out[i]["accept"] = accept
            out[i]["reject"] = sorted(set(out[i].get("reject", []) + original))
            out[i]["control"] = "category_answer_frequency_matched"
        reduce_same_accepts(indices)
    return out


def write_benchmark(path: Path, *, name: str, items: list[dict], description: str, seed: int) -> None:
    counts = Counter(item["category"] for item in items)
    payload = {
        "name": name,
        "description": description,
        "seed": seed,
        "items": items,
        "counts": dict(sorted(counts.items())),
        "total": len(items),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {path} ({len(items)} items; {dict(sorted(counts.items()))})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ts_bridge/heldout_formal_tac_v2.json")
    ap.add_argument("--per_category", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1776)
    ap.add_argument("--exclude_json", action="append", default=["ts_bridge/heldout_formal_tac.json"])
    ap.add_argument("--include_builtin_overlap", action="store_true")
    ap.add_argument("--control_global_out", default="ts_bridge/heldout_formal_tac_v2_control_global.json")
    ap.add_argument("--control_category_out", default="ts_bridge/heldout_formal_tac_v2_control_category.json")
    args = ap.parse_args()

    excluded = load_excluded_prompts(args.exclude_json, exclude_builtin=not args.include_builtin_overlap)
    candidates = {
        "transitivity": dedupe_and_filter(transitivity_items(args.per_category * 4), excluded),
        "arithmetic": dedupe_and_filter(arithmetic_items(args.per_category * 4), excluded),
        "code_reasoning": dedupe_and_filter(code_items(args.per_category * 4), excluded),
    }
    items: list[dict] = []
    for cat, rows in candidates.items():
        if len(rows) < args.per_category:
            raise ValueError(f"not enough {cat} rows after exclusions: {len(rows)} < {args.per_category}")
        items.extend(rows[: args.per_category])

    rng = random.Random(args.seed)
    rng.shuffle(items)

    out = Path(args.out)
    write_benchmark(
        out,
        name="heldout_formal_tac_v2",
        items=items,
        description="Balanced held-out TAC benchmark generated deterministically with built-in/previous held-out prompt exclusion.",
        seed=args.seed,
    )
    write_benchmark(
        Path(args.control_global_out),
        name="heldout_formal_tac_v2_control_global",
        items=make_control(items, mode="global", seed=args.seed + 1),
        description="Global answer-frequency-matched control. Diagnostic only.",
        seed=args.seed + 1,
    )
    write_benchmark(
        Path(args.control_category_out),
        name="heldout_formal_tac_v2_control_category",
        items=make_control(items, mode="category", seed=args.seed + 2),
        description="Category answer-frequency-matched control. Diagnostic only.",
        seed=args.seed + 2,
    )


if __name__ == "__main__":
    main()
