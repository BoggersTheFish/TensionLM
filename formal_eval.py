"""
formal_eval.py — formal reasoning benchmark for TensionLM

Tests the model on syllogisms, equations, and proof-style prompts.
Scores each answer and reports accuracy vs a keyword-match ground truth.

Usage:
    python formal_eval.py --checkpoint checkpoints/stage3_math_117m/ckpt_0014000.pt
    python formal_eval.py --checkpoint checkpoints/stage3_math_117m/ckpt_0014000.pt --temp 0.3
    python formal_eval.py --list_categories
    python formal_eval.py --checkpoint ... --category code_reasoning --limit 10
    # Override tokenizer (when the checkpoint's tok_path is stale) and run on CUDA:
    python formal_eval.py --checkpoint checkpoints/117m-curriculum/pytorch_model.pt \
        --tokenizer checkpoints/117m-curriculum/tokenizer.json --device cuda
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, __file__ and __import__("os").path.dirname(__file__) or ".")


LEGACY_BENCHMARK = [
    # ── Syllogisms ──────────────────────────────────────────────────────────────
    {
        "category": "syllogism",
        "prompt": "All men are mortal. Socrates is a man. Therefore Socrates is",
        "accept": ["mortal"],
        "reject": [],
    },
    {
        "category": "syllogism",
        "prompt": "All cats are animals. Whiskers is a cat. Therefore Whiskers is",
        "accept": ["animal", "animals"],
        "reject": [],
    },
    {
        "category": "syllogism",
        "prompt": "No fish are mammals. All sharks are fish. Therefore no sharks are",
        "accept": ["mammal", "mammals"],
        "reject": [],
    },
    {
        "category": "syllogism",
        "prompt": "All squares are rectangles. All rectangles are quadrilaterals. Therefore all squares are",
        "accept": ["quadrilateral", "quadrilaterals"],
        "reject": [],
    },
    {
        "category": "syllogism",
        "prompt": "If it is raining then the ground is wet. It is raining. Therefore the ground is",
        "accept": ["wet"],
        "reject": ["dry"],
    },
    {
        "category": "syllogism",
        "prompt": "Either the butler did it or the gardener did it. The butler did not do it. Therefore the gardener",
        "accept": ["did it", "did", "guilty"],
        "reject": [],
    },
    # ── Transitivity ────────────────────────────────────────────────────────────
    {
        "category": "transitivity",
        "prompt": "A is greater than B. B is greater than C. Therefore A is greater than",
        "accept": ["c", "C"],
        "reject": ["b", "B"],
    },
    {
        "category": "transitivity",
        "prompt": "If P implies Q and Q implies R then P implies",
        "accept": ["r", "R"],
        "reject": [],
    },
    {
        "category": "transitivity",
        "prompt": "John is taller than Mary. Mary is taller than Sam. Therefore John is taller than",
        "accept": ["sam", "Sam"],
        "reject": [],
    },
    # ── Basic arithmetic ─────────────────────────────────────────────────────────
    {
        "category": "arithmetic",
        "prompt": "2 plus 2 equals",
        "accept": ["4", "four"],
        "reject": [],
    },
    {
        "category": "arithmetic",
        "prompt": "The square root of 9 is",
        "accept": ["3", "three"],
        "reject": [],
    },
    {
        "category": "arithmetic",
        "prompt": "7 multiplied by 8 equals",
        "accept": ["56", "fifty-six", "fifty six"],
        "reject": [],
    },
    {
        "category": "arithmetic",
        "prompt": "The sum of the angles in a triangle is",
        "accept": ["180", "one hundred and eighty", "π", "pi"],
        "reject": [],
    },
    # ── Calculus ─────────────────────────────────────────────────────────────────
    {
        "category": "calculus",
        "prompt": "The derivative of x squared is",
        "accept": ["2x", "2 x", "2*x"],
        "reject": [],
    },
    {
        "category": "calculus",
        "prompt": "The derivative of a constant is",
        "accept": ["0", "zero"],
        "reject": [],
    },
    {
        "category": "calculus",
        "prompt": "The integral of 1 dx is",
        "accept": ["x", "x +", "x+"],
        "reject": [],
    },
    {
        "category": "calculus",
        "prompt": "The limit as x approaches infinity of 1 over x is",
        "accept": ["0", "zero"],
        "reject": [],
    },
    # ── Algebra ──────────────────────────────────────────────────────────────────
    {
        "category": "algebra",
        "prompt": "If x plus 3 equals 7 then x equals",
        "accept": ["4", "four"],
        "reject": [],
    },
    {
        "category": "algebra",
        "prompt": "The solutions to x squared minus 4 equals 0 are x equals 2 and x equals",
        "accept": ["-2", "negative 2", "minus 2"],
        "reject": [],
    },
    {
        "category": "algebra",
        "prompt": "If 2x equals 10 then x equals",
        "accept": ["5", "five"],
        "reject": [],
    },
    # ── Definitions ──────────────────────────────────────────────────────────────
    {
        "category": "definition",
        "prompt": "A prime number is a number greater than 1 that has no divisors other than 1 and",
        "accept": ["itself", "itself."],
        "reject": [],
    },
    {
        "category": "definition",
        "prompt": "The Pythagorean theorem states that in a right triangle a squared plus b squared equals",
        "accept": ["c squared", "c^2", "c²"],
        "reject": [],
    },
    {
        "category": "definition",
        "prompt": "An even number is divisible by",
        "accept": ["2", "two"],
        "reject": [],
    },
]


def _case(category: str, prompt: str, accept: list[str], reject: list[str] | None = None) -> dict:
    return {
        "category": category,
        "prompt": prompt,
        "accept": accept,
        "reject": reject or [],
    }


def _expanded_cases() -> list[dict]:
    """Deterministic 100+ item formal-reasoning benchmark extension."""
    cases: list[dict] = []

    cases += [
        _case("syllogism", "All birds are animals. Robins are birds. Therefore robins are", ["animals", "animal"]),
        _case("syllogism", "All poets are writers. Maya is a poet. Therefore Maya is a", ["writer", "writers"]),
        _case("syllogism", "All copper conducts electricity. This wire is copper. Therefore this wire", ["conducts", "conduct electricity"]),
        _case("syllogism", "No reptiles are warm-blooded. All snakes are reptiles. Therefore no snakes are", ["warm-blooded", "warm blooded"], ["cold"]),
        _case("syllogism", "All primes greater than two are odd. Eleven is a prime greater than two. Therefore eleven is", ["odd"]),
        _case("syllogism", "All squares have four equal sides. This shape is a square. Therefore this shape has", ["four equal sides", "4 equal sides"]),
        _case("syllogism", "If a number is divisible by 10 then it is divisible by 5. 40 is divisible by 10. Therefore 40 is divisible by", ["5", "five"]),
        _case("syllogism", "If a program compiles then its syntax is valid. This program compiles. Therefore its syntax is", ["valid"]),
        _case("syllogism", "Every differentiable function is continuous. f is differentiable. Therefore f is", ["continuous"]),
        _case("syllogism", "All rectangles have four right angles. All squares are rectangles. Therefore all squares have", ["four right angles", "4 right angles"]),
    ]

    cases += [
        _case("transitivity", "If A implies B and B implies C and C implies D then A implies", ["d"]),
        _case("transitivity", "If X is a subset of Y and Y is a subset of Z then X is a subset of", ["z"]),
        _case("transitivity", "Alice is older than Ben. Ben is older than Cara. Therefore Alice is older than", ["cara"]),
        _case("transitivity", "5 is less than 9. 9 is less than 12. Therefore 5 is less than", ["12", "twelve"]),
        _case("transitivity", "Line a is parallel to line b. Line b is parallel to line c. Therefore line a is parallel to", ["c"]),
        _case("transitivity", "If p causes q and q causes r, then p indirectly causes", ["r"]),
        _case("transitivity", "A depends on B. B depends on C. Therefore A depends on", ["c"]),
        _case("transitivity", "Node one points to node two. Node two points to node three. The path from node one reaches node", ["three", "3"]),
        _case("transitivity", "If every group is a monoid and every monoid is a semigroup, then every group is a", ["semigroup"]),
        _case("transitivity", "x equals y. y equals z. Therefore x equals", ["z"]),
    ]

    arithmetic_specs = [
        ("3 plus 5 equals", ["8", "eight"]),
        ("9 minus 4 equals", ["5", "five"]),
        ("6 multiplied by 7 equals", ["42", "forty-two", "forty two"]),
        ("12 divided by 3 equals", ["4", "four"]),
        ("15 plus 27 equals", ["42", "forty-two", "forty two"]),
        ("100 minus 37 equals", ["63", "sixty-three", "sixty three"]),
        ("11 multiplied by 11 equals", ["121", "one hundred twenty one"]),
        ("81 divided by 9 equals", ["9", "nine"]),
        ("The square of 12 is", ["144", "one hundred forty four"]),
        ("The cube of 3 is", ["27", "twenty-seven", "twenty seven"]),
        ("Half of 18 is", ["9", "nine"]),
        ("One quarter of 20 is", ["5", "five"]),
        ("2 to the power of 5 equals", ["32", "thirty-two", "thirty two"]),
        ("The greatest common divisor of 12 and 18 is", ["6", "six"]),
        ("The least common multiple of 4 and 6 is", ["12", "twelve"]),
        ("The remainder when 17 is divided by 5 is", ["2", "two"]),
    ]
    cases += [_case("arithmetic", prompt, accept) for prompt, accept in arithmetic_specs]

    cases += [
        _case("algebra", "If x minus 5 equals 2 then x equals", ["7", "seven"]),
        _case("algebra", "If x divided by 4 equals 3 then x equals", ["12", "twelve"]),
        _case("algebra", "If 3x equals 21 then x equals", ["7", "seven"]),
        _case("algebra", "If x plus x equals 10 then x equals", ["5", "five"]),
        _case("algebra", "If 5x minus 5 equals 20 then x equals", ["5", "five"]),
        _case("algebra", "Solving x plus 2 equals 9 gives x equals", ["7", "seven"]),
        _case("algebra", "The roots of x squared minus 9 equals 0 are 3 and", ["-3", "negative 3", "minus 3"]),
        _case("algebra", "Expanding (x plus 1)(x plus 1) gives x squared plus 2x plus", ["1", "one"]),
        _case("algebra", "Factoring x squared minus 16 gives (x minus 4)(x plus", ["4", "four"]),
        _case("algebra", "If y equals 2x and x equals 6 then y equals", ["12", "twelve"]),
        _case("algebra", "If a plus b equals 10 and a equals 4 then b equals", ["6", "six"]),
        _case("algebra", "The slope of y equals 3x plus 2 is", ["3", "three"]),
    ]

    cases += [
        _case("calculus", "The derivative of x cubed is", ["3x^2", "3 x^2", "3x squared", "3 x squared"]),
        _case("calculus", "The derivative of sin x is", ["cos x", "cosine x"]),
        _case("calculus", "The derivative of cos x is", ["-sin x", "negative sin x", "minus sin x"]),
        _case("calculus", "The derivative of e to the x is", ["e^x", "e to the x", "exp"]),
        _case("calculus", "The integral of x dx is", ["x^2/2", "x squared over 2", "one half x squared"]),
        _case("calculus", "The integral of 2x dx is", ["x^2", "x squared"]),
        _case("calculus", "The limit as x approaches 0 of sin x over x is", ["1", "one"]),
        _case("calculus", "A local maximum has first derivative equal to", ["0", "zero"]),
        _case("calculus", "The second derivative of x squared is", ["2", "two"]),
        _case("calculus", "The derivative of ln x is", ["1/x", "one over x"]),
    ]

    cases += [
        _case("definition", "A group is a set with an operation satisfying closure, associativity, identity, and", ["inverse", "inverses"]),
        _case("definition", "A vector space must be closed under vector addition and scalar", ["multiplication"]),
        _case("definition", "A bijection is a function that is both injective and", ["surjective"]),
        _case("definition", "A tautology is a statement that is always", ["true"]),
        _case("definition", "A contradiction is a statement that is always", ["false"]),
        _case("definition", "A rational number can be written as a ratio of two", ["integers"]),
        _case("definition", "A continuous function has no jumps or", ["breaks", "discontinuities"]),
        _case("definition", "A tree in graph theory is a connected graph with no", ["cycles", "cycle"]),
        _case("definition", "A byte contains", ["8", "eight"]),
        _case("definition", "A Python dictionary stores key value", ["pairs", "pair"]),
    ]

    cases += [
        _case("linear_algebra", "The dot product of orthogonal vectors is", ["0", "zero"]),
        _case("linear_algebra", "The identity matrix multiplied by a vector returns the", ["same vector", "vector"]),
        _case("linear_algebra", "A square matrix with determinant zero is", ["singular"]),
        _case("linear_algebra", "The transpose of a row vector is a", ["column vector"]),
        _case("linear_algebra", "If Ax equals b has exactly one solution then A is", ["invertible", "nonsingular"]),
        _case("linear_algebra", "Eigenvectors of a matrix are scaled by their", ["eigenvalues", "eigenvalue"]),
        _case("linear_algebra", "The rank of a matrix is the dimension of its column", ["space"]),
        _case("linear_algebra", "The inverse of a product AB is B inverse times", ["A inverse", "A^-1", "A^{-1}"]),
        _case("linear_algebra", "A matrix with orthonormal columns satisfies Q transpose Q equals", ["I", "identity"]),
        _case("linear_algebra", "The null space contains vectors x such that Ax equals", ["0", "zero"]),
    ]

    cases += [
        _case("contradiction", "Assume n is even and n is odd. This is a", ["contradiction"]),
        _case("contradiction", "If assuming not P leads to a contradiction, then P is", ["true"]),
        _case("contradiction", "Proof by contradiction starts by assuming the negation and deriving", ["contradiction", "false"]),
        _case("contradiction", "A statement cannot be both true and false at the same time by the law of", ["noncontradiction", "non-contradiction"]),
        _case("contradiction", "If x is greater than 5 and x is less than 3, the assumptions are", ["inconsistent", "contradictory"]),
        _case("contradiction", "To prove there are infinitely many primes, Euclid assumes finitely many primes and derives a", ["contradiction"]),
        _case("contradiction", "If no solution satisfies all constraints, the constraint set is", ["inconsistent", "unsatisfiable"]),
        _case("contradiction", "Reductio ad absurdum is another name for proof by", ["contradiction"]),
    ]

    cases += [
        _case("induction", "Mathematical induction proves a base case and an induction", ["step"]),
        _case("induction", "In induction, after proving P(1), assume P(k) to prove", ["P(k+1)", "k+1", "P(k + 1)"]),
        _case("induction", "The induction hypothesis is the statement assumed true for", ["k", "n"]),
        _case("induction", "Strong induction may assume all previous cases up to", ["k", "n"]),
        _case("induction", "A recursive proof usually needs a base case to", ["start", "terminate"]),
        _case("induction", "To prove a property for all natural numbers, induction checks the base case and the", ["inductive step", "step"]),
    ]

    cases += [
        _case("code_reasoning", "In Python, len([1, 2, 3]) returns", ["3", "three"]),
        _case("code_reasoning", "In Python, bool([]) evaluates to", ["False", "false"]),
        _case("code_reasoning", "In Python, range(3) produces 0, 1, and", ["2", "two"]),
        _case("code_reasoning", "A function with no explicit return in Python returns", ["None", "none"]),
        _case("code_reasoning", "In Python, {'a': 1}['a'] evaluates to", ["1", "one"]),
        _case("code_reasoning", "The time complexity of binary search on a sorted list is", ["O(log n)", "log n", "logarithmic"]),
        _case("code_reasoning", "A stack removes the most recently added item first, also called", ["LIFO", "last in first out"]),
        _case("code_reasoning", "A queue removes the earliest added item first, also called", ["FIFO", "first in first out"]),
        _case("code_reasoning", "A parse error happens before a program successfully", ["runs", "executes", "compiles"]),
        _case("code_reasoning", "If a loop invariant is true before and after each iteration, it helps prove", ["correctness"]),
    ]

    return cases


def build_benchmark() -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in LEGACY_BENCHMARK + _expanded_cases():
        prompt = item["prompt"]
        if prompt in seen:
            continue
        seen.add(prompt)
        out.append(item)
    return out


BENCHMARK = build_benchmark()


def _contains_term(output: str, term: str) -> bool:
    out = output.lower()
    needle = term.lower()
    compact = re.sub(r"\s+", " ", needle).strip()
    if not compact:
        return False
    if len(re.sub(r"[^a-z0-9]", "", compact)) <= 2:
        return re.search(rf"(?<![a-z0-9]){re.escape(compact)}(?![a-z0-9])", out) is not None
    return compact in re.sub(r"\s+", " ", out)


def score(output: str, accept: list, reject: list) -> bool:
    # Must contain at least one accept term
    hit = any(_contains_term(output, a) for a in accept)
    # Must not contain any reject terms
    no_reject = not any(_contains_term(output, r) for r in reject)
    return hit and no_reject


def strict_scores(output: str, accept: list, reject: list) -> dict:
    from ts_bridge.rescore import score_output
    return score_output(output, accept, reject)


def load_model(ckpt_path: str, device: str = "cpu", tokenizer_path: str | None = None):
    import torch
    from model import TensionConfig, TensionLM, generate as _generate
    from ts_bridge.smoke_test import _migrate_fused_kv

    print(f"Loading {ckpt_path} ...")
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["cfg"])
    # Force reference path unless caller's on CUDA — same guard as smoke_test.
    if device == "cpu":
        cfg_dict["use_triton"] = False
    cfg   = TensionConfig(**cfg_dict)
    model = TensionLM(cfg)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    state = _migrate_fused_kv(state, cfg)
    model.load_state_dict(state)
    model.eval().to(device)

    from tokenizers import Tokenizer
    tok_path = tokenizer_path or ckpt["tok_path"]
    tokenizer = Tokenizer.from_file(tok_path)

    ppl = ckpt.get("val_ppl", "?")
    ppl_str = f"{ppl:.2f}" if isinstance(ppl, float) else str(ppl)
    print(f"Model  : {model.num_params:,} params  |  val ppl {ppl_str}  |  device {device}")
    print(f"Step   : {ckpt.get('step', '?')}\n")
    return model, tokenizer, _generate


def load_hf_model(model_name: str, device: str = "cpu"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading HF model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval().to(device)
    print(f"HF model: {model_name}  |  params {sum(p.numel() for p in model.parameters()):,}  |  device {device}\n")
    return model, tokenizer


def hf_encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def hf_decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def hf_generate(
    model,
    tokenizer,
    prompt_ids: list[int],
    *,
    max_new: int,
    temp: float,
    top_p: float,
    rep_penalty: float,
    device: str,
) -> list[int]:
    import torch
    import torch.nn.functional as F

    ids = list(prompt_ids)
    max_ctx = getattr(model.config, "n_positions", 1024)
    for _ in range(max_new):
        ctx = torch.tensor([ids[-max_ctx:]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(ctx).logits[0, -1].float()
        for tok in set(ids[-32:]):
            if logits[tok] > 0:
                logits[tok] /= rep_penalty
            else:
                logits[tok] *= rep_penalty
        logits = logits / max(temp, 1e-5)
        probs = F.softmax(logits, dim=-1)
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum_p = torch.cumsum(sorted_p, dim=-1)
        mask = (cum_p - sorted_p) < top_p
        sorted_p[~mask] = 0.0
        sorted_p = sorted_p / sorted_p.sum()
        ids.append(int(sorted_i[torch.multinomial(sorted_p, 1).item()].item()))
    return ids


def select_benchmark(categories: list[str] | None = None, limit: int | None = None) -> list[dict]:
    selected = BENCHMARK
    if categories:
        wanted = set(categories)
        selected = [item for item in selected if item["category"] in wanted]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("benchmark selection is empty")
    return selected


def load_benchmark_json(path: str) -> list[dict]:
    payload = json.loads(Path(path).read_text())
    items = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
    if not isinstance(items, list):
        raise ValueError("benchmark JSON must be a list or {'items': [...]}")
    required = {"category", "prompt", "accept"}
    for i, item in enumerate(items):
        missing = required - set(item)
        if missing:
            raise ValueError(f"benchmark item {i} missing keys: {sorted(missing)}")
        item.setdefault("reject", [])
    return items


def print_benchmark_summary(items: list[dict]) -> None:
    counts = Counter(item["category"] for item in items)
    print(f"Benchmark: {len(items)} questions")
    for cat in sorted(counts):
        print(f"  {cat:14s}: {counts[cat]}")
    print()


def run_eval(ckpt_path: str, max_new: int = 40, temp: float = 0.3, top_p: float = 0.9,
             device: str = "cpu", tokenizer_path: str | None = None,
             categories_filter: list[str] | None = None,
             limit: int | None = None, seed: int | None = None,
             json_out: str | None = None, ts_mode: str = "base",
             ts_oracle: bool = False, alpha: float = 4.0,
             surface_beta: float = 4.0, edge_weight: float = 1.0,
             span_radius: int = 0, top_k: int = 8,
             hf_model: str | None = None,
             benchmark_json: str | None = None):
    if seed is not None:
        import torch
        random.seed(seed)
        torch.manual_seed(seed)

    if hf_model:
        model, tokenizer = load_hf_model(hf_model, device)
        _generate = None
    else:
        model, tokenizer, _generate = load_model(ckpt_path, device, tokenizer_path)
    if benchmark_json:
        benchmark = load_benchmark_json(benchmark_json)
        if categories_filter:
            wanted = set(categories_filter)
            benchmark = [item for item in benchmark if item["category"] in wanted]
        if limit is not None:
            benchmark = benchmark[:limit]
        if not benchmark:
            raise ValueError("benchmark selection is empty")
    else:
        benchmark = select_benchmark(categories_filter, limit)
    print_benchmark_summary(benchmark)
    use_ts = ts_mode != "base" or ts_oracle
    if use_ts:
        from ts_bridge.rescore import (
            RescoreConfig, build_rule_graph, generate_with_rescore, mode_flags,
        )
        cfg_ts = RescoreConfig(
            alpha=alpha,
            surface_beta=surface_beta,
            sequence_beta=surface_beta,
            edge_weight=edge_weight,
            span_radius=span_radius,
            max_new=max_new,
            temp=temp,
            top_p=top_p,
            rep_penalty=1.2,
            top_k=top_k,
        )
        use_tau, use_surface = mode_flags(ts_mode)

    results = []
    categories = {}

    for item in benchmark:
        if hf_model:
            prompt_ids = hf_encode(tokenizer, item["prompt"])
        else:
            enc = tokenizer.encode(item["prompt"])
            prompt_ids = enc.ids
        graph_info = None
        topk = []
        if use_ts:
            if hf_model:
                raise ValueError("TS modes are only supported for TensionLM checkpoints, not HF baselines")
            graph_info = build_rule_graph(
                item, prompt_ids, tokenizer,
                edge_weight=edge_weight, span_radius=span_radius,
                oracle=ts_oracle,
            )
            trace = generate_with_rescore(
                model, tokenizer, prompt_ids,
                graph_result=graph_info,
                config=cfg_ts,
                use_tau_bias=use_tau,
                use_surface_rescore=use_surface,
                use_sequence_rescore=(ts_mode == "sequence"),
                device=device,
            )
            gen_text = trace.generated_text
            topk = trace.first_topk
        else:
            if hf_model:
                ids_out = hf_generate(
                    model, tokenizer, prompt_ids,
                    max_new=max_new, temp=temp, top_p=top_p,
                    rep_penalty=1.2, device=device,
                )
                gen_ids = ids_out[len(prompt_ids):]
                gen_text = hf_decode(tokenizer, gen_ids)
            else:
                ids_out = _generate(model, prompt_ids, max_new=max_new, temp=temp, top_p=top_p, rep_penalty=1.2)
                gen_ids  = ids_out[len(prompt_ids):]
                gen_text = tokenizer.decode(gen_ids)

        scores = strict_scores(gen_text, item["accept"], item["reject"])
        correct = scores["substring_correct"]

        result = {
            **item,
            "output": gen_text,
            "correct": correct,
            **scores,
            "first_topk": topk,
        }
        if graph_info is not None:
            result["graph"] = graph_info.to_dict()
        results.append(result)
        cat = item["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if correct:
            categories[cat]["correct"] += 1

        tick = "✓" if scores["prefix_correct"] else ("~" if correct else "✗")
        print(f"[{tick}] [{item['category']:12s}] {item['prompt'][:60]}")
        print(f"       Expected: {item['accept']}  |  Got: {gen_text[:80].strip()!r}")
        print(f"       prefix={scores['prefix_correct']} substring={correct}"
              f" first={scores['first_answer']!r}\n")

    # Summary
    total   = len(results)
    correct = sum(r["correct"] for r in results)
    prefix_correct = sum(r["prefix_correct"] for r in results)
    print("=" * 70)
    print(f"OVERALL substring: {correct}/{total}  ({100*correct/total:.1f}%)")
    print(f"OVERALL prefix   : {prefix_correct}/{total}  ({100*prefix_correct/total:.1f}%)\n")
    print("By category:")
    for cat, d in categories.items():
        pct = 100 * d["correct"] / d["total"]
        print(f"  {cat:14s}: {d['correct']}/{d['total']}  ({pct:.0f}%)")
    print("=" * 70)
    if json_out:
        payload = {
            "checkpoint": ckpt_path,
            "hf_model": hf_model,
            "benchmark_json": benchmark_json,
            "tokenizer": tokenizer_path,
            "seed": seed,
            "max_new": max_new,
            "temp": temp,
            "top_p": top_p,
            "correct": correct,
            "prefix_correct": prefix_correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
            "prefix_accuracy": prefix_correct / total if total else 0.0,
            "ts_mode": ts_mode,
            "ts_oracle": ts_oracle,
            "alpha": alpha,
            "surface_beta": surface_beta,
            "edge_weight": edge_weight,
            "span_radius": span_radius,
            "categories": categories,
            "results": results,
        }
        out_path = Path(json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote JSON results: {out_path}")
    return correct, total, results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--hf_model", default=None,
                   help="Evaluate a Hugging Face causal LM such as gpt2 instead of a TensionLM checkpoint.")
    p.add_argument("--max_new",    default=40,  type=int)
    p.add_argument("--temp",       default=0.3, type=float)
    p.add_argument("--top_p",      default=0.9, type=float)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--tokenizer",  default=None,
                   help="Override tokenizer path (default: ckpt['tok_path']). "
                        "Needed when the checkpoint's tok_path is stale.")
    p.add_argument("--category", action="append", default=None,
                   help="Run only this category. Can be passed multiple times.")
    p.add_argument("--benchmark_json", default=None,
                   help="Run a custom benchmark JSON list or {'items': [...]} instead of the built-in benchmark.")
    p.add_argument("--limit", default=None, type=int,
                   help="Run only the first N selected questions.")
    p.add_argument("--seed", default=42, type=int,
                   help="Sampling seed for reproducible eval runs.")
    p.add_argument("--json_out", default=None,
                   help="Write full results and generated text to JSON.")
    p.add_argument("--ts_mode", choices=["base", "tau", "surface", "both", "sequence"], default="base",
                   help="TS-assisted generation mode.")
    p.add_argument("--ts_oracle", action="store_true",
                   help="Build TS graph from accept answers instead of rules; diagnostic only.")
    p.add_argument("--alpha", default=4.0, type=float,
                   help="Tau-bias alpha for TS modes.")
    p.add_argument("--surface_beta", default=4.0, type=float,
                   help="Candidate logit boost for TS surface/both modes.")
    p.add_argument("--edge_weight", default=1.0, type=float)
    p.add_argument("--span_radius", default=0, type=int)
    p.add_argument("--top_k", default=8, type=int)
    p.add_argument("--list_categories", action="store_true",
                   help="Print benchmark category counts and exit.")
    args = p.parse_args()
    if args.list_categories:
        items = load_benchmark_json(args.benchmark_json) if args.benchmark_json else select_benchmark(args.category, args.limit)
        if args.benchmark_json and args.category:
            wanted = set(args.category)
            items = [item for item in items if item["category"] in wanted]
        if args.benchmark_json and args.limit is not None:
            items = items[:args.limit]
        print_benchmark_summary(items)
        return
    if args.checkpoint is None and args.hf_model is None:
        p.error("--checkpoint or --hf_model is required unless --list_categories is used")
    run_eval(args.checkpoint, args.max_new, args.temp, args.top_p,
             device=args.device, tokenizer_path=args.tokenizer,
             categories_filter=args.category, limit=args.limit,
             seed=args.seed, json_out=args.json_out,
             ts_mode=args.ts_mode, ts_oracle=args.ts_oracle,
             alpha=args.alpha, surface_beta=args.surface_beta,
             edge_weight=args.edge_weight, span_radius=args.span_radius,
             top_k=args.top_k, hf_model=args.hf_model,
             benchmark_json=args.benchmark_json)


if __name__ == "__main__":
    main()
