"""
ts_bridge.rescore
=================

Reusable TS substrate -> surface rescoring utilities.

The module builds a small constraint graph from prompt structure, feeds that
graph into TensionLM's tau-bias path, and can optionally rescore the first
surface token using graph-supported candidate nodes.

This is diagnostic infrastructure, not a claim that the base LM solved the
task unaided.  The graph builder is deliberately auditable and rule-based so
we can inspect which node and edge caused each output shift.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .bias import GraphBias
from .graph import UniversalLivingGraph


@dataclass
class RescoreConfig:
    alpha: float = 4.0
    surface_beta: float = 0.0
    edge_weight: float = 1.0
    span_radius: int = 0
    max_new: int = 12
    temp: float = 0.3
    top_p: float = 0.9
    rep_penalty: float = 1.2
    top_k: int = 8
    sequence_beta: float = 4.0


@dataclass
class GraphBuildResult:
    graph: UniversalLivingGraph
    rule: str
    seed_answers: list[str]
    source_contents: list[str]
    source_token_ids: list[int]
    dst_content: str
    candidate_token_ids: list[int]
    candidate_sequences: list[dict]
    inferred: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("graph", None)
        d["nodes"] = len(self.graph.nodes)
        d["edges"] = len(self.graph.edges)
        return d


@dataclass
class GenerationTrace:
    ids: list[int]
    generated_text: str
    first_token_id: int | None
    first_token: str | None
    first_topk: list[dict]
    graph: GraphBuildResult | None


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)


def norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9-]", "", text.lower())


def norm_token(content: str) -> str:
    return norm_text(content.replace("Ġ", "").replace("▁", ""))


def contains_term(output: str, term: str) -> bool:
    out = output.lower()
    compact = re.sub(r"\s+", " ", term.lower()).strip()
    if not compact:
        return False
    if len(re.sub(r"[^a-z0-9]", "", compact)) <= 2:
        return re.search(rf"(?<![a-z0-9]){re.escape(compact)}(?![a-z0-9])", out) is not None
    return compact in re.sub(r"\s+", " ", out)


def substring_score(output: str, accept: Sequence[str], reject: Sequence[str]) -> bool:
    return (
        any(contains_term(output, a) for a in accept)
        and not any(contains_term(output, r) for r in reject)
    )


def first_answer_span(output: str) -> str:
    leading = re.search(r"^\s*([A-Za-z0-9-]+)", output)
    return "" if leading is None else norm_text(leading.group(1))


def prefix_score(output: str, accept: Sequence[str]) -> bool:
    out = first_answer_span(output)
    full = norm_text(output)
    if not out:
        return False
    for answer in accept:
        answer_norm = norm_text(answer)
        if not answer_norm:
            continue
        if out == answer_norm or full.startswith(answer_norm):
            return True
        if len(answer_norm) >= 3 and out.startswith(answer_norm):
            return True
    return False


def score_output(output: str, accept: Sequence[str], reject: Sequence[str]) -> dict:
    return {
        "substring_correct": substring_score(output, accept, reject),
        "prefix_correct": prefix_score(output, accept),
        "first_answer": first_answer_span(output),
    }


def _clean_symbol(text: str) -> str:
    return text.strip().strip(". ,;:").strip()


def _first_group(match: re.Match[str] | None) -> str | None:
    return None if match is None else _clean_symbol(match.group(1))


def _num_word(n: int) -> str:
    small = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
        15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
        19: "nineteen", 20: "twenty",
    }
    return small.get(n, str(n))


def infer_transitive_answer(prompt: str) -> tuple[list[str], str]:
    p = prompt.strip()
    patterns: list[tuple[str, str]] = [
        (r"If\s+\S+\s+implies\s+\S+\s+and\s+\S+\s+implies\s+(\S+)\s+then\s+\S+\s+implies$", "binary_implies"),
        (r"If\s+\S+\s+implies\s+\S+\s+and\s+\S+\s+implies\s+\S+\s+and\s+\S+\s+implies\s+(\S+)\s+then\s+\S+\s+implies$", "ternary_implies"),
        (r"\S+\s+is greater than\s+\S+\.\s+\S+\s+is greater than\s+(\S+)\.\s+Therefore\s+\S+\s+is greater than$", "greater_than"),
        (r"\S+\s+is taller than\s+\S+\.\s+\S+\s+is taller than\s+(\S+)\.\s+Therefore\s+\S+\s+is taller than$", "taller_than"),
        (r"\S+\s+is older than\s+\S+\.\s+\S+\s+is older than\s+(\S+)\.\s+Therefore\s+\S+\s+is older than$", "older_than"),
        (r"\S+\s+is less than\s+\S+\.\s+\S+\s+is less than\s+(\S+)\.\s+Therefore\s+\S+\s+is less than$", "less_than"),
        (r"Line\s+\S+\s+is parallel to line\s+\S+\.\s+Line\s+\S+\s+is parallel to line\s+(\S+)\.\s+Therefore line\s+\S+\s+is parallel to$", "parallel"),
        (r"\S+\s+depends on\s+\S+\.\s+\S+\s+depends on\s+(\S+)\.\s+Therefore\s+\S+\s+depends on$", "depends_on"),
        (r"Node\s+\S+\s+points to node\s+\S+\.\s+Node\s+\S+\s+points to node\s+(\S+)\.\s+The path from node\s+\S+\s+reaches node$", "node_path"),
        (r"\S+\s+equals\s+\S+\.\s+\S+\s+equals\s+(\S+)\.\s+Therefore\s+\S+\s+equals$", "equals"),
    ]
    for pattern, name in patterns:
        answer = _first_group(re.search(pattern, p, flags=re.IGNORECASE))
        if answer:
            return [answer], name

    extra_patterns = [
        (r"If\s+\S+\s+is a subset of\s+\S+\s+and\s+\S+\s+is a subset of\s+(\S+)\s+then\s+\S+\s+is a subset of$", "subset"),
        (r"If\s+\S+\s+causes\s+\S+\s+and\s+\S+\s+causes\s+(\S+),\s+then\s+\S+\s+indirectly causes$", "causes"),
        (r"If every\s+\S+\s+is a\s+\S+\s+and every\s+\S+\s+is a\s+(\S+),\s+then every\s+\S+\s+is a$", "class_chain"),
    ]
    for pattern, name in extra_patterns:
        answer = _first_group(re.search(pattern, p, flags=re.IGNORECASE))
        if answer:
            return [answer], name
    return [], "unmatched"


def infer_syllogism_answer(prompt: str) -> tuple[list[str], str]:
    p = prompt.strip()
    patterns = [
        (r"All\s+\S+\s+are\s+([^\.]+)\.\s+\S+\s+is a\s+\S+\.\s+Therefore\s+\S+\s+is$", "all_are"),
        (r"All\s+\S+\s+are\s+([^\.]+)\.\s+\S+\s+are\s+\S+\.\s+Therefore\s+\S+\s+are$", "class_plural"),
        (r"No\s+\S+\s+are\s+([^\.]+)\.\s+All\s+\S+\s+are\s+\S+\.\s+Therefore no\s+\S+\s+are$", "no_are"),
        (r"If it is raining then the ground is\s+([^\.]+)\.\s+It is raining\.\s+Therefore the ground is$", "modus_ponens"),
        (r"All\s+\S+\s+have\s+([^\.]+)\.\s+This shape is a square\.\s+Therefore this shape has$", "shape_property"),
        (r"All rectangles have\s+([^\.]+)\.\s+All squares are rectangles\.\s+Therefore all squares have$", "rectangle_property"),
    ]
    for pattern, name in patterns:
        answer = _first_group(re.search(pattern, p, flags=re.IGNORECASE))
        if answer:
            return [answer], name
    did = re.search(r"Either the butler did it or the gardener did it\.\s+The butler did not do it\.\s+Therefore the gardener$", p, flags=re.IGNORECASE)
    if did:
        return ["did"], "disjunction"
    return [], "unmatched"


def infer_arithmetic_answer(prompt: str) -> tuple[list[str], str]:
    p = prompt.strip()
    specs = [
        (r"(\d+)\s+plus\s+(\d+)\s+equals$", lambda a, b: a + b, "plus"),
        (r"(\d+)\s+minus\s+(\d+)\s+equals$", lambda a, b: a - b, "minus"),
        (r"(\d+)\s+multiplied by\s+(\d+)\s+equals$", lambda a, b: a * b, "multiply"),
        (r"(\d+)\s+divided by\s+(\d+)\s+equals$", lambda a, b: a // b if b else 0, "divide"),
    ]
    for pattern, fn, name in specs:
        m = re.search(pattern, p, flags=re.IGNORECASE)
        if m:
            val = fn(int(m.group(1)), int(m.group(2)))
            return [str(val), _num_word(val)], name
    direct = [
        (r"The square root of 9 is$", ["3", "three"], "sqrt9"),
        (r"The square of 12 is$", ["144"], "square12"),
        (r"The cube of 3 is$", ["27"], "cube3"),
        (r"Half of 18 is$", ["9", "nine"], "half18"),
        (r"One quarter of 20 is$", ["5", "five"], "quarter20"),
        (r"2 to the power of 5 equals$", ["32"], "pow2_5"),
        (r"The greatest common divisor of 12 and 18 is$", ["6", "six"], "gcd"),
        (r"The least common multiple of 4 and 6 is$", ["12", "twelve"], "lcm"),
        (r"The remainder when 17 is divided by 5 is$", ["2", "two"], "mod"),
        (r"The sum of the angles in a triangle is$", ["180"], "triangle_angles"),
    ]
    for pattern, answers, name in direct:
        if re.search(pattern, p, flags=re.IGNORECASE):
            return answers, name
    return [], "unmatched"


def infer_algebra_answer(prompt: str) -> tuple[list[str], str]:
    p = prompt.strip()
    direct = [
        (r"If x plus 3 equals 7 then x equals$", ["4", "four"], "x_plus"),
        (r"If x minus 5 equals 2 then x equals$", ["7", "seven"], "x_minus"),
        (r"If x divided by 4 equals 3 then x equals$", ["12", "twelve"], "x_div"),
        (r"If 2x equals 10 then x equals$", ["5", "five"], "2x"),
        (r"If 3x equals 21 then x equals$", ["7", "seven"], "3x"),
        (r"If x plus x equals 10 then x equals$", ["5", "five"], "x_plus_x"),
        (r"If 5x minus 5 equals 20 then x equals$", ["5", "five"], "5x_minus"),
        (r"Solving x plus 2 equals 9 gives x equals$", ["7", "seven"], "solve_x"),
        (r"The roots of x squared minus 9 equals 0 are 3 and$", ["-3"], "roots9"),
        (r"The solutions to x squared minus 4 equals 0 are x equals 2 and x equals$", ["-2"], "roots4"),
        (r"Expanding \(x plus 1\)\(x plus 1\) gives x squared plus 2x plus$", ["1", "one"], "expand"),
        (r"Factoring x squared minus 16 gives \(x minus 4\)\(x plus$", ["4", "four"], "factor"),
        (r"If y equals 2x and x equals 6 then y equals$", ["12", "twelve"], "substitute"),
        (r"If a plus b equals 10 and a equals 4 then b equals$", ["6", "six"], "solve_b"),
        (r"The slope of y equals 3x plus 2 is$", ["3", "three"], "slope"),
    ]
    for pattern, answers, name in direct:
        if re.search(pattern, p, flags=re.IGNORECASE):
            return answers, name
    return [], "unmatched"


def infer_code_answer(prompt: str) -> tuple[list[str], str]:
    direct = [
        (r"In Python, len\(\[1, 2, 3\]\) returns$", ["3", "three"], "py_len"),
        (r"In Python, bool\(\[\]\) evaluates to$", ["False", "false"], "py_bool_empty"),
        (r"In Python, range\(3\) produces 0, 1, and$", ["2", "two"], "py_range"),
        (r"A function with no explicit return in Python returns$", ["None", "none"], "py_none"),
        (r"In Python, \{'a': 1\}\['a'\] evaluates to$", ["1", "one"], "py_dict"),
        (r"The time complexity of binary search on a sorted list is$", ["O(log n)", "log n"], "binary_search"),
        (r"A stack removes the most recently added item first, also called$", ["LIFO"], "stack"),
        (r"A queue removes the earliest added item first, also called$", ["FIFO"], "queue"),
        (r"A parse error happens before a program successfully$", ["runs", "executes", "compiles"], "parse_error"),
        (r"If a loop invariant is true before and after each iteration, it helps prove$", ["correctness"], "loop_invariant"),
    ]
    for pattern, answers, name in direct:
        if re.search(pattern, prompt.strip(), flags=re.IGNORECASE):
            return answers, name
    return [], "unmatched"


def infer_definition_answer(prompt: str) -> tuple[list[str], str]:
    direct = [
        (r"A prime number.*other than 1 and$", ["itself"], "prime"),
        (r"The Pythagorean theorem.*equals$", ["c squared", "c^2"], "pythagorean"),
        (r"An even number is divisible by$", ["2", "two"], "even"),
        (r"A group is a set.*and$", ["inverse", "inverses"], "group"),
        (r"A vector space.*scalar$", ["multiplication"], "vector_space"),
        (r"A bijection.*and$", ["surjective"], "bijection"),
        (r"A tautology.*always$", ["true"], "tautology"),
        (r"A contradiction.*always$", ["false"], "contradiction"),
        (r"A rational number.*two$", ["integers"], "rational"),
        (r"A continuous function.*or$", ["breaks", "discontinuities"], "continuous"),
        (r"A tree in graph theory.*no$", ["cycles"], "tree"),
        (r"A byte contains$", ["8", "eight"], "byte"),
        (r"A Python dictionary stores key value$", ["pairs"], "dict"),
        (r"Assume n is even and n is odd\. This is a$", ["contradiction"], "contradiction_case"),
        (r"If assuming not P leads to a contradiction, then P is$", ["true"], "not_p"),
        (r"Proof by contradiction starts by assuming the negation and deriving$", ["contradiction"], "proof_by_contradiction"),
        (r"A statement cannot be both true and false.*law of$", ["noncontradiction"], "law"),
        (r"If x is greater than 5 and x is less than 3, the assumptions are$", ["inconsistent"], "inconsistent"),
        (r"Euclid assumes finitely many primes and derives a$", ["contradiction"], "euclid"),
        (r"If no solution satisfies all constraints, the constraint set is$", ["inconsistent"], "unsat"),
        (r"Reductio ad absurdum is another name for proof by$", ["contradiction"], "reductio"),
        (r"Mathematical induction proves a base case and an induction$", ["step"], "induction_step"),
        (r"In induction, after proving P\(1\), assume P\(k\) to prove$", ["P(k+1)", "k+1"], "induction_successor"),
        (r"The induction hypothesis is the statement assumed true for$", ["k"], "hypothesis"),
        (r"Strong induction may assume all previous cases up to$", ["k"], "strong_induction"),
        (r"A recursive proof usually needs a base case to$", ["start", "terminate"], "recursive_base"),
        (r"induction checks the base case and the$", ["step"], "inductive_step"),
        (r"The dot product of orthogonal vectors is$", ["0", "zero"], "dot_orthogonal"),
        (r"The identity matrix multiplied by a vector returns the$", ["same vector", "vector"], "identity_matrix"),
        (r"A square matrix with determinant zero is$", ["singular"], "singular"),
        (r"The transpose of a row vector is a$", ["column vector"], "transpose"),
        (r"If Ax equals b has exactly one solution then A is$", ["invertible"], "invertible"),
        (r"Eigenvectors of a matrix are scaled by their$", ["eigenvalues"], "eigen"),
        (r"The rank of a matrix is the dimension of its column$", ["space"], "rank"),
        (r"The inverse of a product AB is B inverse times$", ["A inverse"], "inverse_product"),
        (r"A matrix with orthonormal columns satisfies Q transpose Q equals$", ["identity", "I"], "orthonormal"),
        (r"The null space contains vectors x such that Ax equals$", ["0", "zero"], "nullspace"),
        (r"The derivative of x squared is$", ["2x", "2 x"], "dx2"),
        (r"The derivative of x cubed is$", ["3x^2", "3x squared"], "dx3"),
        (r"The derivative of a constant is$", ["0", "zero"], "dconstant"),
        (r"The derivative of sin x is$", ["cos x"], "dsin"),
        (r"The derivative of cos x is$", ["-sin x"], "dcos"),
        (r"The derivative of e to the x is$", ["e^x"], "dexp"),
        (r"The integral of 1 dx is$", ["x"], "int1"),
        (r"The integral of x dx is$", ["x^2/2"], "intx"),
        (r"The integral of 2x dx is$", ["x^2"], "int2x"),
        (r"The limit as x approaches infinity of 1 over x is$", ["0", "zero"], "lim_inf"),
        (r"The limit as x approaches 0 of sin x over x is$", ["1", "one"], "lim_sinx"),
        (r"A local maximum has first derivative equal to$", ["0", "zero"], "local_max"),
        (r"The second derivative of x squared is$", ["2", "two"], "second_x2"),
        (r"The derivative of ln x is$", ["1/x"], "dln"),
    ]
    for pattern, answers, name in direct:
        if re.search(pattern, prompt.strip(), flags=re.IGNORECASE):
            return answers, name
    return [], "unmatched"


def infer_answers(category: str, prompt: str) -> tuple[list[str], str]:
    if category == "transitivity":
        return infer_transitive_answer(prompt)
    if category == "syllogism":
        return infer_syllogism_answer(prompt)
    if category == "arithmetic":
        return infer_arithmetic_answer(prompt)
    if category == "algebra":
        return infer_algebra_answer(prompt)
    if category == "code_reasoning":
        return infer_code_answer(prompt)
    if category in {"definition", "contradiction", "induction", "linear_algebra", "calculus"}:
        return infer_definition_answer(prompt)
    return [], "unmatched"


def answer_norms(answers: Sequence[str]) -> set[str]:
    norms: set[str] = set()
    for answer in answers:
        norm = norm_text(answer)
        if norm:
            norms.add(norm)
        for part in re.split(r"\s+", answer):
            part_norm = norm_text(part)
            if part_norm:
                norms.add(part_norm)
    return norms


def token_ids_for_answer(answer: str, tokenizer) -> list[int]:
    ids: list[int] = []
    for text in [answer, " " + answer]:
        try:
            encoded = tokenizer.encode(text).ids
        except Exception:
            encoded = []
        if encoded:
            tid = int(encoded[0])
            if tid not in ids:
                ids.append(tid)
    return ids


def token_sequence_for_answer(answer: str, tokenizer) -> list[int]:
    variants = [" " + answer, answer]
    for text in variants:
        try:
            ids = [int(t) for t in tokenizer.encode(text).ids]
        except Exception:
            ids = []
        if ids:
            return ids
    return []


def build_rule_graph(
    item: dict,
    prompt_ids: list[int],
    tokenizer,
    *,
    edge_weight: float = 1.0,
    span_radius: int = 0,
    oracle: bool = False,
) -> GraphBuildResult:
    if oracle:
        seed_answers = list(item["accept"])
        rule = "oracle"
    else:
        seed_answers, rule = infer_answers(item["category"], item["prompt"])

    contents = [tokenizer.id_to_token(int(t)) or f"<{int(t)}>" for t in prompt_ids]
    dst = contents[-1] if contents else ""
    wanted = answer_norms(seed_answers)

    matched_indices: list[int] = []
    seen_idx: set[int] = set()
    for idx in range(max(0, len(contents) - 2), -1, -1):
        token_norm = norm_token(contents[idx])
        token_hits_answer = (
            token_norm in wanted
            or any(len(token_norm) >= 3 and answer.startswith(token_norm)
                   for answer in wanted)
        )
        if token_hits_answer and idx not in seen_idx:
            lo = max(0, idx - span_radius)
            hi = min(len(contents) - 1, idx + span_radius + 1)
            for span_idx in range(lo, hi):
                if span_idx not in seen_idx:
                    matched_indices.append(span_idx)
                    seen_idx.add(span_idx)

    graph = UniversalLivingGraph()
    if dst:
        graph.upsert_node(id=f"{dst}#__query", content=dst)
    source_contents = [contents[i] for i in matched_indices]
    source_token_ids: list[int] = []
    for i, src in enumerate(source_contents):
        graph.upsert_node(id=f"{src}#__seed_{i}", content=src)
        graph.add_edge(
            src=f"{src}#__seed_{i}",
            dst=f"{dst}#__query",
            weight=edge_weight,
            relation="tension",
            metadata={"rule": rule, "category": item["category"]},
        )
        tid = tokenizer.token_to_id(src)
        if tid is not None and int(tid) not in source_token_ids:
            source_token_ids.append(int(tid))

    candidate_ids = list(source_token_ids)
    candidate_sequences: list[dict] = []
    for answer in seed_answers:
        for tid in token_ids_for_answer(answer, tokenizer):
            if tid not in candidate_ids:
                candidate_ids.append(tid)
        seq = token_sequence_for_answer(answer, tokenizer)
        if seq and not any(c["token_ids"] == seq for c in candidate_sequences):
            candidate_sequences.append({"text": answer, "token_ids": seq})

    return GraphBuildResult(
        graph=graph,
        rule=rule,
        seed_answers=seed_answers,
        source_contents=source_contents,
        source_token_ids=source_token_ids,
        dst_content=dst,
        candidate_token_ids=candidate_ids,
        candidate_sequences=candidate_sequences,
        inferred=bool(seed_answers),
    )


def _sample_next(
    logits: torch.Tensor,
    recent_ids: Sequence[int],
    temp: float,
    top_p: float,
    rep_penalty: float,
) -> int:
    logits = logits.float().clone()
    for tok in set(recent_ids[-32:]):
        if logits[tok] > 0:
            logits[tok] /= rep_penalty
        else:
            logits[tok] *= rep_penalty
    logits = logits / max(temp, 1e-5)
    probs = F.softmax(logits, dim=-1)
    sorted_p, sorted_i = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    mask = (cum - sorted_p) < top_p
    sorted_p[~mask] = 0.0
    sorted_p = sorted_p / sorted_p.sum()
    return int(sorted_i[torch.multinomial(sorted_p, 1).item()].item())


def _topk(logits: torch.Tensor, tokenizer, k: int) -> list[dict]:
    probs = F.softmax(logits.float(), dim=-1)
    vals, idxs = torch.topk(probs, min(k, probs.numel()))
    return [
        {
            "token_id": int(idx),
            "token": tokenizer.id_to_token(int(idx)) or f"<{int(idx)}>",
            "prob": float(prob),
        }
        for prob, idx in zip(vals.tolist(), idxs.tolist())
    ]


@torch.no_grad()
def sequence_logprob(
    model,
    prompt_ids: list[int],
    sequence_ids: list[int],
    *,
    device: str,
) -> float:
    ids = list(prompt_ids)
    score = 0.0
    for tid in sequence_ids:
        ctx = ids[-model.cfg.max_seq_len:]
        x = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)[0, -1].float()
        logp = F.log_softmax(logits, dim=-1)
        score += float(logp[int(tid)].item())
        ids.append(int(tid))
    return score


@torch.no_grad()
def select_answer_sequence(
    model,
    prompt_ids: list[int],
    graph_result: GraphBuildResult,
    *,
    config: RescoreConfig,
    device: str,
) -> dict | None:
    if not graph_result.candidate_sequences:
        return None
    best = None
    for cand in graph_result.candidate_sequences:
        token_ids = [int(t) for t in cand["token_ids"]]
        logp = sequence_logprob(model, prompt_ids, token_ids, device=device)
        # Average logprob avoids always preferring one-token answers, while the
        # sequence beta is the substrate prior that says graph candidates are live.
        score = (logp / max(1, len(token_ids))) + config.sequence_beta
        row = {**cand, "logprob": logp, "score": score}
        if best is None or row["score"] > best["score"]:
            best = row
    return best


@torch.no_grad()
def generate_with_rescore(
    model,
    tokenizer,
    prompt_ids: list[int],
    *,
    graph_result: GraphBuildResult | None,
    config: RescoreConfig,
    use_tau_bias: bool,
    use_surface_rescore: bool,
    device: str,
    use_sequence_rescore: bool = False,
) -> GenerationTrace:
    ids = list(prompt_ids)
    max_ctx = model.cfg.max_seq_len
    W = model.cfg.window
    first_topk: list[dict] = []
    first_token_id: int | None = None
    if use_sequence_rescore and graph_result is not None:
        selected = select_answer_sequence(
            model, prompt_ids, graph_result, config=config, device=device,
        )
        if selected is not None:
            ids.extend([int(t) for t in selected["token_ids"]])
            first_token_id = int(selected["token_ids"][0])
            # Continue after the selected answer span if requested.
            remaining = max(0, config.max_new - len(selected["token_ids"]))
            if remaining == 0:
                generated = tokenizer.decode(ids[len(prompt_ids):])
                first_token = tokenizer.id_to_token(first_token_id) or f"<{first_token_id}>"
                return GenerationTrace(
                    ids=ids,
                    generated_text=generated,
                    first_token_id=first_token_id,
                    first_token=first_token,
                    first_topk=[],
                    graph=graph_result,
                )
            local_cfg = RescoreConfig(**{**asdict(config), "max_new": remaining})
            tail = generate_with_rescore(
                model, tokenizer, ids,
                graph_result=graph_result,
                config=local_cfg,
                use_tau_bias=use_tau_bias,
                use_surface_rescore=False,
                device=device,
                use_sequence_rescore=False,
            )
            ids = tail.ids
            generated = tokenizer.decode(ids[len(prompt_ids):])
            first_token = tokenizer.id_to_token(first_token_id) or f"<{first_token_id}>"
            return GenerationTrace(
                ids=ids,
                generated_text=generated,
                first_token_id=first_token_id,
                first_token=first_token,
                first_topk=[],
                graph=graph_result,
            )
    for step in range(config.max_new):
        ctx = ids[-max_ctx:]
        ctx_tensor = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
        tau_bias = None
        tau_bias_global = None
        if use_tau_bias and graph_result is not None and config.alpha != 0:
            engine = GraphBias.from_graph(graph_result.graph, alpha=config.alpha)
            tau_bias, _ = engine.local_bias(ctx, tokenizer, window=W, device=device)
            if model.cfg.global_every > 0:
                tau_bias_global, _ = engine.global_bias(ctx, tokenizer, device=device)

        logits, _, _ = model(
            ctx_tensor,
            return_all=True,
            tau_bias=tau_bias,
            tau_bias_global=tau_bias_global,
        )
        next_logits = logits[0, -1].float()
        if step == 0 and use_surface_rescore and graph_result is not None:
            for tid in graph_result.candidate_token_ids:
                if 0 <= tid < next_logits.numel():
                    next_logits[tid] += config.surface_beta
        if step == 0:
            first_topk = _topk(next_logits, tokenizer, config.top_k)
        next_id = _sample_next(
            next_logits, ids, config.temp, config.top_p, config.rep_penalty,
        )
        if step == 0:
            first_token_id = next_id
        ids.append(next_id)

    generated = tokenizer.decode(ids[len(prompt_ids):])
    first_token = None
    if first_token_id is not None:
        first_token = tokenizer.id_to_token(first_token_id) or f"<{first_token_id}>"
    return GenerationTrace(
        ids=ids,
        generated_text=generated,
        first_token_id=first_token_id,
        first_token=first_token,
        first_topk=first_topk,
        graph=graph_result,
    )


def mode_flags(mode: str) -> tuple[bool, bool]:
    if mode == "base":
        return False, False
    if mode == "tau":
        return True, False
    if mode == "surface":
        return False, True
    if mode == "both":
        return True, True
    if mode == "sequence":
        return False, False
    raise ValueError(f"unknown TS mode: {mode}")
