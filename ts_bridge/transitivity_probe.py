"""
ts_bridge.transitivity_probe
============================

Diagnostic substrate intervention for formal transitivity prompts.

This is not a fair benchmark: it seeds an oracle graph edge from the answer
token already present in the prompt to the final query token. The point is to
answer a narrower TS question:

    If the external substrate has the right constraint edge, can TensionLM's
    graph-bias path pull the surface completion toward the right node?

Run:
    python -m ts_bridge.transitivity_probe \
        --checkpoint checkpoints/117m-curriculum/pytorch_model.pt \
        --tokenizer checkpoints/117m-curriculum/tokenizer.json

    # Use prompt-structure rules instead of oracle accept answers:
    python -m ts_bridge.transitivity_probe --mode rule --checkpoint ...
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import formal_eval                                                    # noqa: E402
from ts_bridge import GraphBias, UniversalLivingGraph                  # noqa: E402
from ts_bridge.smoke_test import load_model                            # noqa: E402


MODE_CHOICES = ("oracle", "rule")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _norm_token(content: str) -> str:
    text = content.replace("Ġ", "").replace("▁", "")
    return re.sub(r"[^a-z0-9-]", "", text.lower())


def _norm_answer(answer: str) -> str:
    return re.sub(r"[^a-z0-9-]", "", answer.lower())


def _candidate_answer_norms(answers: list[str]) -> set[str]:
    norms: set[str] = set()
    for answer in answers:
        norm = _norm_answer(answer)
        if norm:
            norms.add(norm)
        for part in re.split(r"\s+", answer):
            part_norm = _norm_answer(part)
            if part_norm:
                norms.add(part_norm)
    return norms


def _clean_symbol(text: str) -> str:
    return text.strip().strip(". ,;:").strip()


def _first_group(match: re.Match[str] | None) -> str | None:
    if match is None:
        return None
    return _clean_symbol(match.group(1))


def infer_transitive_answer(prompt: str) -> tuple[str | None, str]:
    """
    Infer the terminal node from prompt structure only.

    This is intentionally small and auditable.  It models common transitivity
    prompts as a constraint path X -> Y -> Z and returns Z, without reading the
    benchmark's accept list.
    """
    p = prompt.strip()

    patterns: list[tuple[str, str]] = [
        # Symbolic implication chains.
        (r"If\s+\S+\s+implies\s+\S+\s+and\s+\S+\s+implies\s+(\S+)\s+then\s+\S+\s+implies$",
         "binary_implies"),
        (r"If\s+\S+\s+implies\s+\S+\s+and\s+\S+\s+implies\s+\S+\s+and\s+\S+\s+implies\s+(\S+)\s+then\s+\S+\s+implies$",
         "ternary_implies"),
        # Relational natural-language chains.
        (r"\S+\s+is greater than\s+\S+\.\s+\S+\s+is greater than\s+(\S+)\.\s+Therefore\s+\S+\s+is greater than$",
         "greater_than"),
        (r"\S+\s+is taller than\s+\S+\.\s+\S+\s+is taller than\s+(\S+)\.\s+Therefore\s+\S+\s+is taller than$",
         "taller_than"),
        (r"\S+\s+is older than\s+\S+\.\s+\S+\s+is older than\s+(\S+)\.\s+Therefore\s+\S+\s+is older than$",
         "older_than"),
        (r"\S+\s+is less than\s+\S+\.\s+\S+\s+is less than\s+(\S+)\.\s+Therefore\s+\S+\s+is less than$",
         "less_than"),
        (r"Line\s+\S+\s+is parallel to line\s+\S+\.\s+Line\s+\S+\s+is parallel to line\s+(\S+)\.\s+Therefore line\s+\S+\s+is parallel to$",
         "parallel"),
        (r"\S+\s+depends on\s+\S+\.\s+\S+\s+depends on\s+(\S+)\.\s+Therefore\s+\S+\s+depends on$",
         "depends_on"),
        # Directed graph path.
        (r"Node\s+\S+\s+points to node\s+\S+\.\s+Node\s+\S+\s+points to node\s+(\S+)\.\s+The path from node\s+\S+\s+reaches node$",
         "node_path"),
        # Equality chain.
        (r"\S+\s+equals\s+\S+\.\s+\S+\s+equals\s+(\S+)\.\s+Therefore\s+\S+\s+equals$",
         "equals"),
    ]
    for pattern, name in patterns:
        answer = _first_group(re.search(pattern, p, flags=re.IGNORECASE))
        if answer:
            return answer, name

    subset = _first_group(re.search(
        r"If\s+\S+\s+is a subset of\s+\S+\s+and\s+\S+\s+is a subset of\s+(\S+)\s+then\s+\S+\s+is a subset of$",
        p,
        flags=re.IGNORECASE,
    ))
    if subset:
        return subset, "subset"

    cause = _first_group(re.search(
        r"If\s+\S+\s+causes\s+\S+\s+and\s+\S+\s+causes\s+(\S+),\s+then\s+\S+\s+indirectly causes$",
        p,
        flags=re.IGNORECASE,
    ))
    if cause:
        return cause, "causes"

    class_chain = _first_group(re.search(
        r"If every\s+\S+\s+is a\s+\S+\s+and every\s+\S+\s+is a\s+(\S+),\s+then every\s+\S+\s+is a$",
        p,
        flags=re.IGNORECASE,
    ))
    if class_chain:
        return class_chain, "class_chain"

    return None, "unmatched"


def seed_answer_graph(
    prompt_ids: list[int],
    tokenizer,
    answers: list[str],
    weight: float,
    span_radius: int = 0,
) -> tuple[UniversalLivingGraph, list[str], str]:
    """
    Seed answer-token -> final-query-token edges using prompt token contents.

    The source is any prompt token whose normalised content matches an accepted
    answer form.  The destination is the final prompt token because that query
    row drives the first generated token.
    """
    contents = [tokenizer.id_to_token(int(t)) or f"<{int(t)}>" for t in prompt_ids]
    dst = contents[-1]
    wanted = _candidate_answer_norms(answers)

    matched_indices: list[int] = []
    seen_idx: set[int] = set()
    for idx in range(len(contents) - 2, -1, -1):
        content = contents[idx]
        token_norm = _norm_token(content)
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
    graph.upsert_node(id=f"{dst}#__query", content=dst)
    matched = [contents[i] for i in matched_indices]
    for i, src in enumerate(matched):
        graph.upsert_node(id=f"{src}#__answer_{i}", content=src)
        graph.add_edge(
            src=f"{src}#__answer_{i}",
            dst=f"{dst}#__query",
            weight=weight,
            relation="tension",
            metadata={"probe": "oracle_transitivity"},
        )
    return graph, matched, dst


def candidate_token_ids(contents: list[str], tokenizer) -> list[int]:
    ids: list[int] = []
    for content in contents:
        tid = tokenizer.token_to_id(content)
        if tid is not None and tid not in ids:
            ids.append(int(tid))
    return ids


def generated_text(tokenizer, ids: list[int], prompt_len: int) -> str:
    return tokenizer.decode(ids[prompt_len:])


def prefix_score(output: str, accept: list[str]) -> bool:
    leading = re.search(r"^\s*([A-Za-z0-9-]+)", output)
    if leading is None:
        return False
    out = _norm_answer(leading.group(1))
    for answer in accept:
        answer_norm = _norm_answer(answer)
        if not answer_norm:
            continue
        if out == answer_norm:
            return True
        if len(answer_norm) >= 3 and out.startswith(answer_norm):
            return True
    return False


def _sample_next(
    logits: torch.Tensor,
    recent_ids: list[int],
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


@torch.no_grad()
def probe_generate(
    model,
    tokenizer,
    prompt_ids: list[int],
    *,
    graph: UniversalLivingGraph | None,
    alpha: float,
    candidate_ids: list[int],
    surface_beta: float,
    max_new: int,
    temp: float,
    top_p: float,
    rep_penalty: float,
    device: str,
) -> list[int]:
    ids = list(prompt_ids)
    max_ctx = model.cfg.max_seq_len
    W = model.cfg.window
    for step in range(max_new):
        ctx = ids[-max_ctx:]
        ctx_tensor = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
        tau_bias = None
        tau_bias_global = None
        if graph is not None and alpha != 0:
            engine = GraphBias.from_graph(graph, alpha=alpha)
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
        if step == 0 and surface_beta > 0 and candidate_ids:
            for tid in candidate_ids:
                if 0 <= tid < next_logits.numel():
                    next_logits[tid] += surface_beta
        ids.append(_sample_next(next_logits, ids, temp, top_p, rep_penalty))
    return ids


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--edge_weight", type=float, default=1.0)
    ap.add_argument("--span_radius", type=int, default=0,
                    help="Also seed neighboring prompt tokens around the inferred answer token.")
    ap.add_argument("--max_new", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--rep_penalty", type=float, default=1.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=MODE_CHOICES, default="oracle",
                    help="oracle uses benchmark accept answers; rule infers the answer from prompt structure.")
    ap.add_argument("--surface_beta", type=float, default=0.0,
                    help="First-token logit bonus for graph-supported answer tokens.")
    ap.add_argument("--only_failures", default=None,
                    help="Optional formal_eval JSON; probe only items marked incorrect.")
    ap.add_argument("--json_out", default="logs/eval/transitivity_probe.json")
    args = ap.parse_args()

    model, tokenizer, cfg = load_model(args.checkpoint, args.device, args.tokenizer)
    benchmark = formal_eval.select_benchmark(["transitivity"])
    if args.only_failures:
        failed_prompts = {
            r["prompt"]
            for r in json.loads(Path(args.only_failures).read_text())["results"]
            if not r["correct"]
        }
        benchmark = [item for item in benchmark if item["prompt"] in failed_prompts]

    print(f"model: dim={cfg.dim} L={cfg.num_layers} H={cfg.num_heads} "
          f"W={cfg.window}  prompts={len(benchmark)}  alpha={args.alpha}")

    rows = []
    for item in benchmark:
        prompt_ids = tokenizer.encode(item["prompt"]).ids[: cfg.max_seq_len]
        if args.mode == "oracle":
            seed_answers = item["accept"]
            rule_name = "oracle"
        else:
            inferred, rule_name = infer_transitive_answer(item["prompt"])
            seed_answers = [inferred] if inferred else []
        graph, sources, dst = seed_answer_graph(
            prompt_ids, tokenizer, seed_answers, args.edge_weight,
            span_radius=args.span_radius,
        )
        if not sources:
            print(f"[skip] mode={args.mode} rule={rule_name} no answer token found in prompt: {item['prompt']}")
            continue
        surface_ids = candidate_token_ids(sources, tokenizer)

        _set_seed(args.seed)
        ids_base = probe_generate(
            model, tokenizer, prompt_ids,
            graph=None,
            alpha=0.0,
            candidate_ids=[],
            surface_beta=0.0,
            max_new=args.max_new,
            temp=args.temp,
            top_p=args.top_p,
            rep_penalty=args.rep_penalty,
            device=args.device,
        )
        base_text = generated_text(tokenizer, ids_base, len(prompt_ids))
        base_ok = formal_eval.score(base_text, item["accept"], item["reject"])
        base_prefix_ok = prefix_score(base_text, item["accept"])

        _set_seed(args.seed)
        ids_bias = probe_generate(
            model, tokenizer, prompt_ids,
            graph=graph,
            alpha=args.alpha,
            candidate_ids=surface_ids,
            surface_beta=args.surface_beta,
            max_new=args.max_new,
            temp=args.temp,
            top_p=args.top_p,
            rep_penalty=args.rep_penalty,
            device=args.device,
        )
        bias_text = generated_text(tokenizer, ids_bias, len(prompt_ids))
        bias_ok = formal_eval.score(bias_text, item["accept"], item["reject"])
        bias_prefix_ok = prefix_score(bias_text, item["accept"])

        marker = "RECOVER" if bias_prefix_ok and not base_prefix_ok else (
            "KEEP" if bias_prefix_ok else "MISS"
        )
        print(f"[{marker:7s}] mode={args.mode} rule={rule_name} src={sources} -> dst={dst!r}")
        if args.surface_beta:
            print(f"  surface ids: {surface_ids} beta={args.surface_beta}")
        print(f"  prompt : {item['prompt']}")
        print(f"  base   : {base_text[:90]!r}  ok={base_ok} prefix={base_prefix_ok}")
        print(f"  biased : {bias_text[:90]!r}  ok={bias_ok} prefix={bias_prefix_ok}\n")

        rows.append({
            "prompt": item["prompt"],
            "accept": item["accept"],
            "reject": item["reject"],
            "sources": sources,
            "dst": dst,
            "mode": args.mode,
            "rule": rule_name,
            "seed_answers": seed_answers,
            "surface_ids": surface_ids,
            "baseline": base_text,
            "baseline_correct": base_ok,
            "baseline_prefix_correct": base_prefix_ok,
            "biased": bias_text,
            "biased_correct": bias_ok,
            "biased_prefix_correct": bias_prefix_ok,
            "edges": len(graph.edges),
        })

    base_correct = sum(r["baseline_correct"] for r in rows)
    bias_correct = sum(r["biased_correct"] for r in rows)
    recovered = sum(r["biased_correct"] and not r["baseline_correct"] for r in rows)
    base_prefix_correct = sum(r["baseline_prefix_correct"] for r in rows)
    bias_prefix_correct = sum(r["biased_prefix_correct"] for r in rows)
    prefix_recovered = sum(
        r["biased_prefix_correct"] and not r["baseline_prefix_correct"]
        for r in rows
    )
    print("=" * 70)
    print(f"substring baseline: {base_correct}/{len(rows)}")
    print(f"substring biased  : {bias_correct}/{len(rows)}")
    print(f"substring recovered failures: {recovered}")
    print(f"prefix baseline   : {base_prefix_correct}/{len(rows)}")
    print(f"prefix biased     : {bias_prefix_correct}/{len(rows)}")
    print(f"prefix recovered failures: {prefix_recovered}")
    print("=" * 70)

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "tokenizer": args.tokenizer,
        "alpha": args.alpha,
        "edge_weight": args.edge_weight,
        "span_radius": args.span_radius,
        "surface_beta": args.surface_beta,
        "max_new": args.max_new,
        "temp": args.temp,
        "top_p": args.top_p,
        "seed": args.seed,
        "mode": args.mode,
        "baseline_correct": base_correct,
        "biased_correct": bias_correct,
        "baseline_prefix_correct": base_prefix_correct,
        "biased_prefix_correct": bias_prefix_correct,
        "total": len(rows),
        "recovered": recovered,
        "prefix_recovered": prefix_recovered,
        "results": rows,
    }, indent=2))
    print(f"Wrote JSON results: {out}")


if __name__ == "__main__":
    main()
