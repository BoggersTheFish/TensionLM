"""
Learned TS field rescorer.

This is the first learned layer above the rule-built substrate:

  prompt -> rule graph candidates -> model first-step logits -> feature rows
  feature rows -> tiny MLP ranker -> candidate selection

It is intentionally CPU-sized and transparent.  The goal is not to claim a
general benchmark result from a tiny in-repo dataset; the goal is to establish
the data path and see whether learned rescoring can replace a fixed beta.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import formal_eval  # noqa: E402
from ts_bridge.rescore import (  # noqa: E402
    RescoreConfig, build_rule_graph, contains_term, first_answer_span,
    generate_with_rescore, mode_flags, norm_token, score_output, set_seed,
)
from ts_bridge.smoke_test import load_model  # noqa: E402


@dataclass
class CandidateRow:
    prompt_idx: int
    category: str
    prompt: str
    token_id: int
    token: str
    label: int
    in_graph: int
    base_logit: float
    base_prob: float
    tau_logit: float
    tau_prob: float
    base_rank: int
    token_len: int
    token_is_numeric: int
    rule: str


FEATURES = [
    "in_graph",
    "base_logit",
    "base_prob",
    "tau_logit",
    "tau_prob",
    "base_rank",
    "token_len",
    "token_is_numeric",
]


def row_feature_values(row: CandidateRow) -> list[float]:
    return [float(getattr(row, k)) for k in FEATURES]


class FieldRescorer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def token_matches(token: str, accept: list[str]) -> bool:
    content = token.replace("Ġ", " ").replace("▁", " ").strip()
    norm = first_answer_span(content)
    if not norm:
        norm = norm_token(token)
    for answer in accept:
        if contains_term(content, answer):
            return True
        ans = "".join(ch for ch in answer.lower() if ch.isalnum() or ch == "-")
        if ans and (norm == ans or (len(ans) >= 3 and norm.startswith(ans))):
            return True
    return False


@torch.no_grad()
def first_logits(model, prompt_ids: list[int], device: str, tau_bias=None, tau_bias_global=None) -> torch.Tensor:
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    logits, _, _ = model(
        x, return_all=True,
        tau_bias=tau_bias, tau_bias_global=tau_bias_global,
    )
    return logits[0, -1].float().cpu()


def build_rows_for_item(
    model,
    tokenizer,
    item: dict,
    prompt_idx: int,
    cfg: RescoreConfig,
    device: str,
    negatives_k: int,
) -> list[CandidateRow]:
    prompt_ids = tokenizer.encode(item["prompt"]).ids
    graph = build_rule_graph(item, prompt_ids, tokenizer)

    base_logits = first_logits(model, prompt_ids, device)
    base_probs = F.softmax(base_logits, dim=-1)

    tau_logits = base_logits
    tau_probs = base_probs
    if graph.inferred:
        use_tau, _ = mode_flags("tau")
        trace = generate_with_rescore(
            model, tokenizer, prompt_ids,
            graph_result=graph,
            config=cfg,
            use_tau_bias=use_tau,
            use_surface_rescore=False,
            device=device,
        )
        # Recompute tau logits directly so rows are not tied to sampled token.
        from ts_bridge.rescore import GraphBias
        engine = GraphBias.from_graph(graph.graph, alpha=cfg.alpha)
        local, _ = engine.local_bias(prompt_ids, tokenizer, window=model.cfg.window, device=device)
        global_bias = None
        if model.cfg.global_every > 0:
            global_bias, _ = engine.global_bias(prompt_ids, tokenizer, device=device)
        tau_logits = first_logits(model, prompt_ids, device, local, global_bias)
        tau_probs = F.softmax(tau_logits, dim=-1)

    _, top_ids = torch.topk(base_probs, min(negatives_k, base_probs.numel()))
    candidate_ids = list(graph.candidate_token_ids)
    for tid in top_ids.tolist():
        tid = int(tid)
        if tid not in candidate_ids:
            candidate_ids.append(tid)

    rows: list[CandidateRow] = []
    sorted_ids = torch.argsort(base_probs, descending=True)
    rank_map = {int(t): i for i, t in enumerate(sorted_ids[: max(negatives_k, 64)].tolist())}
    for tid in candidate_ids:
        token = tokenizer.id_to_token(int(tid)) or f"<{tid}>"
        label = 1 if token_matches(token, item["accept"]) else 0
        rows.append(CandidateRow(
            prompt_idx=prompt_idx,
            category=item["category"],
            prompt=item["prompt"],
            token_id=int(tid),
            token=token,
            label=label,
            in_graph=1 if tid in graph.candidate_token_ids else 0,
            base_logit=float(base_logits[tid]),
            base_prob=float(base_probs[tid]),
            tau_logit=float(tau_logits[tid]),
            tau_prob=float(tau_probs[tid]),
            base_rank=int(rank_map.get(int(tid), negatives_k + 1)),
            token_len=len(norm_token(token)),
            token_is_numeric=1 if norm_token(token).isdigit() else 0,
            rule=graph.rule,
        ))
    return rows


def write_rows(rows: list[CandidateRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")


def load_rows(path: Path) -> list[CandidateRow]:
    return [CandidateRow(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def feature_matrix(rows: list[CandidateRow]) -> tuple[torch.Tensor, torch.Tensor, dict]:
    raw = torch.tensor([[float(getattr(r, k)) for k in FEATURES] for r in rows], dtype=torch.float32)
    mean = raw.mean(0)
    std = raw.std(0).clamp(min=1e-6)
    x = (raw - mean) / std
    y = torch.tensor([r.label for r in rows], dtype=torch.float32)
    return x, y, {"features": FEATURES, "mean": mean.tolist(), "std": std.tolist()}


def train_ranker(rows: list[CandidateRow], epochs: int = 400, lr: float = 1e-2, seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    x, y, norm = feature_matrix(rows)
    model = FieldRescorer(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    pos = y.sum().clamp(min=1)
    neg = (len(y) - y.sum()).clamp(min=1)
    pos_weight = neg / pos
    for _ in range(epochs):
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, norm


def score_rows(model: FieldRescorer, norm: dict, rows: list[CandidateRow]) -> torch.Tensor:
    raw = torch.tensor([row_feature_values(r) for r in rows], dtype=torch.float32)
    mean = torch.tensor(norm["mean"], dtype=torch.float32)
    std = torch.tensor(norm["std"], dtype=torch.float32)
    with torch.no_grad():
        return model((raw - mean) / std)


def eval_ranker(model: FieldRescorer, norm: dict, rows: list[CandidateRow]) -> dict:
    by_prompt: dict[int, list[CandidateRow]] = {}
    for row in rows:
        by_prompt.setdefault(row.prompt_idx, []).append(row)
    correct = 0
    total = 0
    picks = []
    for prompt_idx, group in by_prompt.items():
        scores = score_rows(model, norm, group)
        best_i = int(torch.argmax(scores).item())
        picked = group[best_i]
        correct += int(picked.label == 1)
        total += 1
        picks.append({
            "prompt_idx": prompt_idx,
            "category": picked.category,
            "token": picked.token,
            "token_id": picked.token_id,
            "label": picked.label,
            "score": float(scores[best_i]),
            "prompt": picked.prompt,
        })
    return {"correct": correct, "total": total, "accuracy": correct / total if total else 0.0, "picks": picks}


def load_ranker(path: str | Path) -> tuple[FieldRescorer, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = FieldRescorer(len(ckpt["features"]))
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt["norm"]


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--tokenizer", default=None)
    b.add_argument("--device", default="cpu")
    b.add_argument("--category", action="append", default=None)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--alpha", type=float, default=4.0)
    b.add_argument("--negatives_k", type=int, default=12)
    b.add_argument("--out", default="logs/eval/field_rows.jsonl")

    t = sub.add_parser("train")
    t.add_argument("--rows", required=True)
    t.add_argument("--out", default="logs/eval/field_rescorer.pt")
    t.add_argument("--epochs", type=int, default=400)
    t.add_argument("--lr", type=float, default=1e-2)
    t.add_argument("--seed", type=int, default=42)

    e = sub.add_parser("eval")
    e.add_argument("--rows", required=True)
    e.add_argument("--model", required=True)
    e.add_argument("--out", default="logs/eval/field_rescorer_eval.json")

    args = ap.parse_args()
    if args.cmd == "build":
        model, tokenizer, _ = load_model(args.checkpoint, args.device, args.tokenizer)
        items = formal_eval.select_benchmark(args.category, args.limit)
        cfg = RescoreConfig(alpha=args.alpha, surface_beta=0.0, max_new=1)
        rows = []
        for i, item in enumerate(items):
            rows.extend(build_rows_for_item(model, tokenizer, item, i, cfg, args.device, args.negatives_k))
        write_rows(rows, Path(args.out))
        print(f"Wrote {len(rows)} candidate rows: {args.out}")
    elif args.cmd == "train":
        rows = load_rows(Path(args.rows))
        model, norm = train_ranker(rows, epochs=args.epochs, lr=args.lr, seed=args.seed)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "norm": norm, "features": FEATURES}, out)
        metrics = eval_ranker(model, norm, rows)
        print(f"train-set rank accuracy: {metrics['correct']}/{metrics['total']} ({metrics['accuracy']:.1%})")
        print(f"Wrote model: {out}")
    elif args.cmd == "eval":
        rows = load_rows(Path(args.rows))
        ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
        model = FieldRescorer(len(ckpt["features"]))
        model.load_state_dict(ckpt["state"])
        metrics = eval_ranker(model, ckpt["norm"], rows)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
        print(f"rank accuracy: {metrics['correct']}/{metrics['total']} ({metrics['accuracy']:.1%})")
        print(f"Wrote eval: {out}")


if __name__ == "__main__":
    main()
