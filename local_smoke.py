"""
local_smoke.py — local substrate smoke test for TensionLM.

This does not require a trained checkpoint. It builds a tiny random TensionLM
and a synthetic tokenizer, then verifies the repo's core LLM path:

  - model forward and generation
  - checkpoint save/load via formal_eval.py
  - expanded benchmark selection/scoring
  - tau export into UniversalLivingGraph
  - graph -> tau-bias construction

The goal is environment stability: prove this checkout can execute the LLM
substrate end-to-end before downloading large checkpoints or launching runs.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import formal_eval
from model import TensionConfig, TensionLM, generate
from ts_bridge import GraphBias, TauExporter, UniversalLivingGraph


SMOKE_DIR = Path("logs/local_smoke")
TOKENIZER_PATH = SMOKE_DIR / "tokenizer.json"
CHECKPOINT_PATH = SMOKE_DIR / "tiny_tension.pt"


def build_tokenizer(path: Path, vocab_size: int) -> Tokenizer:
    base_tokens = [
        "[UNK]", "All", "men", "are", "mortal", ".", "Socrates", "is", "a",
        "man", "Therefore", "2", "plus", "equals", "4", "If", "A", "then",
        "B", "C", "and", "implies", "Python", "returns", "False", "True",
    ]
    vocab = {tok: i for i, tok in enumerate(base_tokens)}
    for i in range(len(vocab), vocab_size):
        vocab[f"tok{i}"] = i

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return tokenizer


def build_checkpoint(path: Path, tokenizer_path: Path, vocab_size: int) -> TensionLM:
    torch.manual_seed(7)
    cfg = TensionConfig(
        vocab_size=vocab_size,
        dim=32,
        num_layers=2,
        num_heads=4,
        window=8,
        max_seq_len=64,
        dropout=0.0,
        use_triton=False,
        global_every=0,
    )
    model = TensionLM(cfg).eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": "tension",
            "cfg": cfg.__dict__,
            "model": model.state_dict(),
            "tok_path": str(tokenizer_path),
            "step": 0,
            "val_ppl": None,
        },
        path,
    )
    return model


@torch.no_grad()
def run() -> None:
    vocab_size = 512
    tokenizer = build_tokenizer(TOKENIZER_PATH, vocab_size)
    model = build_checkpoint(CHECKPOINT_PATH, TOKENIZER_PATH, vocab_size)

    ids = tokenizer.encode("All men are mortal . Socrates is a man . Therefore").ids
    x = torch.tensor([ids], dtype=torch.long)
    logits, _, all_tau = model(x, return_all=True)
    assert logits.shape == (1, len(ids), vocab_size), logits.shape
    assert len(all_tau) == model.cfg.num_layers

    generated = generate(model, ids, max_new=2, temp=1.0, top_p=0.9)
    assert len(generated) == len(ids) + 2

    graph = UniversalLivingGraph()
    exporter = TauExporter(graph, edge_threshold=0.0)
    stats = exporter.ingest(ids, all_tau, tokenizer)
    assert stats.tokens == len(ids)
    assert len(graph.nodes) == len(ids)
    assert stats.candidate_pairs > 0

    bias, bias_stats = GraphBias.from_graph(graph, alpha=0.5).local_bias(
        ids, tokenizer, window=model.cfg.window,
    )
    assert bias.shape == (1, len(ids), model.cfg.window), bias.shape
    assert bias_stats.nonzero_pairs >= 0

    selected = formal_eval.select_benchmark(["syllogism"], limit=2)
    assert len(selected) == 2
    assert formal_eval.score(" mortal.", ["mortal"], [])
    assert not formal_eval.score(" immortal.", ["mortal"], ["immortal"])

    correct, total, _ = formal_eval.run_eval(
        str(CHECKPOINT_PATH),
        max_new=1,
        temp=1.0,
        top_p=0.9,
        device="cpu",
        categories_filter=["syllogism"],
        limit=1,
    )
    assert total == 1
    assert 0 <= correct <= total

    print("local smoke OK")
    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"tokenizer : {TOKENIZER_PATH}")
    print(f"graph     : {len(graph.nodes)} nodes / {len(graph.edges)} edges")


if __name__ == "__main__":
    run()
