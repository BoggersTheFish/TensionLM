"""Package a TensionLM checkpoint as a safetensors Hugging Face repo folder.

The exported folder is intentionally self-contained:
- sharded model-*.safetensors weights
- config.json
- tokenizer.json
- model.py
- inference.py
- README.md
- requirements.txt

Example:
    python package_hf_safetensors.py \
      --checkpoint checkpoints/cpu-repair-117m-heldout-top4/latest.pt \
      --tokenizer checkpoints/cpu-repair-117m-heldout-top4/tokenizer.json \
      --out_dir hf_exports/tensionlm-117m-cpu-repair-heldout \
      --repo_id BoggersTheFish/TensionLM-117M-CPU-Repair-Heldout
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import torch
from safetensors.torch import save_file

from model import TensionConfig, TensionLM
from ts_bridge.smoke_test import _migrate_fused_kv


SPLIT_PREFIX = "__tensionlm_split__."


README_TEMPLATE = """\
---
language: en
license: mit
tags:
  - causal-lm
  - tensionlm
  - safetensors
  - research
library_name: pytorch
---

# {model_name}

This is a research TensionLM checkpoint packaged in `safetensors` format.

It is the CPU-localized Path A repair checkpoint from the `bozo` workspace:
the original 117M curriculum TensionLM was copied, lower/middle blocks were
kept stable, and only upper blocks `8-11` were relaxed on a small dense
formal/code answer-prefix corpus.

## Architecture

- Parameters: {params}
- Vocab size: {vocab_size}
- Layers: {num_layers}
- Hidden size: {dim}
- Heads: {num_heads}
- Local tension window: {window}
- Max sequence length: {max_seq_len}
- RoPE: {use_rope}
- Global layer cadence: every {global_every} blocks
- Attention mechanism: sigmoid tension, not softmax

## Files

- `model-*.safetensors` - sharded model weights; oversized tensors are split
  into storage parts and reassembled by `inference.py`
- `config.json` - TensionLM config and checkpoint metadata
- `tokenizer.json` - tokenizer used by the checkpoint
- `model.py` - TensionLM architecture
- `inference.py` - minimal generation script

## Usage

```bash
pip install torch tokenizers safetensors huggingface_hub
python inference.py --repo_id {repo_id} --prompt "If A implies B and B implies C then A implies"
```

Or after cloning the repo:

```bash
python inference.py --model_dir . --prompt "In Python, list(range(4)) ends with"
```

## Local Held-Out Eval

Held-out TAC v2 benchmark, raw generation, seed `42`.
The v2 prompt set has 120 items with zero prompt overlap against the repair
holdout exclusions, the built-in formal eval, and the earlier v1 held-out TAC
file.

| Model | Prefix | Substring | Arithmetic prefix | Transitivity prefix | Code prefix |
|---|---:|---:|---:|---:|---:|
| GPT-2 124M | 3/120 | 5/120 | 1/40 | 0/40 | 2/40 |
| Base TensionLM 117M | 7/120 | 11/120 | 0/40 | 6/40 | 1/40 |
| Prefix-only repair | 20/120 | 21/120 | 1/40 | 13/40 | 6/40 |
| Prefix-only category control | 6/120 | 6/120 | 0/40 | 2/40 | 4/40 |
| Prefix-only global control | 5/120 | 5/120 | 1/40 | 3/40 | 1/40 |

The main improvement comes from aligning the repair corpus with the eval
contract: completions begin immediately with the answer, without
`Question:`/`Answer:` wrapper examples. Arithmetic remains weak and this should
not be read as a broad reasoning result.

## Limitations

This is not an instruction-tuned assistant. It is a small research language
model and can produce wrong, repetitive, or incoherent continuations. The
held-out eval above is repo-local and narrow; it should not be read as broad
GPT-2 superiority.
"""


INFERENCE = '''\
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer

from model import TensionConfig, TensionLM


def resolve_model_dir(model_dir: str | None, repo_id: str | None) -> Path:
    if model_dir:
        return Path(model_dir)
    if not repo_id:
        raise SystemExit("pass --model_dir or --repo_id")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id))


def sample_next(logits: torch.Tensor, ids: list[int], temp: float, top_p: float, rep_penalty: float) -> int:
    logits = logits.float()
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
    return int(sorted_i[torch.multinomial(sorted_p, 1).item()].item())


def reassemble_split_tensors(state: dict[str, torch.Tensor], payload: dict) -> None:
    for name, spec in payload.get("split_tensors", {}).items():
        pieces = []
        for part in spec["parts"]:
            key = part["key"]
            if key not in state:
                raise RuntimeError(f"missing split tensor part: {key}")
            pieces.append(state.pop(key))
        state[name] = torch.cat(pieces, dim=spec.get("dim", 0)).contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--repo_id", default=None)
    ap.add_argument("--prompt", default="If A implies B and B implies C then A implies")
    ap.add_argument("--max_new", type=int, default=40)
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model_dir = resolve_model_dir(args.model_dir, args.repo_id)
    payload = __import__("json").loads((model_dir / "config.json").read_text())
    cfg = TensionConfig(**payload["cfg"])
    if args.device == "cpu":
        cfg.use_triton = False
    model = TensionLM(cfg)
    state = {}
    shards = sorted(model_dir.glob("model-*.safetensors"))
    if not shards:
        shards = [model_dir / "model.safetensors"]
    for shard in shards:
        state.update(load_file(str(shard), device=args.device))
    reassemble_split_tensors(state, payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = list(unexpected)
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(f"state load mismatch: missing={missing}, unexpected={unexpected}")
    model.lm_head.weight = model.embedding.weight
    model.eval().to(args.device)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    ids = tokenizer.encode(args.prompt).ids
    for _ in range(args.max_new):
        ctx = torch.tensor([ids[-cfg.max_seq_len:]], dtype=torch.long, device=args.device)
        with torch.no_grad():
            logits = model(ctx)[0, -1]
        ids.append(sample_next(logits, ids, args.temp, args.top_p, 1.2))
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
'''


def load_state(ckpt: dict[str, Any], cfg: TensionConfig) -> tuple[dict[str, torch.Tensor], int]:
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    state = _migrate_fused_kv(state, cfg)
    model = TensionLM(cfg)
    model.load_state_dict(state)
    params = model.num_params
    out = {}
    for k, v in model.state_dict().items():
        if k == "lm_head.weight":
            continue
        out[k] = v.detach().cpu().contiguous().clone()
    return out, params


def split_oversized_tensors(
    state: dict[str, torch.Tensor],
    *,
    max_tensor_bytes: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    out: dict[str, torch.Tensor] = {}
    split_manifest: dict[str, Any] = {}

    for name, tensor in state.items():
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes <= max_tensor_bytes:
            out[name] = tensor
            continue
        if tensor.ndim == 0:
            raise ValueError(f"cannot split scalar tensor {name} larger than max_tensor_bytes")

        row_bytes = tensor[0].numel() * tensor.element_size()
        rows_per_part = max(1, max_tensor_bytes // row_bytes)
        parts = []
        for part_idx, start in enumerate(range(0, tensor.shape[0], rows_per_part)):
            end = min(start + rows_per_part, tensor.shape[0])
            key = f"{SPLIT_PREFIX}{name}.part_{part_idx:05d}"
            out[key] = tensor[start:end].contiguous()
            parts.append({"key": key, "dim": 0, "start": start, "end": end})
        split_manifest[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "dim": 0,
            "parts": parts,
        }
    return out, split_manifest


def save_sharded_safetensors(
    state: dict[str, torch.Tensor],
    out: Path,
    *,
    max_shard_bytes: int,
    metadata: dict[str, str],
) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0

    for name, tensor in state.items():
        nbytes = tensor.numel() * tensor.element_size()
        if current and current_bytes + nbytes > max_shard_bytes:
            shards.append({"state": current, "bytes": current_bytes})
            current = {}
            current_bytes = 0
        current[name] = tensor
        current_bytes += nbytes
    if current:
        shards.append({"state": current, "bytes": current_bytes})

    manifest = []
    total = len(shards)
    for i, shard in enumerate(shards, start=1):
        name = f"model-{i:05d}-of-{total:05d}.safetensors"
        save_file(shard["state"], str(out / name), metadata=metadata)
        manifest.append({
            "file": name,
            "bytes": shard["bytes"],
            "tensors": sorted(shard["state"]),
        })
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--model_name", default="TensionLM 117M CPU Repair Held-Out")
    ap.add_argument("--max_shard_mb", type=int, default=96)
    args = ap.parse_args()

    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = dict(ckpt["cfg"])
    cfg_dict["use_triton"] = False
    cfg = TensionConfig(**cfg_dict)
    state, params = load_state(ckpt, cfg)
    state, split_manifest = split_oversized_tensors(
        state,
        max_tensor_bytes=args.max_shard_mb * 1024 * 1024,
    )

    shard_manifest = save_sharded_safetensors(
        state,
        out,
        max_shard_bytes=args.max_shard_mb * 1024 * 1024,
        metadata={
            "format": "pt",
            "architecture": "TensionLM",
            "source_checkpoint": args.checkpoint,
        },
    )

    config = {
        "architectures": ["TensionLM"],
        "model_type": "tensionlm",
        "cfg": cfg.__dict__,
        "checkpoint_step": ckpt.get("step"),
        "val_ppl": ckpt.get("val_ppl"),
        "source_checkpoint": args.checkpoint,
        "parameter_count": params,
        "repo_id": args.repo_id,
        "weight_format": "safetensors",
        "weight_shards": shard_manifest,
        "split_tensors": split_manifest,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    shutil.copy(args.tokenizer, out / "tokenizer.json")
    shutil.copy(Path(__file__).parent / "model.py", out / "model.py")
    (out / "inference.py").write_text(INFERENCE)
    (out / "requirements.txt").write_text(
        "torch>=2.0.0\\ntokenizers>=0.15.0\\nsafetensors>=0.4.0\\nhuggingface_hub>=0.20.0\\n"
    )
    (out / "README.md").write_text(
        README_TEMPLATE.format(
            model_name=args.model_name,
            repo_id=args.repo_id,
            params=f"{params:,}",
            vocab_size=cfg.vocab_size,
            num_layers=cfg.num_layers,
            dim=cfg.dim,
            num_heads=cfg.num_heads,
            window=cfg.window,
            max_seq_len=cfg.max_seq_len,
            use_rope=cfg.use_rope,
            global_every=cfg.global_every,
        )
    )
    print(f"Packaged {params:,} parameters -> {out}")


if __name__ == "__main__":
    main()
