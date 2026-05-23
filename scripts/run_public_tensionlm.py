#!/usr/bin/env python3
"""Download and run a public TensionLM checkpoint from Hugging Face."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import TensionConfig, TensionLM  # noqa: E402


DEFAULT_REPO = "BoggersTheFish/TensionLM-Curriculum-13M"
DEFAULT_PROMPT = "If all mammals are animals and all whales are mammals then"


def _require_hf_download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install huggingface_hub") from exc
    return hf_hub_download


def _load_tokenizer(path: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install tokenizers") from exc
    return Tokenizer.from_file(str(path))


def _download(repo_id: str, filename: str, cache_dir: str | None) -> Path:
    hf_hub_download = _require_hf_download()
    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir))
    except Exception as exc:
        raise RuntimeError(f"Could not download {filename} from {repo_id}: {exc}") from exc


def _load_safetensors_model(repo_id: str, cache_dir: str | None, device: torch.device) -> tuple[TensionLM, object]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("Missing dependency for model.safetensors: pip install safetensors") from exc

    config_path = _download(repo_id, "config.json", cache_dir)
    tokenizer_path = _download(repo_id, "tokenizer.json", cache_dir)
    weights_path = _download(repo_id, "model.safetensors", cache_dir)

    cfg_data = json.loads(config_path.read_text(encoding="utf-8"))
    cfg_data.pop("arch", None)
    cfg_data["use_triton"] = False
    cfg_data["use_grad_checkpoint"] = False
    state = {key.replace("_orig_mod.", ""): value for key, value in load_file(str(weights_path), device="cpu").items()}
    if "pos_embedding.weight" in state:
        cfg_data["use_rope"] = False

    model = TensionLM(TensionConfig(**cfg_data))
    missing, unexpected = model.load_state_dict(state, strict=False)
    important_missing = [key for key in missing if not key.endswith("pos_buf")]
    if important_missing or unexpected:
        raise RuntimeError(f"State dict mismatch: missing={important_missing[:8]} unexpected={unexpected[:8]}")

    model.to(device)
    model.eval()
    return model, _load_tokenizer(tokenizer_path)


def _load_torch_checkpoint(repo_id: str, cache_dir: str | None, device: torch.device) -> tuple[TensionLM, object]:
    checkpoint_path = _download(repo_id, "pytorch_model.pt", cache_dir)
    tokenizer_path = _download(repo_id, "tokenizer.json", cache_dir)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = TensionConfig(**ckpt["cfg"])
    cfg.use_triton = False
    cfg.use_grad_checkpoint = False
    model = TensionLM(cfg)
    state = {key.replace("_orig_mod.", ""): value for key, value in ckpt["model"].items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, _load_tokenizer(tokenizer_path)


def load_public_model(repo_id: str, cache_dir: str | None, device: torch.device) -> tuple[TensionLM, object, str]:
    try:
        model, tokenizer = _load_safetensors_model(repo_id, cache_dir, device)
        return model, tokenizer, "model.safetensors"
    except Exception as safetensors_error:
        try:
            model, tokenizer = _load_torch_checkpoint(repo_id, cache_dir, device)
            return model, tokenizer, "pytorch_model.pt"
        except Exception as torch_error:
            raise SystemExit(
                "Could not load the public TensionLM checkpoint.\n"
                f"- safetensors path failed: {safetensors_error}\n"
                f"- pytorch checkpoint path failed: {torch_error}\n"
                "This script is a public reproducibility smoke test; install requirements and retry."
            ) from torch_error


@torch.inference_mode()
def sample(
    model: TensionLM,
    tokenizer,
    prompt: str,
    max_new: int,
    temperature: float,
    top_p: float,
    rep_penalty: float,
    device: torch.device,
) -> str:
    ids = list(tokenizer.encode(prompt).ids) or [0]
    max_ctx = model.cfg.max_seq_len
    for _ in range(max_new):
        ctx = torch.tensor([ids[-max_ctx:]], dtype=torch.long, device=device)
        logits = model(ctx)[0, -1].float()
        for token_id in set(ids[-48:]):
            if 0 <= token_id < logits.numel():
                logits[token_id] = logits[token_id] / rep_penalty if logits[token_id] > 0 else logits[token_id] * rep_penalty
        probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = (cumulative - sorted_probs) < top_p
        sorted_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        sorted_probs = sorted_probs / sorted_probs.sum().clamp_min(1e-12)
        ids.append(sorted_ids[torch.multinomial(sorted_probs, 1).item()].item())
    return tokenizer.decode(ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a public TensionLM Hugging Face checkpoint.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--rep-penalty", type=float, default=1.25)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available; rerun with --device cpu.")
    device = torch.device(args.device)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    model, tokenizer, weight_file = load_public_model(args.repo_id, args.cache_dir, device)
    output = sample(model, tokenizer, args.prompt, args.max_new, args.temperature, args.top_p, args.rep_penalty, device)

    print("TensionLM public run")
    print(f"Repo: {args.repo_id}")
    print(f"Weights: {weight_file}")
    print(f"Device: {device}")
    print(f"Params: {model.num_params:,}")
    print(f"Config: dim={model.cfg.dim} layers={model.cfg.num_layers} heads={model.cfg.num_heads} window={model.cfg.window}")
    print(f"Prompt: {args.prompt}")
    print()
    print(output.strip())
    print()
    print("Limit: this is a raw narrow research checkpoint, not an instruction-tuned assistant or proof system.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
