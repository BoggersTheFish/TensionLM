"""Export the GPT-2 tokenizer as a `tokenizers` JSON file.

TensionLM training reads tokenizers.Tokenizer JSON files. Hugging Face GPT-2
tokenizers expose the same backend tokenizer, so this script makes the 50,257
vocabulary available to `generate_logic_data.py`, `prepare_data.py`, and
`train.py`.

Example:
    python export_gpt2_tokenizer.py --out data/gpt2-tokenizer/tokenizer.json
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out", default="data/gpt2-tokenizer/tokenizer.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from tokenizers import Tokenizer

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    hf_tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_tokenizer.backend_tokenizer.save(str(out))

    tokenizer = Tokenizer.from_file(str(out))
    print(f"Exported {args.model} tokenizer -> {out}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
