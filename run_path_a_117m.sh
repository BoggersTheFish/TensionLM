#!/usr/bin/env bash
set -euo pipefail

# Path A means raw TensionLM must beat raw GPT-2 as a model.
# This launcher sets up the next 117M run:
#   - GPT-2 tokenizer / vocab 50257
#   - W=256, max_seq_len=2048, global_every=3
#   - regenerated logic shards under the GPT-2 tokenizer
#   - ProofPile/formal-language stage
#   - math+code stage with logic_mix=0.10

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PY="${PY:-$ROOT/.venv/bin/python}"
NPROC="${NPROC:-2}"

TOK="$ROOT/data/gpt2-tokenizer/tokenizer.json"
LOGIC_DIR="$ROOT/data/path-a-logic-gpt2"
PROOF_DIR="$ROOT/data/path-a-proofpile-gpt2"
MATH_CODE_DIR="$ROOT/data/path-a-math-code-gpt2"

STAGE1_DIR="$ROOT/checkpoints/path-a-117m-stage1-logic"
STAGE2_DIR="$ROOT/checkpoints/path-a-117m-stage2-proofpile"
STAGE3_DIR="$ROOT/checkpoints/path-a-117m-stage3-math-code"

mkdir -p "$ROOT/logs"

common_train_args=(
  --preset large
  --window 256
  --max_seq_len 2048
  --seq_len 1024
  --global_every 3
  --vocab_size 50257
  --w_diversity 0.02
  --w_closure 0.01
  --log_every 100
  --eval_every 1000
  --save_every 2000
)

export_tokenizer() {
  "$PY" "$ROOT/export_gpt2_tokenizer.py" --out "$TOK"
}

prepare_logic() {
  export_tokenizer
  "$PY" "$ROOT/generate_logic_data.py" \
    --tokenizer "$TOK" \
    --out_dir "$LOGIC_DIR" \
    --target_tokens "${LOGIC_TOKENS:-200000000}" \
    --shard_tokens "${LOGIC_SHARD_TOKENS:-10000000}"
}

prepare_proofpile() {
  export_tokenizer
  "$PY" "$ROOT/prepare_path_a_data.py" \
    --source "${PROOFPILE_SOURCE:-hf:aklein4/proof-pile-2-fixed:algebraic-stack:text:train}" \
    --out_dir "$PROOF_DIR" \
    --tokenizer "$TOK" \
    --max_tokens "${PROOFPILE_TOKENS:-500000000}" \
    --shard_size "${SHARD_TOKENS:-100000000}"
}

prepare_math_code() {
  export_tokenizer
  "$PY" "$ROOT/prepare_path_a_data.py" \
    --source "${MATH_SOURCE:-hf:open-web-math/open-web-math:text:train}" \
    --source "${CODE_SOURCE:-hf:codeparrot/codecomplex:src:train}" \
    --out_dir "$MATH_CODE_DIR" \
    --tokenizer "$TOK" \
    --max_tokens "${MATH_CODE_TOKENS:-5000000000}" \
    --shard_size "${SHARD_TOKENS:-100000000}"
}

prepare_smoke() {
  export_tokenizer
  "$PY" "$ROOT/generate_logic_data.py" \
    --tokenizer "$TOK" \
    --out_dir "$ROOT/data/path-a-logic-smoke-gpt2" \
    --target_tokens 50000 \
    --shard_tokens 10000
  "$PY" "$ROOT/prepare_path_a_data.py" \
    --source smoke-formal \
    --out_dir "$ROOT/data/path-a-smoke-gpt2" \
    --tokenizer "$TOK" \
    --max_tokens 50000 \
    --shard_size 10000
}

smoke_train() {
  prepare_smoke
  "$PY" "$ROOT/train.py" \
    --data_dir "$ROOT/data/path-a-smoke-gpt2" \
    --dim 64 --num_layers 2 --num_heads 4 --window 16 --ffn_mult 2 \
    --max_seq_len 128 --seq_len 64 --batch_size 2 --grad_accum 1 \
    --train_tokens 1024 \
    --rope --no_osc \
    --w_consistency 0.01 --w_entropy 0.01 \
    --log_every 1 --eval_every 4 --save_every 4 \
    --out_dir "$ROOT/checkpoints/path-a-smoke" \
    --log_csv "$ROOT/logs/path_a_smoke.csv"
}

stage1() {
  torchrun --nproc_per_node="$NPROC" "$ROOT/train.py" \
    "${common_train_args[@]}" \
    --data_dir "$LOGIC_DIR" \
    --train_tokens "${STAGE1_TOKENS:-200000000}" \
    --w_consistency 0.1 --w_entropy 0.05 \
    --out_dir "$STAGE1_DIR" \
    --log_csv "$ROOT/logs/path_a_stage1_logic.csv" \
    >> "$ROOT/logs/path_a_stage1_logic.log" 2>&1
}

stage2() {
  mkdir -p "$STAGE2_DIR"
  cp "$STAGE1_DIR/latest.pt" "$STAGE2_DIR/latest.pt"
  torchrun --nproc_per_node="$NPROC" "$ROOT/train.py" \
    "${common_train_args[@]}" \
    --data_dir "$PROOF_DIR" \
    --train_tokens "${STAGE2_TOKENS:-500000000}" \
    --resume \
    --transfer_lr "${STAGE2_LR:-0.0001}" \
    --w_consistency 0.05 --w_entropy 0.02 \
    --out_dir "$STAGE2_DIR" \
    --log_csv "$ROOT/logs/path_a_stage2_proofpile.csv" \
    >> "$ROOT/logs/path_a_stage2_proofpile.log" 2>&1
}

stage3() {
  mkdir -p "$STAGE3_DIR"
  cp "$STAGE2_DIR/latest.pt" "$STAGE3_DIR/latest.pt"
  torchrun --nproc_per_node="$NPROC" "$ROOT/train.py" \
    "${common_train_args[@]}" \
    --data_dir "$MATH_CODE_DIR" \
    --train_tokens "${STAGE3_TOKENS:-5000000000}" \
    --resume \
    --transfer_lr "${STAGE3_LR:-0.0001}" \
    --w_consistency 0.05 --w_entropy 0.02 \
    --logic_mix "${LOGIC_MIX:-0.10}" \
    --logic_dir "$LOGIC_DIR" \
    --out_dir "$STAGE3_DIR" \
    --log_csv "$ROOT/logs/path_a_stage3_math_code.csv" \
    >> "$ROOT/logs/path_a_stage3_math_code.log" 2>&1
}

eval_stage3() {
  "$PY" "$ROOT/formal_eval.py" \
    --checkpoint "$STAGE3_DIR/latest.pt" \
    --tokenizer "$STAGE3_DIR/tokenizer.json" \
    --device "${DEVICE:-cpu}" \
    --ts_mode base \
    --max_new 12 --temp 0.3 --top_p 0.9 --seed 42 \
    --json_out "$ROOT/logs/eval/tension_pathA_stage3_raw_full_seed42.json"
}

case "${1:-}" in
  export-tokenizer) export_tokenizer ;;
  prepare-logic) prepare_logic ;;
  prepare-proofpile) prepare_proofpile ;;
  prepare-math-code) prepare_math_code ;;
  prepare-smoke) prepare_smoke ;;
  smoke) smoke_train ;;
  stage1) stage1 ;;
  stage2) stage2 ;;
  stage3) stage3 ;;
  eval-stage3) eval_stage3 ;;
  all)
    prepare_logic
    prepare_proofpile
    prepare_math_code
    stage1
    stage2
    stage3
    eval_stage3
    ;;
  *)
    echo "Usage: $0 {export-tokenizer|prepare-logic|prepare-proofpile|prepare-math-code|prepare-smoke|smoke|stage1|stage2|stage3|eval-stage3|all}" >&2
    exit 2
    ;;
esac
