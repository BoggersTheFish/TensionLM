#!/usr/bin/env bash
set -euo pipefail

# CPU Path A repair run.
#
# This does not retrain the model from scratch. It copies the existing 117M
# curriculum checkpoint, freezes the lower/middle blocks via --train_layers, and
# fine-tunes a small active region on dense formal/code answer-prefix data.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PY="${PY:-$ROOT/.venv/bin/python}"

BASE_CKPT="${BASE_CKPT:-$ROOT/checkpoints/117m-curriculum/pytorch_model.pt}"
TOKENIZER="${TOKENIZER:-$ROOT/checkpoints/117m-curriculum/tokenizer.json}"
DATA_DIR="${DATA_DIR:-$ROOT/data/formal-repair-117m}"
OUT_DIR="${OUT_DIR:-$ROOT/checkpoints/cpu-repair-117m-top4}"
LOG_DIR="$ROOT/logs"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$ROOT/logs/eval"

prepare_data() {
  args=(
    "$PY" "$ROOT/generate_formal_repair_data.py"
    --tokenizer "$TOKENIZER" \
    --out_dir "$DATA_DIR" \
    --target_tokens "${REPAIR_TOKENS:-2000000}" \
    --shard_tokens "${REPAIR_SHARD_TOKENS:-250000}" \
    --val_tokens "${REPAIR_VAL_TOKENS:-50000}" \
    --seed "${SEED:-42}"
  )
  if [[ -n "${EXCLUDE_PROMPTS_JSON:-}" ]]; then
    args+=(--exclude_prompts_json "$EXCLUDE_PROMPTS_JSON")
  fi
  if [[ "${NO_CANONICAL_EVAL:-0}" == "1" ]]; then
    args+=(--no_canonical_eval)
  fi
  if [[ "${SHUFFLE_ANSWERS:-0}" == "1" ]]; then
    args+=(--shuffle_answers)
  fi
  if [[ "${SHUFFLE_WITHIN_CATEGORY:-0}" == "1" ]]; then
    args+=(--shuffle_within_category)
  fi
  "${args[@]}"
}

seed_checkpoint() {
  cp "$BASE_CKPT" "$OUT_DIR/latest.pt"
  cp "$TOKENIZER" "$OUT_DIR/tokenizer.json"
}

train_repair() {
  "$PY" "$ROOT/train.py" \
    --data_dir "$DATA_DIR" \
    --preset large \
    --vocab_size 32768 \
    --window 64 \
    --global_every 4 \
    --max_seq_len 1024 \
    --seq_len "${SEQ_LEN:-128}" \
    --batch_size "${BATCH_SIZE:-1}" \
    --grad_accum "${GRAD_ACCUM:-4}" \
    --train_tokens "${TRAIN_TOKENS:-65536}" \
    --resume \
    --train_layers "${TRAIN_LAYERS:-8-11}" \
    --transfer_lr "${LR:-0.00005}" \
    --min_lr "${MIN_LR:-0.000005}" \
    --warmup_steps "${WARMUP_STEPS:-10}" \
    --w_closure 0 \
    --w_diversity 0 \
    --w_consistency 0 \
    --w_entropy 0 \
    --triton \
    --rope --no_osc \
    --out_dir "$OUT_DIR" \
    --log_every "${LOG_EVERY:-4}" \
    --eval_every "${EVAL_EVERY:-32}" \
    --save_every "${SAVE_EVERY:-32}" \
    --log_csv "$LOG_DIR/cpu_repair_117m_top4.csv" \
    2>&1 | tee "$LOG_DIR/cpu_repair_117m_top4.log"
}

eval_repair() {
  benchmark_args=()
  if [[ -n "${BENCHMARK_JSON:-}" ]]; then
    benchmark_args+=(--benchmark_json "$BENCHMARK_JSON")
  else
    benchmark_args+=(--category transitivity --category arithmetic --category code_reasoning)
  fi
  "$PY" "$ROOT/formal_eval.py" \
    --checkpoint "$OUT_DIR/latest.pt" \
    --tokenizer "$OUT_DIR/tokenizer.json" \
    --device cpu \
    "${benchmark_args[@]}" \
    --ts_mode base \
    --max_new 12 --temp 0.3 --top_p 0.9 --seed "${EVAL_SEED:-42}" \
    --json_out "${REPAIR_EVAL_JSON:-$ROOT/logs/eval/cpu_repair_117m_raw_tac_seed42.json}"
  "$PY" -m ts_bridge.path_a_compare \
    --run gpt2="${GPT2_EVAL_JSON:-$ROOT/logs/eval/gpt2_pathA_raw_tac_seed42.json}" \
    --run tension117m_repair="${REPAIR_EVAL_JSON:-$ROOT/logs/eval/cpu_repair_117m_raw_tac_seed42.json}" \
    --primary tension117m_repair \
    --baseline gpt2 \
    --out "${COMPARE_JSON:-$ROOT/logs/eval/pathA_cpu_repair_vs_gpt2_seed42.json}"
}

case "${1:-}" in
  prepare-data) prepare_data ;;
  seed-checkpoint) seed_checkpoint ;;
  train) train_repair ;;
  eval) eval_repair ;;
  smoke)
    REPAIR_TOKENS="${REPAIR_TOKENS:-20000}"
    REPAIR_SHARD_TOKENS="${REPAIR_SHARD_TOKENS:-5000}"
    REPAIR_VAL_TOKENS="${REPAIR_VAL_TOKENS:-5000}"
    TRAIN_TOKENS="${TRAIN_TOKENS:-512}"
    EVAL_EVERY="${EVAL_EVERY:-4}"
    SAVE_EVERY="${SAVE_EVERY:-4}"
    LOG_EVERY="${LOG_EVERY:-1}"
    prepare_data
    seed_checkpoint
    train_repair
    ;;
  all)
    prepare_data
    seed_checkpoint
    train_repair
    eval_repair
    ;;
  *)
    echo "Usage: $0 {prepare-data|seed-checkpoint|train|eval|smoke|all}" >&2
    exit 2
    ;;
esac
