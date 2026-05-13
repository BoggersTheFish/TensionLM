# TensionLM Improvement Plan

This is the working path from the current CPU/local state to a stronger
TensionLM result. Treat every wave as a constraint-graph update: define the
active node, reduce tension, verify, then publish only stable receipts.

## Current State

- Base substrate: 117M curriculum TensionLM, W=64, vocab=32768.
- Strongest current signal: raw TAC multi-seed and CPU top-layer repair.
- Weakest current signal: repair attribution. Shuffled-answer controls also
  improve, so some gain is format/domain adaptation.
- Compute constraint: local PyTorch is CPU-only; full Path A needs GPU compute.
- Public posture: evidence-bounded, not a general softmax-superiority claim.

## Target State

1. A larger balanced held-out TAC suite with answer-frequency controls.
2. A repair sweep that shows which token budget improves code without losing
   arithmetic/transitivity.
3. A full Path A model: GPT-2 tokenizer, W=256, max_seq_len=2048,
   ProofPile/formal stage, math+code stage, logic replay.
4. Verifier-backed code/proof tasks that move correctness from text matching
   toward executable or symbolic ground truth.
5. A publishable evidence packet: source commit, model card, eval JSON,
   negative controls, and limitation language.

## Wave 1 - Evaluation Substrate

Active node: held-out/control quality.

Tasks:

- Generate `heldout_formal_tac_v2.json` with balanced transitivity,
  arithmetic, and code_reasoning items.
- Generate global and category answer-frequency-matched controls.
- Add category-preserving shuffled repair mode.
- Verify `formal_eval.py --benchmark_json ... --list_categories`.

Exit criteria:

- 120 item TAC benchmark by default: 40/40/40.
- No prompt overlap with the built-in benchmark or v1 held-out set.
- Controls preserve answer frequency globally/category-wise.

## Wave 2 - CPU Repair Sweep

Active node: localized graph relaxation.

Runs:

- Base 117M on v2 TAC, seeds `1,2,3,42`.
- GPT-2 on v2 TAC, seeds `1,2,3,42`.
- Correct repair at `65k`, `125k`, `250k` train tokens.
- Global-shuffled repair at the same budgets.
- Category-shuffled repair at the same budgets.

Stop rule:

- Keep a repair only if code improves and arithmetic/transitivity do not
  regress versus base across seeds.
- Treat gains that also appear in category-shuffled controls as adaptation, not
  answer-specific reasoning.

## Wave 3 - Selector/Rescorer Discipline

Active node: softmax last-mile replacement.

- Retire displacement-only selection as a primary claim.
- Keep top-N oracle as a signal target.
- Test richer features: base logit, tau logit, graph support, answer-sequence
  likelihood, provenance/edge support, and verifier outcome where available.

Exit criteria:

- Held-out positive delta over softmax top-1 across at least 3 seeds.
- Category-balanced gains, not one-category leakage.

## Wave 4 - Full Path A Training

Active node: actual architecture improvement.

Configuration:

- Tokenizer: GPT-2, vocab=50257.
- Window: W=256.
- Context: max_seq_len=2048.
- Global layer cadence: every 3 blocks.
- Stage 1: synthetic logic.
- Stage 2: ProofPile/formal language.
- Stage 3: open-web-math + code + logic_mix.

Exit criteria:

- Beats GPT-2 on balanced TAC and larger built-in formal eval.
- Does not lose code_reasoning.
- Publishes model card with matched controls and known limitations.

## Wave 5 - Grounded Proof/Code Tasks

Active node: correctness grounding.

- Python expression/function tasks with executable unit tests.
- Horn proof-control tasks with symbolic verifier.
- Lean/Coq-style traces later, only after the simple verifier loop is stable.

Exit criteria:

- Correctness measured by execution/verifier, not only substring/prefix match.
- Tension fields/provenance exported as explanation receipts.

## Immediate Commands

```bash
python generate_balanced_heldout_tac.py
python formal_eval.py --benchmark_json ts_bridge/heldout_formal_tac_v2.json --list_categories
python generate_formal_repair_data.py --shuffle_within_category --target_tokens 20000 --out_dir data/formal-repair-heldout-category-control-smoke
```

## Wave 1 Receipt

Implemented:

- `generate_balanced_heldout_tac.py`
- `ts_bridge/heldout_formal_tac_v2.json`
- `ts_bridge/heldout_formal_tac_v2_control_global.json`
- `ts_bridge/heldout_formal_tac_v2_control_category.json`
- `generate_formal_repair_data.py --shuffle_within_category`
- `run_cpu_repair_117m.sh` support for `SHUFFLE_WITHIN_CATEGORY=1`

Verified:

- v2 benchmark counts: `arithmetic=40`, `code_reasoning=40`,
  `transitivity=40`.
- v2 prompt overlap with built-in formal eval: `0`.
- v2 prompt overlap with held-out v1: `0`.
- global control preserves answer multiset and has zero same-answer positions.
- category control preserves per-category answer multisets and has zero
  same-answer positions.
- category-shuffled repair data smoke writes train/val shards successfully.

Next command wave:

```bash
for seed in 1 2 3 42; do
  python formal_eval.py --hf_model gpt2 \
    --benchmark_json ts_bridge/heldout_formal_tac_v2.json \
    --max_new 12 --temp 0.3 --top_p 0.9 --seed "$seed" \
    --json_out "logs/eval/gpt2_heldout_tac_v2_seed${seed}.json"

  python formal_eval.py \
    --checkpoint checkpoints/117m-curriculum/pytorch_model.pt \
    --tokenizer checkpoints/117m-curriculum/tokenizer.json \
    --benchmark_json ts_bridge/heldout_formal_tac_v2.json \
    --max_new 12 --temp 0.3 --top_p 0.9 --seed "$seed" \
    --json_out "logs/eval/tension117m_heldout_tac_v2_seed${seed}.json"
done
```
