# TensionLM Public Evidence Ledger

This file is the public claim boundary for TensionLM. It separates observed
evidence, published artifacts, and active hypotheses so the project can move
fast without overstating results.

## Stable Claims

| Claim | Current evidence | Boundary |
|---|---|---|
| Sigmoid tension is a working attention replacement. | 1.1M WikiText-2 parity: TensionLM val PPL 57.7 vs matched transformer 57.8. | Mechanism parity, not broad superiority. |
| The tension field is inspectable as weighted token edges. | Per-head/token visualizers, tau export, streaming tau export, corpus/head profiling. | Interpretability utility, not proof of better reasoning. |
| Curriculum order matters for formal domains. | Logic -> language -> math reduces first-contact math shock versus cold start in local runs. | Local/repo benchmark evidence; needs larger external replication. |
| Logic replay reduces forgetting. | `logic_mix` sweeps and the 117M abbreviated run preserve transitivity better than no replay. | Ratio is not settled; 0.10 is a working prior, not an optimum. |
| CPU top-layer repair can move formal/code behavior. | Multi-seed TAC repair: 46/172 prefix vs base 28/172 and GPT-2 12/172. Held-out repair improves 21/120 vs base 10/120. | Gains are partly format/domain adaptation; shuffled-answer control also improves. |
| Matched small-scale tension-vs-softmax did not show a capability edge. | 22M-ish three-seed FineWeb pilot: tension mean PPL 537.85 vs softmax 538.13, outcome_iii_no_capability_edge. | Do not call the mechanism a general capability win from this result. |

## Public Artifacts

- Hugging Face models:
  - `BoggersTheFish/TensionLM-117M`
  - `BoggersTheFish/TensionLM-117M-FineWeb`
  - `BoggersTheFish/TensionLM-117M-Curriculum`
  - `BoggersTheFish/TensionLM-117M-Curriculum-Stage2`
  - `BoggersTheFish/TensionLM-Curriculum-13M`
  - `BoggersTheFish/TensionLM-Phase2-TSNative`
  - `BoggersTheFish/TensionLM-Wave02-22M-H2H`
- Source repo: `github.com/BoggersTheFish/bozo`
- Related proof-control artifacts: `BoggersTheFish/ts-proof-ranker-v0` through `v4`

## Claim Language To Use

- Use: "TensionLM is a sigmoid-tension language-model architecture with
  interpretable token-edge fields and promising formal-domain signals."
- Use: "The best current evidence is local and controlled; some runs are
  positive, and matched softmax comparisons are currently neutral."
- Use: "The TS bridge/rescorer path is diagnostic infrastructure unless the
  output is explicitly labelled extractor-style or rule-supported."

## Claim Language To Avoid

- Avoid: "TensionLM proves sigmoid tension is superior to softmax."
- Avoid: "The model reasons because the graph says so."
- Avoid: "CPU repair proves answer-specific learning." The negative control
  shows part of the gain is adaptation.
- Avoid: "Surface rescoring is unconstrained LM reasoning." It is graph-supported
  answer selection unless evaluated as free generation.

## Next Evidence Required

1. Publish a clean Path A source/doc release with raw-model GPT-2 comparisons,
   CPU repair results, held-out checks, and negative controls.
2. Run a larger held-out TAC set with stronger controls that preserve answer
   frequency and category balance.
3. Finish a GPT-2-tokenized W=256 Path A run with ProofPile/code curriculum when
   GPU compute is available.
4. Keep the matched softmax result visible even though it is neutral; it is the
   credibility anchor for future claims.
