# Method Maturity

This document separates three claims that are easy to blur together:

1. A method is implemented in Abliterix.
2. A method is supported by recent external research.
3. A method is leading under Abliterix's own reproducible evaluation contract.

Only the third claim should be used for leaderboard or marketing language.

## Maturity Levels

| Level | Label | Evidence required |
|---:|---|---|
| 0 | Implemented | User-facing config/API exists and the method is reachable from the optimization path. |
| 1 | Unit-tested | Unit tests cover math properties, validation, shape handling, and dispatch. |
| 2 | GPU-smoked | At least one real GPU run confirms the method can execute end-to-end on a real model. |
| 3 | E2E-validated | A frozen benchmark run reports refusal, over-refusal, KL, degeneration, length drift, and runtime. |
| 4 | Competitive | Beats the Abliterix baseline on at least two model families under the same benchmark contract. |
| 5 | Leading | Beats strong external baselines or published prior art on multiple model families, with reproducible artifacts. |

## Current Assessment

| Method | Current level | Evidence in repo | Claim language |
|---|---:|---|---|
| Mean-diff refusal direction | 5 | Default path across shipped configs; original Arditi et al. basis; many released models. | Mature baseline, not novel by itself. |
| Optuna multi-objective search | 5 | Core optimizer and public results tables; inherited and extended from Heretic-style automation. | Strong framework differentiator. |
| Direct-mode weight editing | 5 | Gemma 4 / RR / DeepRefusal configs and results; core steering implementation. | Strong Abliterix production path. |
| Projected abliteration | 4 | Enabled in multiple production configs; backed by grimjim work; partial A/B docs. | Production-ready improvement, still needs method-matrix breadth. |
| Expert-Granular Abliteration (EGA) | 4 | MoE docs, direct path, vLLM in-place path, MoE production configs. | Strong MoE differentiator; needs standalone EGA-off/EGA-on rows. |
| LLM judge + HonestBench spec | 4 | `benchmarks/SPEC.md`, `scripts/build_leaderboard.py`, dataset hashes. | Strong evaluation differentiator; leaderboard rows still missing. |
| Discriminative layer selection | 3 | Implemented in steering path; enabled in quality configs; Qwen/validation notes. | Promising quality knob; needs cross-model matrix. |
| ORBA direct transform | 2 | Pure transform implementation and tests; pod validation row. | Experimental/advanced transform until matrix validates. |
| Biprojected direct transform | 2 | Pure transform implementation and tests; pod validation row. | Experimental/advanced transform until matrix validates. |
| Householder transform | 1 | Pure transform implementation and tests; opt-in only. | Completeness feature, not recommended default. |
| SRA | 3 | `src/abliterix/sra.py`, tests, Granite quality config, docs. | Frontline candidate for quality preservation; not yet proven broadly. |
| SOM directions | 2 | `src/abliterix/som.py`, tests, pod validation shape/correlation check. | Recent research integrated; needs E2E refusal/KL rows. |
| SAE feature-basis steering | 2 | SAE loader/scoring/tests; synthetic pod validation. | Interpretable research path; needs real SAE checkpoint E2E rows. |
| COSMIC | 2 | `src/abliterix/cosmic.py`, quality configs. Current implementation approximates token-position candidates from prompt-level residuals. | Useful approximation; avoid claiming full-paper parity. |
| Optimal Transport | 2 | Vector method and quality configs. | Promising vector method; needs selective-layer E2E runs. |
| Harmfulness + refusal joint ablation | 2 | `src/abliterix/harmfulness.py`, tests, pod orthogonality validation. | Experimental hedge-reduction path. |
| SAFEx expert profiling | 2 | `src/abliterix/safex.py`, tests, pod stats validation. | Strong MoE profiling candidate; needs production A/B against standard profiling. |
| Cliff-head ablation | 2 | `src/abliterix/cliff_head.py`, tests, negative control on non-reasoning Qwen2.5. | Reasoning-model-only candidate; needs R1/Qwen-thinking runs. |
| MoTE hooks | 1 | Hook install/remove tests; dense no-op validation. | Inference-time research primitive, not a default. |
| PolyRefuse harness | 1 | Evaluation helper and tests. | Evaluation add-on, not an abliteration method. |
| SVF | 1 | Scorer training and hook path. No broad E2E evidence yet. | Experimental. |
| GRP-Obliteration | 1 | Minimal GRPO core and tests. Not integrated into default CLI flow. | Research fallback, experimental. |

## Claim Rules

- Use "integrates" for levels 0-2.
- Use "validated" for level 3.
- Use "competitive" for level 4.
- Use "leading" only for level 5.
- A README headline should name the exact model, dataset slice, judge, and KL contract when the evidence is not independently reproduced.

## Promotion Checklist

To promote any method by one level:

1. Add or update tests for the method's expected invariants.
2. Run a GPU smoke on `Qwen/Qwen2.5-7B-Instruct` or another shared control model.
3. Add a result JSON under `benchmarks/results/` using the HonestBench schema when possible.
4. Rebuild `benchmarks/leaderboard.json` and the README leaderboard with `python scripts/build_leaderboard.py`.
5. Record the exact config, commit, GPU, judge model, and prompt hashes.
