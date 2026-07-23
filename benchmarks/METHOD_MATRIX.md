# Method Evidence Matrix

This is the first reproducible experiment plan for deciding which Abliterix methods are merely available and which are genuinely leading.

## Goals

- Compare methods under one benchmark contract instead of mixed private slices.
- Promote methods through the maturity ladder in `docs/method_maturity.md`.
- Identify safe defaults by architecture family.
- Produce machine-readable rows for `benchmarks/results/`.

## Fixed Evaluation Contract

Use `benchmarks/SPEC.md` whenever full HonestBench is feasible:

| Dimension | Required setting |
|---|---|
| Harmful prompts | `datasets/harmful_500`, SHA-pinned by `benchmarks/SPEC.md` |
| Benign prompts | `datasets/good_500`, SHA-pinned by `benchmarks/SPEC.md` |
| Generation | Greedy, `min_new_tokens = 100`, `max_new_tokens = 150` |
| Judge | OpenRouter `google/gemini-3-flash-preview` unless the spec is bumped |
| Primary metrics | refusal rate, KL vs declared base, over-refusal, degenerate rate |
| Utility metrics | mean length, length z-score, runtime |
| Sort key | refusal rate, KL, over-refusal, degenerate rate |

For early GPU smoke runs, use a 100/100 slice, but label the row `smoke_only` and do not include it in `leaderboard.json`.

## Model Matrix

| Slot | Model | Why it exists | Minimum GPU |
|---|---|---|---|
| Dense control | `Qwen/Qwen2.5-7B-Instruct` | Stable, cheap, widely comparable. | 1x 24 GB |
| Strong refusal dense | `NousResearch/Meta-Llama-3-8B-Instruct` or Mistral RR stripped base | Tests strong RLHF/instruct refusal. | 1x 24-48 GB |
| MoE control | `Qwen/Qwen3.5-35B-A3B` or `Qwen/Qwen3.6-35B-A3B` | Exercises EGA/router/SAFEx. | 1x 80 GB |
| Hard architecture | `google/gemma-4-26B-A4B-it` or `google/gemma-4-31B-it` | Double-norm / PLE / delayed-refusal stress test. | 1x 80-96 GB |
| Large MoE stretch | `openai/gpt-oss-120b`-class config | Validates vLLM TP in-place editing. | 4x 80-96 GB |

## Method Matrix

Run these rows per model when the method is architecturally applicable:

| Family | Variant | Config knobs | Promotion target |
|---|---|---|---|
| Baseline | mean + current best steering mode | `vector_method="mean"` | Control row |
| Projection | projected mean | `projected_abliteration=true` | Level 4 |
| Layer selection | projected + discriminative | `discriminative_layer_selection=true` | Level 4 |
| Direct transform | standard | `direct_transform="standard"` | Control for direct-mode rows |
| Direct transform | ORBA | `direct_transform="orba"` | Level 4 if cross-model Pareto win |
| Direct transform | biprojected | `direct_transform="biprojected"` | Level 4 if cross-model Pareto win |
| Vector method | SRA | `vector_method="sra"` | Level 4 |
| Vector method | SOM | `vector_method="som"` | Level 3/4 |
| Vector method | OT | `vector_method="optimal_transport"` | Level 3/4 |
| Vector method | COSMIC | `vector_method="cosmic"` | Level 3, with implementation caveat |
| Dual direction | harmfulness + refusal | `ablate_harmfulness_direction=true` | Level 3 |
| MoE | EGA off/on | compare direct dense-only vs EGA | Level 5 candidate |
| MoE | standard profiling vs SAFEx | `profiling_method="standard"` vs `"safex"` | Level 4 |
| Reasoning | cliff-head off/on | `cliff_head_ablation=true` | Level 3 on reasoning models only |
| Experimental | SVF | `steering_mode="vector_field"` | Level 2/3 |
| Experimental | GRPO | `src/abliterix/grpo.py` training path | Level 2 until CLI-integrated |

## Result File Naming

Use deterministic names:

```text
benchmarks/results/<date>_<model_slug>_<method_slug>_<commit7>.json
```

Example:

```text
benchmarks/results/2026-06-25_qwen2p5-7b_biprojected_a1b2c3d.json
```

## Minimum Result Fields

Full leaderboard rows must satisfy `benchmarks/SPEC.md`. Method-matrix rows should additionally include:

```json
{
  "method_slug": "biprojected",
  "method_family": "direct_transform",
  "config_path": "configs/qwen2.5_7b_biprojected.toml",
  "abliterix_version": "1.8.0",
  "maturity_level_before": 2,
  "maturity_level_after": 3,
  "notes": "Any known caveats, such as smoke slice or judge fallback."
}
```

`scripts/build_leaderboard.py` ignores unknown keys, so these fields are safe to add.

## Phases

| Phase | Scope | Exit criterion |
|---|---|---|
| 0 | CPU/doc prep | Maturity table and matrix merged. |
| 1 | Dense smoke | Qwen2.5-7B runs baseline/projected/disc/ORBA/biprojected/SRA/SOM on 100/100 slice. |
| 2 | Full dense bench | Best 3-4 variants promoted to full HonestBench 500/500. |
| 3 | MoE bench | Qwen3.5/3.6 MoE EGA, router, SAFEx A/B. |
| 4 | Hard architecture bench | Gemma 4 direct-mode variants with delayed-refusal settings. |
| 5 | External baseline comparison | Compare against Heretic, llm-abliteration/DECCP, and released abliterated models under the same runner. |

## Resource Estimate

| Phase | GPU time | API judge calls | Notes |
|---|---:|---:|---|
| Dense smoke | 4-8 GPU-hours | 1k-3k | Best first spend. |
| Dense full bench | 8-16 GPU-hours | 4k-8k | Produces first leaderboard rows. |
| MoE bench | 20-40 H100-hours | 4k-10k | Most important for Abliterix differentiation. |
| Gemma 4 bench | 20-60 H100/Blackwell-hours | 4k-10k | Hardest but highest marketing value. |
| 120B TP stretch | 20-50 4xH100-hours | 2k-6k | Only after smaller matrices converge. |

## Decision Rules

- Promote a method only if it improves the Pareto frontier, not just refusal rate.
- Penalize methods that increase degenerate rate or over-refusal.
- Prefer defaults that win on at least two architecture families.
- Keep model-specific recipes when a method wins only on one architecture.
- Treat smoke slices as directional evidence, never as public leaderboard evidence.
