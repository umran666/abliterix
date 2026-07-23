# Evidence Run Resources

This is the operational checklist for turning Abliterix's method catalog into reproducible evidence.

## What I Need From You

| Resource | Needed for | Minimum |
|---|---|---|
| OpenRouter API key | LLM judge under HonestBench | `OPENROUTER_API_KEY` with enough quota for 5k-20k judgements |
| Hugging Face token | Private/gated models, uploads, tokenizer sync | `HF_TOKEN` with read access; write access only for publishing |
| Dense GPU | Phase 1 dense smoke | 1x 24 GB CUDA GPU |
| H100/A100 80 GB or RTX PRO 6000 96 GB | MoE and Gemma 4 runs | 1x 80-96 GB |
| 4x 80 GB node | gpt-oss-120b/vLLM TP stretch | Optional |
| Artifact storage | Checkpoints, generated responses, result JSON | 500 GB minimum; 1-2 TB preferred |
| Budget window | Full method matrix | Start with 24-48 GPU-hours; expand after Phase 1 |

## Recommended First Allocation

Start small and evidence-dense:

1. One 24-48 GB GPU for Qwen2.5-7B smoke.
2. 5k OpenRouter judge calls.
3. One 80 GB GPU day for the best dense variants and first MoE A/B.

This should be enough to produce the first credible method-maturity promotions and at least a few benchmark result JSON files.

## Environment Notes

Linux CUDA is the production environment. Apple Silicon is useful for docs, scripts, and some CPU tests, but not for the full package because `bitsandbytes` does not ship a macOS arm64 wheel.

For Mac-only development, use one of:

```bash
PYTHONPATH=src python -m pytest tests/test_weight_transforms.py tests/test_pareto.py
```

or a minimal venv with CPU-safe dependencies. Do not treat Mac test failures involving CUDA-only dependencies as method failures.

## Run Order

1. Freeze the commit and config path.
2. Run a smoke slice and save raw responses.
3. Re-run the best variants under full HonestBench.
4. Commit result JSON under `benchmarks/results/`.
5. Run `python scripts/build_leaderboard.py`.
6. Update `docs/method_maturity.md` only after rows exist.

## External Baselines

To claim "leading", run the same benchmark against:

- Heretic outputs for the same base model.
- grimjim / DECCP style biprojected outputs where available.
- Existing Hugging Face abliterated checkpoints for Gemma/Qwen/Llama families.
- The unmodified base model.

The base model row is required for calibration, even though it will rank poorly on refusal rate.

## Stop Conditions

Stop a run early when:

- KL exceeds the prune threshold for several consecutive trials.
- Degenerate rate rises above 5% on benign smoke prompts.
- The method cannot beat projected mean on both refusal and KL after a fair warmup.
- Judge cache or dataset hash does not match `benchmarks/SPEC.md`.
