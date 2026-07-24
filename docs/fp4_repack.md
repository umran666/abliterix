# Editing native FP4 models without a BF16 blow-up

Ultra-sparse MoE checkpoints now ship pre-quantised to 4 bits — gpt-oss (MXFP4),
DeepSeek-V4-Flash (NVFP4, 284B), and friends. Abliterix's direct/EGA editing
needs *writable* weights, and the historical answer was to dequantise the whole
model to BF16 first: a 284B FP4 checkpoint (~142 GB) becomes ~568 GB of BF16 and
demands 4× B200 just to hold it, most of which is never touched.

The **offline edit-and-repack** path (`abliterix.core.fp4_repack`) avoids that.

## The idea

Abliteration edits only a *subset* of tensors — fused expert `down_proj`,
`attn.o_proj`, dense `mlp.down_proj` — and each edit is local to one tensor.
So instead of expanding the whole model, stream the original FP4 shards and, for
each edited tensor only:

```
dequant (FP4 → BF16)  →  project (the exact steering math)  →  re-pack (BF16 → FP4)
```

Every other tensor is copied through byte-for-byte. Peak memory is a single
layer's expert block (single-digit GB), and the output is a **standalone FP4
checkpoint** that loads natively on any MXFP4/NVFP4-capable stack (vLLM) with no
special support. It is the mirror image of `abliterix-dequant-fp8` (which goes
FP4/FP8 → BF16 on disk); this keeps the model 4-bit end to end.

## Two-phase workflow

1. **Search / export** — run abliterix as usual, then record the resolved plan
   instead of merging to BF16:

   ```bash
   python scripts/export_model.py \
       --model deepseek-ai/DeepSeek-V4-Flash \
       --checkpoint checkpoints_dsv4_flash_v1 --trial 13 \
       --config configs/deepseek_v4_flash.toml \
       --emit-fp4-plan plan.pt
   ```

   `--emit-fp4-plan` records what the best trial *would* edit (direction +
   strength + projection geometry, keyed by canonical parameter name) without
   mutating weights or uploading anything. It is cheap.

2. **Bake** — replay the plan against the original FP4 snapshot, on a single GPU:

   ```bash
   abliterix-abliterate-fp4 <fp4_snapshot_dir> plan.pt <out_dir>
   ```

   `<out_dir>` is a complete FP4 model directory (`quantization_config`
   preserved, tokenizer + modeling files copied). Push it to the Hub as-is.

## Faithfulness

The projection is the **shared** `abliterix.weight_transforms` kernel the
in-engine HF path uses (`apply_ega_projection` / the standard rank-1 ablation),
so the abliteration fingerprint is bit-identical to an in-memory edit. The EGA
axis is re-resolved from the on-disk dequantised shape (via the recorded
`hidden_dim` + `transposed` flag), so a producer that stores experts transposed
(gpt-oss) still steers the correct axis. Before writing, each source tensor is
checked for pack/unpack self-consistency; pass a `reference_dequant` callback to
`abliterate_fp4_to_disk` to additionally assert fp4_utils' dequant matches the
model's own dequant (the real guard against a producer whose nibble order or
scale encoding differs from what fp4_utils assumes).

## The requant caveat — measure it

MXFP4/NVFP4 is a **4-bit** format: E2M1's eight magnitude levels
(`{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`) are spaced ~1.5–2× apart, so
re-quantising an *edited* (off-grid) tensor inherently carries **~10–15% mean
relative error**. This is not new error stacked on top — the base weights
already live in exactly this regime — but it is the one genuinely unknown
quantity, and abliterix's whole thesis is co-minimising KL.

The bake tool reports the per-tensor requant error (`RepackStats.requant_rel_err`
and the "worst requant rel-err" summary line). **Validate downstream KL on a
small model first** — gpt-oss-20b MXFP4 is the recommended canary — before
trusting the path on a 284B target. If the added KL is unacceptable, keep the
search on true BF16 and only use the repack to shrink the *output*.

**MSE-optimal block-scale search** (`--scale-search S`, default `1`) trims some
of this: for each block it also tries a power-of-two scale one (or more) steps
tighter than the amax rule, keeping whichever minimises reconstruction MSE —
clipping the block's largest element to buy resolution on the bulk. Measured
gain is modest and consistent (mean rel-err ~0.117 → ~0.106 on Gaussian weights,
~0.164 → ~0.144 on heavy-tailed blocks; ~10–12% MSE reduction). `S=1` captures
essentially all of it, so the default is cheap (one extra candidate per block);
higher `S` rarely helps. It is value-idempotent — an already-quantised block
re-packs unchanged — so it never degrades a no-op. The **~11% 4-bit floor
remains**: E2M1 has eight magnitude levels, and no scale choice escapes that.

## Formats

| Format | Block | Element | Block scale | Global scale |
|--------|-------|---------|-------------|--------------|
| MXFP4 (gpt-oss) | 32 | E2M1 (4-bit) | ue8m0 (power-of-two, `uint8`) | — |
| NVFP4 (DeepSeek-V4) | 16 | E2M1 (4-bit) | e4m3 (FP8) | one `fp32` per tensor |

On disk, MXFP4 stores `<name>_blocks` + `<name>_scales`; NVFP4 stores the packed
weight plus a `_scale` (and a `_scale_2` / global) sibling. `fp4_utils` handles
both; `detect_fp4_format` reads the layout from `quantization_config`.

## What this does and does not save

- **Saves**: the *output* footprint (142 GB FP4 vs 568 GB BF16 merge), the
  ability to serve the abliterated model natively as FP4, and the
  `export_merged` rejection of quantized direct edits.
- **Does not (yet) save**: the *search* footprint. Because native MXFP4 still
  loads as BF16 in the engine (`Mxfp4Config(dequantize=True)`), the search/export
  step keeps the historical VRAM sizing. Shrinking the search itself is the
  frozen-FP4 + adapter path (path A) — a separate, larger piece of work.
