# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""FP4 (MXFP4 / NVFP4) pack, unpack, and re-quantisation kernels.

Where :mod:`abliterix.core.fp8_utils` only ever goes FP8 → BF16 (dequant),
this module adds the *missing direction*: BF16 → FP4 (pack / re-quantise).
That closes the loop needed by the offline "edit-and-repack" bake path
(:mod:`abliterix.core.fp4_repack`): a steered BF16 tensor can be written back
into the model's native 4-bit container instead of forcing the whole
checkpoint to be expanded to BF16 (2×+ the footprint).

Two on-disk FP4 layouts are supported, both built on the **E2M1** 4-bit
element format (1 sign, 2 exponent, 1 mantissa; value set
``{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}``, max magnitude 6):

MXFP4 (OCP micro-scaling; gpt-oss)
    * Block of **32** elements along the contraction axis.
    * One **ue8m0** power-of-two scale per block, stored as a ``uint8``
      biased exponent (value ``2**(byte - 127)``).
    * On disk as sibling tensors ``<name>_blocks`` (packed nibbles, last axis
      ``block//2 = 16`` bytes) and ``<name>_scales`` (one byte per block).

NVFP4 (NVIDIA; DeepSeek-V4-Flash expert layout)
    * Block of **16** elements along the contraction axis.
    * One **e4m3 FP8** scale per block, plus a single per-tensor ``fp32``
      global scale (two-level scaling). Dequant:
      ``w = code · block_scale_e4m3 · global_scale``.

Layout convention (both formats)
    All kernels here operate on a logical tensor whose **last axis is the
    contraction / quantisation axis** and is a multiple of ``block_size``.
    This matches the native checkpoints:
    ``down_proj (E, hidden, inter)`` is blocked over ``inter``; fused
    ``gate_up_proj (E, 2·inter, hidden)`` over ``hidden``.

Safety
    Because a repacked tensor must be **bit-layout-compatible** with the
    kernel that will read it back (nibble order, block dim, scale dtype),
    and this module is developed without a real FP4 checkpoint on hand, the
    caller should use :func:`assert_roundtrip_faithful` on real source
    tensors *before* trusting the pack path. It re-packs already-quantised
    (hence exactly representable) values and checks the bytes are recovered
    — a loud failure instead of silent corruption when a layout assumption
    is wrong for a given producer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# E2M1 element format
# ---------------------------------------------------------------------------

# Positive E2M1 levels indexed by the 3-bit ``eem`` field (exp[2] mant[1]).
# bias = 1: subnormal (e=0) → {0, 0.5}; normal (e≥1) → 2^(e-1)·(1 + 0.5·m).
E2M1_LEVELS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX: float = 6.0

# Midpoints between consecutive positive levels — round-to-nearest bucket
# boundaries for magnitude quantisation. len == 7 (one fewer than levels).
_E2M1_MIDPOINTS: tuple[float, ...] = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)

# e4m3 FP8 finite max (NVFP4 block-scale ceiling).
_E4M3_MAX: float = 448.0
_FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)


def _levels_tensor(device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.tensor(E2M1_LEVELS, device=device, dtype=dtype)


def _midpoints_tensor(device: torch.device) -> Tensor:
    return torch.tensor(_E2M1_MIDPOINTS, device=device, dtype=torch.float32)


def quantize_to_e2m1_codes(x: Tensor) -> Tensor:
    """Round a float tensor to nearest E2M1 value; return 4-bit codes (uint8 0..15).

    ``x`` is assumed already divided by its block scale, i.e. its magnitudes
    live in roughly ``[0, 6]``; anything larger saturates to ``±6``. The code
    is ``(sign << 3) | level_index`` which is exactly the E2M1 bit pattern.
    Negative zero is canonicalised to code 0 (+0) so packing is deterministic.
    """
    x32 = x.to(torch.float32)
    sign = (x32 < 0).to(torch.uint8)
    mag = x32.abs()
    # bucketize: number of midpoints strictly below mag → level index 0..7.
    idx = torch.bucketize(mag, _midpoints_tensor(x.device)).to(torch.uint8)
    code = (sign << 3) | idx
    # Canonicalise -0 (code 0b1000) → +0 (0): level index 0 with sign set.
    code = torch.where(idx == 0, torch.zeros_like(code), code)
    return code


def e2m1_codes_to_values(codes: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    """Expand 4-bit E2M1 codes (uint8 0..15) to their float values."""
    idx = (codes & 0x7).to(torch.long)
    sign = ((codes >> 3) & 0x1).bool()
    vals = _levels_tensor(codes.device, dtype)[idx]
    return torch.where(sign, -vals, vals)


# ---------------------------------------------------------------------------
# Nibble packing (2 elements per byte)
# ---------------------------------------------------------------------------


def pack_nibbles(codes: Tensor) -> Tensor:
    """Pack an even-length last axis of 4-bit codes into ``uint8`` bytes.

    Convention (matches OCP / transformers gpt-oss): the **even** element
    goes in the low nibble, the **odd** element in the high nibble
    (``byte = lo | (hi << 4)``). Input last axis must be even; output last
    axis is halved.
    """
    if codes.shape[-1] % 2 != 0:
        raise ValueError(f"nibble pack needs even last axis, got {codes.shape[-1]}")
    codes = codes.to(torch.uint8)
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return (lo & 0xF) | ((hi & 0xF) << 4)


def unpack_nibbles(packed: Tensor) -> Tensor:
    """Inverse of :func:`pack_nibbles`. Output last axis is doubled (uint8 codes)."""
    packed = packed.to(torch.uint8)
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    out = torch.stack((lo, hi), dim=-1)  # (..., n, 2)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


# ---------------------------------------------------------------------------
# MXFP4 (block 32, ue8m0 power-of-two scale)
# ---------------------------------------------------------------------------


def _blockwise_amax(w: Tensor, block_size: int) -> tuple[Tensor, Tensor]:
    """Reshape last axis into blocks; return (blocked_view, per-block amax)."""
    k = w.shape[-1]
    if k % block_size != 0:
        raise ValueError(
            f"contraction axis {k} not divisible by block_size {block_size}"
        )
    blocked = w.reshape(*w.shape[:-1], k // block_size, block_size)
    amax = blocked.abs().amax(dim=-1)  # (..., n_blocks)
    return blocked, amax


def quantize_to_mxfp4(
    w: Tensor, block_size: int = 32, scale_search: int = 0
) -> tuple[Tensor, Tensor]:
    """BF16/FP32 → MXFP4. Returns ``(blocks_uint8, scales_uint8)``.

    ``blocks`` shape ``(..., n_blocks, block_size // 2)``; ``scales`` shape
    ``(..., n_blocks)`` holding the ue8m0 biased exponent (``2**(byte-127)``).

    With ``scale_search == 0`` (default) each block's scale is the smallest
    power of two that keeps every element within ``±E2M1_MAX`` (no clipping) —
    the standard amax rule.

    With ``scale_search == S > 0`` the block scale is chosen to **minimise
    per-block MSE** over the candidate exponents ``{e, e-1, ..., e-S}`` (``e``
    is the amax-tight exponent). A *smaller* scale clips the block's largest
    element but doubles the resolution on the bulk; for heavy-tailed weight
    blocks that trade lowers reconstruction error — the standard way to cut
    4-bit requant damage. Ties prefer the largest exponent (no clipping), so an
    already-quantised block still re-packs to identical values (its amax-tight
    scale is the unique MSE-0 choice), preserving idempotency.
    """
    w32 = w.to(torch.float32)
    blocked, amax = _blockwise_amax(w32, block_size)

    # Amax-tight exponent e: smallest with 2**e >= amax / 6  →  amax / 2**e <= 6.
    ratio = (amax / E2M1_MAX).clamp(min=torch.finfo(torch.float32).tiny)
    base_exp = torch.ceil(torch.log2(ratio))
    base_exp = torch.where(amax > 0, base_exp, torch.zeros_like(base_exp))

    if scale_search <= 0:
        scale_byte = (base_exp + 127.0).clamp(0, 255).to(torch.uint8)
    else:
        # Search {e, e-1, ..., e-S}; keep the min-MSE byte per block. Iterate
        # k ascending with a strict-improvement test so k=0 (largest exponent,
        # no clip) wins on ties → idempotent on already-quantised input.
        best_byte = (base_exp + 127.0).clamp(0, 255)
        best_mse = torch.full_like(amax, float("inf"))
        for k in range(scale_search + 1):
            byte_k = (base_exp - k + 127.0).clamp(0, 255)
            scale_k = torch.pow(2.0, byte_k - 127.0)  # (..., n_blocks)
            codes_k = quantize_to_e2m1_codes(blocked / scale_k.unsqueeze(-1))
            deq_k = e2m1_codes_to_values(codes_k) * scale_k.unsqueeze(-1)
            mse_k = ((deq_k - blocked) ** 2).mean(dim=-1)
            better = mse_k < best_mse
            best_byte = torch.where(better, byte_k, best_byte)
            best_mse = torch.where(better, mse_k, best_mse)
        scale_byte = best_byte.to(torch.uint8)

    scale = torch.pow(2.0, scale_byte.to(torch.float32) - 127.0)  # (..., n_blocks)
    normed = blocked / scale.unsqueeze(-1)
    codes = quantize_to_e2m1_codes(normed)  # (..., n_blocks, block_size)
    blocks = pack_nibbles(codes)  # (..., n_blocks, block_size // 2)
    return blocks, scale_byte


def dequantize_mxfp4(
    blocks: Tensor,
    scales: Tensor,
    block_size: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """MXFP4 → ``out_dtype``. Inverse of :func:`quantize_to_mxfp4`.

    ``blocks`` last axis is ``block_size // 2`` packed bytes; ``scales`` is one
    ue8m0 byte per block. Returns a tensor whose last axis is the fully
    reconstructed contraction dim (``n_blocks * block_size``).
    """
    codes = unpack_nibbles(blocks)  # (..., n_blocks, block_size)
    vals = e2m1_codes_to_values(codes, dtype=torch.float32)
    scale = torch.pow(2.0, scales.to(torch.float32) - 127.0)  # (..., n_blocks)
    out = vals * scale.unsqueeze(-1)
    out = out.reshape(*out.shape[:-2], out.shape[-2] * out.shape[-1])
    return out.to(out_dtype)


# ---------------------------------------------------------------------------
# NVFP4 (block 16, e4m3 block scale + fp32 global scale)
# ---------------------------------------------------------------------------


def compute_nvfp4_global_scale(w: Tensor) -> Tensor:
    """Per-tensor fp32 global scale so block scales fit inside e4m3.

    Convention (NVIDIA ModelOpt / llm-compressor): ``global = amax(w) /
    (E2M1_MAX * E4M3_MAX)`` so the largest block scale ``amax_block /
    (6·global)`` is at most ``E4M3_MAX = 448``. Returns a 0-dim tensor.
    """
    amax = w.abs().amax().to(torch.float32)
    g = amax / (E2M1_MAX * _E4M3_MAX)
    return torch.where(amax > 0, g, torch.ones_like(g))


def quantize_to_nvfp4(
    w: Tensor, block_size: int = 16, global_scale: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """BF16/FP32 → NVFP4. Returns ``(blocks_uint8, block_scales_e4m3, global_fp32)``.

    ``blocks`` shape ``(..., n_blocks, block_size // 2)``; ``block_scales`` is
    one e4m3 value per block; ``global`` is a 0-dim fp32 tensor. Dequant is
    ``code · block_scale · global``.
    """
    if _FP8_E4M3 is None:
        raise RuntimeError("torch build lacks float8_e4m3fn; cannot pack NVFP4")
    w32 = w.to(torch.float32)
    if global_scale is None:
        global_scale = compute_nvfp4_global_scale(w32)
    g = global_scale.to(torch.float32).clamp(min=torch.finfo(torch.float32).tiny)

    blocked, amax = _blockwise_amax(w32, block_size)
    # Per-block scale in "global units": amax_block / (6 · global), clamped to e4m3.
    block_scale = (amax / (E2M1_MAX * g)).clamp(
        min=torch.finfo(torch.float32).tiny, max=_E4M3_MAX
    )
    block_scale_e4m3 = block_scale.to(_FP8_E4M3)
    # Effective per-element scale is block_scale_e4m3 · global.
    eff = block_scale_e4m3.to(torch.float32) * g
    normed = blocked / eff.unsqueeze(-1)
    codes = quantize_to_e2m1_codes(normed)
    blocks = pack_nibbles(codes)
    return blocks, block_scale_e4m3, g


def dequantize_nvfp4(
    blocks: Tensor,
    block_scales: Tensor,
    global_scale: Tensor,
    block_size: int = 16,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """NVFP4 → ``out_dtype``. Inverse of :func:`quantize_to_nvfp4`."""
    codes = unpack_nibbles(blocks)
    vals = e2m1_codes_to_values(codes, dtype=torch.float32)
    eff = block_scales.to(torch.float32) * global_scale.to(torch.float32)
    out = vals * eff.unsqueeze(-1)
    out = out.reshape(*out.shape[:-2], out.shape[-2] * out.shape[-1])
    return out.to(out_dtype)


# ---------------------------------------------------------------------------
# Format description + round-trip safety
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fp4Format:
    """Which FP4 layout a tensor uses and its block size."""

    kind: Literal["mxfp4", "nvfp4"]
    block_size: int

    @property
    def bytes_per_block(self) -> int:
        return self.block_size // 2


def detect_fp4_format(quantization_config: dict | None) -> Fp4Format | None:
    """Map a HF ``quantization_config`` dict to an :class:`Fp4Format`.

    Recognises ``quant_method`` values ``"mxfp4"`` and ``"nvfp4"`` /
    ``"modelopt_fp4"`` / ``"nvfp4"``. Block size is taken from the config
    when present, else the format default (32 for MXFP4, 16 for NVFP4).
    Returns ``None`` when the config is not an FP4 method.
    """
    if not isinstance(quantization_config, dict):
        return None
    method = str(quantization_config.get("quant_method", "")).lower()
    group = quantization_config.get("group_size") or quantization_config.get(
        "block_size"
    )
    if method == "mxfp4":
        return Fp4Format("mxfp4", int(group) if group else 32)
    if method in ("nvfp4", "modelopt_fp4", "modelopt", "nvfp4_ptq"):
        return Fp4Format("nvfp4", int(group) if group else 16)
    return None


def dequantize_fp4(
    fmt: Fp4Format,
    blocks: Tensor,
    scales: Tensor,
    *,
    global_scale: Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Format-dispatching dequant. ``global_scale`` required for NVFP4."""
    if fmt.kind == "mxfp4":
        return dequantize_mxfp4(blocks, scales, fmt.block_size, out_dtype=out_dtype)
    if global_scale is None:
        raise ValueError("NVFP4 dequant requires global_scale")
    return dequantize_nvfp4(
        blocks, scales, global_scale, fmt.block_size, out_dtype=out_dtype
    )


def quantize_fp4(
    fmt: Fp4Format,
    w: Tensor,
    *,
    global_scale: Tensor | None = None,
    scale_search: int = 0,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Format-dispatching pack. Returns ``(blocks, scales, global_scale_or_None)``.

    ``scale_search`` selects the MSE-optimal MXFP4 block-scale search depth
    (see :func:`quantize_to_mxfp4`); it is ignored for NVFP4, whose e4m3 block
    scale is already fine-grained (not restricted to powers of two).

    For NVFP4 the returned global scale is reused from ``global_scale`` when
    given (the streaming tool keeps the source tensor's global scale so block
    scales stay in range), else recomputed.
    """
    if fmt.kind == "mxfp4":
        blocks, scales = quantize_to_mxfp4(w, fmt.block_size, scale_search=scale_search)
        return blocks, scales, None
    blocks, bscale, g = quantize_to_nvfp4(w, fmt.block_size, global_scale=global_scale)
    return blocks, bscale, g


def assert_repack_idempotent(
    blocks: Tensor,
    scales: Tensor,
    fmt: Fp4Format,
    *,
    global_scale: Tensor | None = None,
    name: str = "<tensor>",
) -> None:
    """Verify pack∘unpack reproduces the source *values* exactly (self-consistency).

    Already-quantised weights lie exactly on the FP4 value manifold, so a
    consistent pack/unpack pair must map them back to themselves without
    drift: ``dequant(pack(dequant(source))) == dequant(source)``. Note this
    is *value* equality, not byte equality — an equal-magnitude block whose
    peak rounds to level 3 admits a tighter power-of-two scale, so the bytes
    may legitimately change while every decoded weight is identical.

    This is a regression guard on fp4_utils itself; it does **not** prove the
    layout matches the checkpoint's producer (see
    :func:`assert_matches_reference` for that, which needs a ground-truth
    dequant). Raises ``AssertionError`` on drift.
    """
    w = dequantize_fp4(
        fmt, blocks, scales, global_scale=global_scale, out_dtype=torch.float32
    )
    re_blocks, re_scales, re_g = quantize_fp4(fmt, w, global_scale=global_scale)
    w2 = dequantize_fp4(
        fmt,
        re_blocks,
        re_scales,
        global_scale=re_g if re_g is not None else global_scale,
        out_dtype=torch.float32,
    )
    if not torch.equal(w.cpu(), w2.cpu()):
        max_abs = (w.cpu() - w2.cpu()).abs().max().item()
        raise AssertionError(
            f"FP4 re-pack of '{name}' drifted by {max_abs:g} on already-"
            f"quantised input — fp4_utils pack/unpack are not mutually "
            f"consistent for {fmt.kind}."
        )


def assert_matches_reference(
    our_dequant: Tensor,
    reference_dequant: Tensor,
    *,
    name: str = "<tensor>",
    atol: float = 1e-3,
    rtol: float = 1e-2,
) -> None:
    """Verify fp4_utils' dequant matches a ground-truth dequant of the same tensor.

    This is the real layout-faithfulness guard: the streaming repack tool
    passes the model's own dequantised weight (from transformers / the model's
    modeling code) as ``reference_dequant``. If the nibble order, block axis,
    or scale encoding assumed by fp4_utils is wrong for this producer, the
    decoded values diverge and this fires — a loud abort instead of silently
    writing corrupted weights.
    """
    a = our_dequant.detach().to(torch.float32).cpu()
    b = reference_dequant.detach().to(torch.float32).cpu()
    if a.shape != b.shape:
        raise AssertionError(
            f"FP4 layout check for '{name}': shape {tuple(a.shape)} != "
            f"reference {tuple(b.shape)}."
        )
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        max_abs = (a - b).abs().max().item()
        denom = b.abs().mean().clamp(min=1e-9)
        rel = ((a - b).abs().mean() / denom).item()
        raise AssertionError(
            f"FP4 dequant of '{name}' disagrees with the model's own dequant "
            f"(max abs {max_abs:g}, mean rel {rel:g}). fp4_utils' layout "
            "assumptions do not match this checkpoint's producer; do NOT "
            "repack in place."
        )
