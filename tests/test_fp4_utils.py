"""Tests for abliterix.core.fp4_utils — MXFP4 / NVFP4 pack + unpack kernels.

All CPU, small synthetic tensors — no GPU, no model download. Correctness of
these kernels is the foundation of the offline FP4 edit-and-repack path, so
they are checked hard: exact round-trip on representable values, idempotent
re-pack, and the layout-faithfulness guard.
"""

import pytest
import torch

from abliterix.core import fp4_utils as f4

_HAS_E4M3 = hasattr(torch, "float8_e4m3fn")


# ---------------------------------------------------------------------------
# E2M1 element format
# ---------------------------------------------------------------------------


def test_e2m1_levels_exact_roundtrip():
    """Every signed E2M1 level encodes and decodes to itself."""
    levels = list(f4.E2M1_LEVELS)
    signed = torch.tensor(levels + [-v for v in levels], dtype=torch.float32)
    codes = f4.quantize_to_e2m1_codes(signed)
    back = f4.e2m1_codes_to_values(codes)
    assert torch.equal(back, signed.where(signed != 0, torch.zeros_like(signed)))


def test_e2m1_rounds_to_nearest_level():
    # 0.6 -> 0.5, 0.8 -> 1.0, 2.4 -> 2.0, 2.6 -> 3.0, 100 -> 6 (saturate)
    x = torch.tensor([0.6, 0.8, 2.4, 2.6, 100.0, -100.0])
    vals = f4.e2m1_codes_to_values(f4.quantize_to_e2m1_codes(x))
    assert vals.tolist() == [0.5, 1.0, 2.0, 3.0, 6.0, -6.0]


def test_e2m1_negative_zero_canonicalised():
    codes = f4.quantize_to_e2m1_codes(torch.tensor([-0.0, 0.0, -0.1]))
    # all three round to the +0 code (0), never the -0 pattern (0b1000 = 8).
    assert codes.tolist() == [0, 0, 0]


# ---------------------------------------------------------------------------
# Nibble packing
# ---------------------------------------------------------------------------


def test_nibble_pack_unpack_roundtrip():
    codes = torch.randint(0, 16, (4, 32), dtype=torch.uint8)
    packed = f4.pack_nibbles(codes)
    assert packed.shape == (4, 16)
    assert torch.equal(f4.unpack_nibbles(packed), codes)


def test_nibble_order_low_then_high():
    codes = torch.tensor([[0x3, 0xA]], dtype=torch.uint8)  # lo=3, hi=A
    packed = f4.pack_nibbles(codes)
    assert packed.item() == (0xA << 4) | 0x3  # 0xA3


def test_nibble_pack_odd_axis_rejected():
    with pytest.raises(ValueError):
        f4.pack_nibbles(torch.zeros(3, dtype=torch.uint8))


# ---------------------------------------------------------------------------
# MXFP4
# ---------------------------------------------------------------------------


def test_mxfp4_shapes():
    w = torch.randn(8, 64)  # 64 = 2 blocks of 32
    blocks, scales = f4.quantize_to_mxfp4(w, block_size=32)
    assert blocks.shape == (8, 2, 16)  # block//2 bytes
    assert scales.shape == (8, 2)
    assert blocks.dtype == torch.uint8 and scales.dtype == torch.uint8


def test_mxfp4_roundtrip_close():
    torch.manual_seed(0)
    w = torch.randn(16, 128) * 0.05  # weight-like magnitudes
    blocks, scales = f4.quantize_to_mxfp4(w, 32)
    w_hat = f4.dequantize_mxfp4(blocks, scales, 32, out_dtype=torch.float32)
    assert w_hat.shape == w.shape
    # 4-bit within-block relative error: each block's max element within one
    # E2M1 step of the top of range. Median relative error should be modest.
    rel = (w_hat - w).abs() / w.abs().clamp(min=1e-6)
    assert rel.median() < 0.15


def test_mxfp4_idempotent_repack_values():
    """Re-packing already-quantised values reproduces the exact values.

    Byte-identity is NOT guaranteed: a block whose peak rounds to level 3
    admits a tighter power-of-two scale (3·2^E == 6·2^(E-1)), so bytes may
    change while every decoded weight is identical. Value-identity is the
    invariant that matters for a non-destructive round-trip.
    """
    torch.manual_seed(1)
    w = torch.randn(4, 96) * 0.1
    blocks, scales = f4.quantize_to_mxfp4(w, 32)
    w_hat = f4.dequantize_mxfp4(blocks, scales, 32, out_dtype=torch.float32)
    re_blocks, re_scales = f4.quantize_to_mxfp4(w_hat, 32)
    w_hat2 = f4.dequantize_mxfp4(re_blocks, re_scales, 32, out_dtype=torch.float32)
    assert torch.equal(w_hat, w_hat2)


def test_mxfp4_scale_is_power_of_two_and_bounds_range():
    w = torch.randn(2, 32) * 3.0
    blocks, scales = f4.quantize_to_mxfp4(w, 32)
    w_hat = f4.dequantize_mxfp4(blocks, scales, 32, out_dtype=torch.float32)
    # No element should exceed 6 * scale; scale is 2**(byte-127).
    scale = torch.pow(2.0, scales.float() - 127.0)
    assert (w_hat.abs() <= 6.0 * scale.unsqueeze(-1) + 1e-4).all()


def test_mxfp4_all_zero_block():
    w = torch.zeros(1, 32)
    blocks, scales = f4.quantize_to_mxfp4(w, 32)
    w_hat = f4.dequantize_mxfp4(blocks, scales, 32, out_dtype=torch.float32)
    assert torch.equal(w_hat, torch.zeros_like(w_hat))


def test_mxfp4_non_divisible_axis_rejected():
    with pytest.raises(ValueError):
        f4.quantize_to_mxfp4(torch.randn(2, 30), 32)


# ---------------------------------------------------------------------------
# MXFP4 MSE-optimal block-scale search
# ---------------------------------------------------------------------------


def test_mxfp4_scale_search_default_is_amax():
    """scale_search=0 (default) is the plain amax rule — bytes unchanged."""
    torch.manual_seed(5)
    w = torch.randn(4, 128) * 0.1
    b0, s0 = f4.quantize_to_mxfp4(w, 32)
    b_def, s_def = f4.quantize_to_mxfp4(w, 32, scale_search=0)
    assert torch.equal(b0, b_def) and torch.equal(s0, s_def)


def test_mxfp4_scale_search_reduces_error():
    """Searching a tighter power-of-two scale lowers reconstruction MSE."""
    torch.manual_seed(6)
    # Heavy-tailed block distribution — where clipping-for-resolution helps.
    w = torch.distributions.Laplace(0.0, 0.03).sample((32, 4096))

    def mse(S):
        b, s = f4.quantize_to_mxfp4(w, 32, scale_search=S)
        wh = f4.dequantize_mxfp4(b, s, 32, out_dtype=torch.float32)
        return ((wh - w) ** 2).mean().item()

    m0, m1 = mse(0), mse(1)
    assert m1 < m0, f"scale search did not help: {m0} -> {m1}"
    # S=1 captures the gain; deeper search should not be worse.
    assert mse(2) <= m1 + 1e-12


def test_mxfp4_scale_search_preserves_value_idempotency():
    """Re-packing already-quantised values with search on is a fixed point."""
    torch.manual_seed(7)
    w = torch.randn(4, 128) * 0.1
    b, s = f4.quantize_to_mxfp4(w, 32)  # on-grid after this
    w_hat = f4.dequantize_mxfp4(b, s, 32, out_dtype=torch.float32)
    b2, s2 = f4.quantize_to_mxfp4(w_hat, 32, scale_search=3)
    w_hat2 = f4.dequantize_mxfp4(b2, s2, 32, out_dtype=torch.float32)
    assert torch.equal(w_hat, w_hat2)


def test_mxfp4_scale_search_does_not_clip_on_grid_data():
    """On already-quantised data the amax-tight scale is uniquely MSE-0, so the
    search returns exactly the no-search result — it never needlessly clips."""
    torch.manual_seed(8)
    w = torch.randn(2, 32) * 0.1
    b, s = f4.quantize_to_mxfp4(w, 32)
    w_hat = f4.dequantize_mxfp4(b, s, 32, out_dtype=torch.float32)
    # Re-quantising w_hat: search (S=2) must pick the same scale as no-search
    # (S=0). (Both may differ from `s` itself — an equal-value tighter scale is
    # legitimate — which is why this compares S=2 vs S=0, not against `s`.)
    _, s_plain = f4.quantize_to_mxfp4(w_hat, 32, scale_search=0)
    _, s_search = f4.quantize_to_mxfp4(w_hat, 32, scale_search=2)
    assert torch.equal(s_plain, s_search)


# ---------------------------------------------------------------------------
# Ground truth: agreement with transformers' own MXFP4 dequant
#
# Every other test here is self-consistent (our packer feeding our unpacker),
# which proves nothing about matching what gpt-oss actually wrote to disk.
# This one pins our element decoding — codebook order, nibble order, ue8m0
# scale — against the reference implementation.
# ---------------------------------------------------------------------------

try:
    from transformers.integrations.mxfp4 import (
        _convert_moe_packed_tensors as _REF_MXFP4_DEQUANT,
    )
except Exception:  # pragma: no cover - transformers may lack the integration
    _REF_MXFP4_DEQUANT = None


@pytest.mark.skipif(
    _REF_MXFP4_DEQUANT is None, reason="transformers MXFP4 reference unavailable"
)
def test_mxfp4_decoding_matches_transformers_reference():
    """Our element decoding is bit-identical to transformers' MXFP4 dequant.

    The reference additionally applies a final ``transpose(1, 2)`` (a MoE
    layout convention, handled in fp4_repack, not here), so we compare after
    undoing it. Random bytes + random scales exercise the whole codebook.
    """
    torch.manual_seed(4)
    E, rows, G, B = 2, 6, 4, 16  # block_size 32 → 16 bytes/block
    blocks = torch.randint(0, 256, (E, rows, G, B), dtype=torch.uint8)
    scales = torch.randint(100, 150, (E, rows, G), dtype=torch.uint8)

    mine = f4.dequantize_mxfp4(blocks, scales, 32, out_dtype=torch.float32)
    ref = _REF_MXFP4_DEQUANT(blocks, scales, dtype=torch.float32)

    # Reference shape is (E, K, rows); ours preserves on-disk (E, rows, K).
    assert mine.shape == (E, rows, G * B * 2)
    assert ref.shape == (E, G * B * 2, rows)
    assert torch.equal(mine, ref.transpose(1, 2).contiguous())


@pytest.mark.skipif(
    _REF_MXFP4_DEQUANT is None, reason="transformers MXFP4 reference unavailable"
)
def test_mxfp4_codebook_matches_reference_values():
    """Our E2M1 level table matches transformers' FP4_VALUES, code for code."""
    from transformers.integrations.mxfp4 import FP4_VALUES

    codes = torch.arange(16, dtype=torch.uint8)
    ours = f4.e2m1_codes_to_values(codes).tolist()
    # FP4_VALUES[8] is -0.0; we canonicalise to +0.0 (same numeric value).
    assert ours == [abs(v) if v == 0 else v for v in FP4_VALUES]


# ---------------------------------------------------------------------------
# NVFP4
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_E4M3, reason="torch build lacks float8_e4m3fn")
def test_nvfp4_shapes_and_roundtrip():
    torch.manual_seed(2)
    w = torch.randn(8, 64) * 0.05  # 64 = 4 blocks of 16
    blocks, bscale, g = f4.quantize_to_nvfp4(w, 16)
    assert blocks.shape == (8, 4, 8)  # 16//2 bytes
    assert bscale.shape == (8, 4)
    assert g.ndim == 0
    w_hat = f4.dequantize_nvfp4(blocks, bscale, g, 16, out_dtype=torch.float32)
    assert w_hat.shape == w.shape
    rel = (w_hat - w).abs() / w.abs().clamp(min=1e-6)
    assert rel.median() < 0.2


@pytest.mark.skipif(not _HAS_E4M3, reason="torch build lacks float8_e4m3fn")
def test_nvfp4_idempotent_repack_values():
    torch.manual_seed(3)
    w = torch.randn(4, 48) * 0.1
    blocks, bscale, g = f4.quantize_to_nvfp4(w, 16)
    w_hat = f4.dequantize_nvfp4(blocks, bscale, g, 16, out_dtype=torch.float32)
    # Re-quantise with the SAME global scale (as the streaming tool does).
    re_blocks, re_bscale, _ = f4.quantize_to_nvfp4(w_hat, 16, global_scale=g)
    w_hat2 = f4.dequantize_nvfp4(re_blocks, re_bscale, g, 16, out_dtype=torch.float32)
    assert torch.equal(w_hat, w_hat2)


# ---------------------------------------------------------------------------
# Format detection + faithfulness guard
# ---------------------------------------------------------------------------


def test_detect_fp4_format():
    assert f4.detect_fp4_format({"quant_method": "mxfp4"}) == f4.Fp4Format("mxfp4", 32)
    assert f4.detect_fp4_format({"quant_method": "nvfp4"}) == f4.Fp4Format("nvfp4", 16)
    assert f4.detect_fp4_format(
        {"quant_method": "modelopt_fp4", "group_size": 32}
    ) == f4.Fp4Format("nvfp4", 32)
    assert f4.detect_fp4_format({"quant_method": "fp8"}) is None
    assert f4.detect_fp4_format(None) is None


def test_assert_repack_idempotent_passes():
    w = torch.randn(4, 64) * 0.1
    blocks, scales = f4.quantize_to_mxfp4(w, 32)
    f4.assert_repack_idempotent(blocks, scales, f4.Fp4Format("mxfp4", 32), name="w")


def test_assert_matches_reference_passes_and_fires():
    w = torch.randn(4, 64) * 0.1
    blocks, scales = f4.quantize_to_mxfp4(w, 32)
    our = f4.dequantize_mxfp4(blocks, scales, 32, out_dtype=torch.float32)
    # Matching reference (the model's own dequant would be exactly this).
    f4.assert_matches_reference(our, our.clone(), name="w")
    # A wrong-layout dequant (e.g. swapped nibble order) diverges → must fire.
    wrong = f4.dequantize_mxfp4(blocks.flip(-1), scales, 32, out_dtype=torch.float32)
    with pytest.raises(AssertionError):
        f4.assert_matches_reference(wrong, our, name="w")
