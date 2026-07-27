"""Tests for the FP8 quant direction and the scale-key detection fix.

``fp8_utils`` only ever went FP8 → BF16. Two problems followed:

* a producer whose scale sibling is named something unrecognised (DeepSeek-V4
  uses a bare ``.scale``) fell through to a *bare cast*, silently dropping the
  128×128 block scaling and writing plausible-but-wrong weights;
* an edited tensor could not be written back as FP8 at all.

CPU only, synthetic tensors.
"""

import pytest
import torch
from safetensors.torch import save_file

from abliterix.core import fp8_utils as f8

_HAS_E8M0 = hasattr(torch, "float8_e8m0fnu")


# ---------------------------------------------------------------------------
# Scale-key detection (the silent-corruption bug)
# ---------------------------------------------------------------------------


def test_scale_attrs_covers_deepseek_bare_scale():
    """DeepSeek-V4 names it `.scale`; missing that meant a silent bare cast."""
    assert "scale" in f8._SCALE_ATTRS
    assert "weight_scale_inv" in f8._SCALE_ATTRS  # DeepSeek-V3 / MiniMax
    assert "weight_scale" in f8._SCALE_ATTRS  # per-tensor FP8


def _write_fp8_shard(
    path, key="layers.0.attn.wo_b", scale_suffix="scale", with_scale=True
):
    torch.manual_seed(0)
    w = torch.randn(256, 512) * 0.05
    fp8, scale = f8.quantize_to_fp8_blockwise(w, (128, 128))
    tensors = {f"{key}.weight": fp8}
    if with_scale:
        tensors[f"{key}.{scale_suffix}"] = scale
    save_file(tensors, str(path), metadata={"format": "pt"})
    return w, fp8, scale


def test_dequant_shard_applies_deepseek_style_scale(tmp_path):
    """The dequanted output must carry the block scaling, not a bare cast."""
    src, dst = tmp_path / "in.safetensors", tmp_path / "out.safetensors"
    w, fp8, scale = _write_fp8_shard(src)

    f8.dequant_safetensors_shard(src, dst, use_cuda=False)

    from safetensors import safe_open

    with safe_open(dst, framework="pt") as f:
        keys = list(f.keys())
        got = f.get_tensor("layers.0.attn.wo_b.weight").float()
    assert "layers.0.attn.wo_b.scale" not in keys  # folded in, then dropped

    want = f8.dequant_blockwise(fp8, scale, is_inv=True, out_dtype=torch.float32)
    assert torch.allclose(got, want, atol=1e-2)
    # A bare cast would be off by the block scale — make sure we are not that.
    bare = fp8.float()
    assert not torch.allclose(got, bare, atol=1e-2)


def test_dequant_shard_refuses_unscaled_fp8_instead_of_guessing(tmp_path):
    """No sibling scale => raise, because a bare cast corrupts silently."""
    src, dst = tmp_path / "in.safetensors", tmp_path / "out.safetensors"
    _write_fp8_shard(src, with_scale=False)
    with pytest.raises(ValueError, match="no sibling scale"):
        f8.dequant_safetensors_shard(src, dst, use_cuda=False)


def test_dequant_shard_allow_unscaled_opt_in(tmp_path):
    """Genuinely-unscaled checkpoints can still opt into the bare cast."""
    src, dst = tmp_path / "in.safetensors", tmp_path / "out.safetensors"
    _, fp8, _ = _write_fp8_shard(src, with_scale=False)
    f8.dequant_safetensors_shard(src, dst, use_cuda=False, allow_unscaled=True)

    from safetensors import safe_open

    with safe_open(dst, framework="pt") as f:
        got = f.get_tensor("layers.0.attn.wo_b.weight").float()
    assert torch.allclose(got, fp8.float(), atol=1e-2)


# ---------------------------------------------------------------------------
# The quant direction
# ---------------------------------------------------------------------------


def test_quantize_shapes_and_dtypes():
    w = torch.randn(256, 512) * 0.05
    fp8, scale = f8.quantize_to_fp8_blockwise(w, (128, 128))
    assert fp8.shape == w.shape and fp8.dtype == torch.float8_e4m3fn
    assert scale.shape == (2, 4)  # 256/128, 512/128
    if _HAS_E8M0:
        assert scale.dtype == torch.float8_e8m0fnu


def test_quantize_roundtrip_close():
    torch.manual_seed(1)
    w = torch.randn(256, 256) * 0.02
    fp8, scale = f8.quantize_to_fp8_blockwise(w, (128, 128))
    back = f8.dequant_blockwise(fp8, scale, is_inv=True, out_dtype=torch.float32)
    rel = (back - w).abs().mean() / w.abs().mean()
    # e4m3 has 3 mantissa bits, so a few percent is expected — far tighter
    # than FP4's ~10%.
    assert rel < 0.05, rel


def test_quantize_is_the_inverse_of_dequant_blockwise():
    """Round-tripping an already-FP8 tensor is a fixed point."""
    torch.manual_seed(2)
    w = torch.randn(256, 384) * 0.03
    fp8, scale = f8.quantize_to_fp8_blockwise(w, (128, 128))
    deq = f8.dequant_blockwise(fp8, scale, is_inv=True, out_dtype=torch.float32)
    fp8_2, scale_2 = f8.quantize_to_fp8_blockwise(deq, (128, 128))
    deq2 = f8.dequant_blockwise(fp8_2, scale_2, is_inv=True, out_dtype=torch.float32)
    assert torch.equal(deq, deq2)
    f8.assert_fp8_repack_idempotent(fp8, scale, name="w")


def test_power_of_two_scale_divides_exactly():
    """A power-of-two scale contributes no rounding of its own."""
    w = torch.randn(128, 128) * 0.1
    _, scale = f8.quantize_to_fp8_blockwise(w, (128, 128), power_of_two_scale=True)
    s = scale.float()
    assert torch.allclose(torch.log2(s), torch.round(torch.log2(s)), atol=1e-5)


def test_values_stay_inside_e4m3_range():
    w = torch.randn(128, 256) * 50.0  # large magnitudes
    fp8, scale = f8.quantize_to_fp8_blockwise(w, (128, 128))
    assert fp8.float().abs().max() <= 448.0


def test_non_multiple_dimensions_are_padded():
    """Shapes that are not a whole number of blocks still round-trip."""
    torch.manual_seed(3)
    w = torch.randn(200, 300) * 0.05
    fp8, scale = f8.quantize_to_fp8_blockwise(w, (128, 128))
    assert fp8.shape == (200, 300)
    assert scale.shape == (2, 3)
    back = f8.dequant_blockwise(fp8, scale, is_inv=True, out_dtype=torch.float32)
    assert back.shape == w.shape
    assert (back - w).abs().mean() / w.abs().mean() < 0.05


def test_zero_block_survives():
    w = torch.zeros(128, 128)
    fp8, scale = f8.quantize_to_fp8_blockwise(w, (128, 128))
    back = f8.dequant_blockwise(fp8, scale, is_inv=True, out_dtype=torch.float32)
    assert torch.equal(back, torch.zeros_like(back))


def test_quantize_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        f8.quantize_to_fp8_blockwise(torch.randn(2, 4, 8))


@pytest.mark.skipif(not _HAS_E8M0, reason="torch lacks float8_e8m0fnu")
def test_e8m0_scale_decodes_as_power_of_two():
    """torch decodes float8_e8m0fnu as 2^(byte-127) — the DeepSeek convention."""
    b = torch.tensor([120, 127, 130], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    assert b.to(torch.float32).tolist() == [2.0**-7, 1.0, 2.0**3]


@pytest.mark.skipif(not _HAS_E8M0, reason="torch lacks float8_e8m0fnu")
def test_edit_then_repack_preserves_stored_dtypes():
    """An edited FP8 tensor re-packs into the producer's dtypes, not float32."""
    torch.manual_seed(4)
    w = torch.randn(256, 256) * 0.04
    fp8, scale = f8.quantize_to_fp8_blockwise(w, (128, 128))
    assert scale.dtype == torch.float8_e8m0fnu

    deq = f8.dequant_blockwise(fp8, scale, is_inv=True, out_dtype=torch.float32)
    d = torch.randn(256)
    d = d / d.norm()
    edited = deq - 1.5 * d.unsqueeze(1) * (d @ deq).unsqueeze(0)  # output-side ablation

    re_w, re_s = f8.quantize_to_fp8_blockwise(
        edited, (128, 128), weight_dtype=fp8.dtype, scale_dtype=scale.dtype
    )
    assert re_w.dtype == torch.float8_e4m3fn and re_s.dtype == torch.float8_e8m0fnu
    assert re_w.shape == fp8.shape and re_s.shape == scale.shape

    back = f8.dequant_blockwise(re_w, re_s, is_inv=True, out_dtype=torch.float32)
    rel = (back - edited).abs().mean() / edited.abs().mean()
    assert rel < 0.05, rel
    assert not torch.allclose(back, deq, atol=1e-3)  # the edit landed
