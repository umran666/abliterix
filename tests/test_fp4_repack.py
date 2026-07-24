"""Tests for abliterix.core.fp4_repack — offline FP4 edit-and-repack bake path.

Builds a tiny synthetic MXFP4 checkpoint on disk (an expert down_proj as
``_blocks``/``_scales``, a plain BF16 o_proj, and copied-through tensors),
replays hand-built edits, and verifies the output. CPU only, no model.
"""

import json

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from safetensors.torch import save_file

from abliterix.core import fp4_utils as f4
from abliterix.core import fp4_repack as rp
from abliterix.types import DecayKernel, SteeringProfile, WeightNorm
from abliterix.weight_transforms import apply_ega_projection, resolve_ega_axis


# ---------------------------------------------------------------------------
# apply_tensor_edit parity with the shared pure helpers
# ---------------------------------------------------------------------------


def test_apply_tensor_edit_ega_matches_helper():
    torch.manual_seed(0)
    W = torch.randn(4, 8, 64)  # (E, hidden=8, inter=64)
    d = torch.randn(8)
    edit = rp.TensorEdit(
        logical_name="x",
        kind="ega",
        direction=d,
        strength=1.3,
        preserve_row_norm=True,
        hidden_dim=8,
        transposed=False,
    )
    got = rp.apply_tensor_edit(W, edit)
    axis = resolve_ega_axis(W.shape, 8)
    exp = apply_ega_projection(
        W, d, strength=1.3, axis_is_in=axis, preserve_row_norm=True
    )
    assert torch.allclose(got, exp, atol=1e-6)


def test_apply_tensor_edit_direct_output_side_matches_engine_math():
    torch.manual_seed(1)
    W = torch.randn(8, 8)  # square o_proj: engine prefers output-side
    d = torch.randn(8)
    edit = rp.TensorEdit(
        logical_name="o",
        kind="direct",
        direction=d,
        strength=0.7,
        preserve_row_norm=True,
        projection_side="output",
    )
    got = rp.apply_tensor_edit(W, edit)

    v = d / d.norm()
    W32 = W.to(torch.float32)
    W_new = W32 - 0.7 * v.unsqueeze(1) * (v @ W32).unsqueeze(0)
    orig = torch.linalg.vector_norm(W32, dim=1, keepdim=True)
    new = torch.linalg.vector_norm(W_new, dim=1, keepdim=True).clamp(min=1e-8)
    W_new = W_new * (orig / new)
    assert torch.allclose(got, W_new, atol=1e-6)


def test_apply_tensor_edit_moves_direction_to_weight_device():
    """Direction may live on CPU (loaded plan) while the weight is on GPU.

    Regression for a GPU-only crash found during the gpt-oss-20b bake: the
    direct branch multiplied a CPU direction against a CUDA weight. CI has no
    CUDA, so we assert at the source level that both branches move the
    direction to the WEIGHT's device (``device=dev`` derived from
    ``W32.device``) rather than hard-coding it, and that the happy path still
    works when both are on CPU.
    """
    import inspect

    src = inspect.getsource(rp.apply_tensor_edit)
    assert "dev = W32.device" in src
    assert src.count("device=dev") >= 2  # direct-standard + advanced branches

    for edit in (
        rp.TensorEdit(
            "o", "direct", torch.randn(8), 0.7, True, projection_side="output"
        ),
        rp.TensorEdit("e", "ega", torch.randn(8), 1.0, True, hidden_dim=8),
    ):
        W_in = torch.randn(4, 8, 8) if edit.kind == "ega" else torch.randn(8, 8)
        out = rp.apply_tensor_edit(W_in, edit)
        assert out.device == W_in.device


def test_apply_tensor_edit_rejects_wrong_rank():
    with pytest.raises(ValueError):
        rp.apply_tensor_edit(
            torch.randn(8, 8),
            rp.TensorEdit("x", "ega", torch.randn(8), 1.0, True, hidden_dim=8),
        )


# ---------------------------------------------------------------------------
# FP4 key resolution + name normalisation
# ---------------------------------------------------------------------------


def test_resolve_fp4_keys_mxfp4():
    present = {"m.down_proj_blocks", "m.down_proj_scales", "m.o_proj.weight"}
    ks = rp.resolve_fp4_keys("m.down_proj", present, f4.Fp4Format("mxfp4", 32))
    assert ks == rp._Fp4KeySet("m.down_proj_blocks", "m.down_proj_scales")
    # Plain weight → not FP4.
    assert rp.resolve_fp4_keys("m.o_proj", present, f4.Fp4Format("mxfp4", 32)) is None


def test_resolve_fp4_keys_nvfp4_with_global():
    present = {"w", "w_scale", "w_scale_2"}
    ks = rp.resolve_fp4_keys("w", present, f4.Fp4Format("nvfp4", 16))
    assert ks == rp._Fp4KeySet("w", "w_scale", "w_scale_2")


def test_normalize_param_name():
    assert (
        rp.normalize_param_name(
            "base_model.model.model.layers.0.self_attn.o_proj.base_layer.weight"
        )
        == "model.layers.0.self_attn.o_proj.weight"
    )
    # Idempotent on canonical names.
    assert rp.normalize_param_name("model.layers.0.mlp.experts.down_proj") == (
        "model.layers.0.mlp.experts.down_proj"
    )


# ---------------------------------------------------------------------------
# Decay-kernel strength (mirror of the engine)
# ---------------------------------------------------------------------------


def test_layer_strength_gating_and_kernels():
    sp = SteeringProfile(
        max_weight=4.0,
        max_weight_position=0.0,
        min_weight=1.0,
        min_weight_distance=10.0,
    )
    # At the peak, linear returns max_weight.
    assert rp._layer_strength(sp, 0, DecayKernel.LINEAR) == pytest.approx(4.0)
    # Beyond falloff → None (skip).
    assert rp._layer_strength(sp, 11, DecayKernel.LINEAR) is None
    # Exactly-zero strength → None.
    spz = SteeringProfile(
        max_weight=0.0, max_weight_position=0.0, min_weight=0.0, min_weight_distance=5.0
    )
    assert rp._layer_strength(spz, 0, DecayKernel.LINEAR) is None


# ---------------------------------------------------------------------------
# Plan (de)serialisation
# ---------------------------------------------------------------------------


def test_save_load_plan_roundtrip(tmp_path):
    edits = [
        rp.TensorEdit(
            "a", "ega", torch.randn(8), 1.0, True, hidden_dim=8, transposed=True
        ),
        rp.TensorEdit(
            "b", "direct", torch.randn(8), 0.5, False, projection_side="output"
        ),
    ]
    p = tmp_path / "plan.pt"
    rp.save_plan(edits, p)
    back = rp.load_plan(p)
    assert [e.logical_name for e in back] == ["a", "b"]
    assert back[0].transposed is True and back[0].hidden_dim == 8
    assert torch.equal(back[1].direction, edits[1].direction)


# ---------------------------------------------------------------------------
# End-to-end streaming bake on a synthetic MXFP4 checkpoint
# ---------------------------------------------------------------------------


def _build_mxfp4_checkpoint(path, E=4, hidden=8, inter=64):
    """Write a 1-shard MXFP4 model: fused expert down_proj + plain o_proj + norm."""
    path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)

    down = torch.randn(E, hidden, inter) * 0.1  # (E, hidden, inter)
    blocks, scales = f4.quantize_to_mxfp4(down, 32)
    o_proj = (torch.randn(hidden, hidden) * 0.1).to(torch.bfloat16)
    norm = torch.ones(hidden, dtype=torch.bfloat16)

    tensors = {
        "model.layers.0.mlp.experts.down_proj_blocks": blocks,
        "model.layers.0.mlp.experts.down_proj_scales": scales,
        "model.layers.0.self_attn.o_proj.weight": o_proj,
        "model.norm.weight": norm,
    }
    save_file(tensors, str(path / "model.safetensors"), metadata={"format": "pt"})
    (path / "config.json").write_text(
        json.dumps(
            {"model_type": "gpt_oss", "quantization_config": {"quant_method": "mxfp4"}}
        )
    )
    (path / "tokenizer_config.json").write_text("{}")
    # Return the source dequantised down_proj for assertions.
    return f4.dequantize_mxfp4(blocks, scales, 32, out_dtype=torch.float32)


def test_streaming_bake_end_to_end(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    E, hidden, inter = 4, 8, 64
    src_down = _build_mxfp4_checkpoint(src, E, hidden, inter)

    d_ega = torch.randn(hidden)
    d_o = torch.randn(hidden)
    edits = [
        rp.TensorEdit(
            "model.layers.0.mlp.experts.down_proj",
            "ega",
            d_ega,
            1.5,
            True,
            hidden_dim=hidden,
            transposed=False,
        ),
        rp.TensorEdit(
            "model.layers.0.self_attn.o_proj.weight",
            "direct",
            d_o,
            0.8,
            True,
            projection_side="output",
        ),
    ]

    stats = rp.abliterate_fp4_to_disk(src, dst, edits, use_cuda=False, verbose=False)

    assert stats.fp4_edited == 1
    assert stats.dense_edited == 1
    assert stats.copied == 1  # model.norm.weight
    assert not stats.skipped_edits

    # --- Read back the baked checkpoint ---
    from safetensors import safe_open

    with safe_open(dst / "model.safetensors", framework="pt") as f:
        out = {k: f.get_tensor(k) for k in f.keys()}

    # FP4 expert tensor: dequant of output ≈ projected source (within requant err).
    baked_down = f4.dequantize_mxfp4(
        out["model.layers.0.mlp.experts.down_proj_blocks"],
        out["model.layers.0.mlp.experts.down_proj_scales"],
        32,
        out_dtype=torch.float32,
    )
    # Compare in the model's LOGICAL orientation — the bake edits (E, K, rows),
    # matching transformers' reference dequant, not the raw on-disk element
    # order. This matters beyond the projection axis: preserve_row_norm
    # normalises along dim=-1, so the orientation picks which vectors get
    # renormalised.
    baked_down = baked_down.transpose(1, 2).contiguous()
    src_logical = src_down.transpose(1, 2).contiguous()
    axis = resolve_ega_axis(src_logical.shape, hidden)
    expected_down = apply_ega_projection(
        src_logical, d_ega, strength=1.5, axis_is_in=axis, preserve_row_norm=True
    )
    src_down = src_logical
    # MXFP4 is a 4-bit format: E2M1's 8 magnitude levels are spaced ~1.5-2x
    # apart, so re-quantising an off-grid (edited) tensor inherently carries
    # ~10-15% mean relative error. This is the SAME regime the base weights
    # already live in, not new error stacked on top — but it is exactly the
    # "requant vs KL" quantity the bake tool exists to surface. Assert it is
    # bounded and non-trivial (a 4-bit round-trip, not a lossless copy).
    rel = float((baked_down - expected_down).abs().mean() / expected_down.abs().mean())
    assert 0.01 < rel < 0.2, f"requant drift out of expected 4-bit band: {rel}"
    # The tool reports the same error it actually incurred.
    reported = stats.requant_rel_err["model.layers.0.mlp.experts.down_proj"]
    assert reported == pytest.approx(rel, abs=0.02)

    # The edit actually changed the weights (not a no-op copy).
    assert not torch.allclose(baked_down, src_down, atol=1e-3)

    # Dense o_proj: exact projection (BF16 storage round-trip only).
    v = d_o / d_o.norm()
    o_src = torch.randn(1)  # placeholder to satisfy linter; real value below
    with safe_open(src / "model.safetensors", framework="pt") as f:
        o_src = f.get_tensor("model.layers.0.self_attn.o_proj.weight").float()
    W_new = o_src - 0.8 * v.unsqueeze(1) * (v @ o_src).unsqueeze(0)
    onorm = torch.linalg.vector_norm(o_src, dim=1, keepdim=True)
    nnorm = torch.linalg.vector_norm(W_new, dim=1, keepdim=True).clamp(min=1e-8)
    W_new = W_new * (onorm / nnorm)
    baked_o = out["model.layers.0.self_attn.o_proj.weight"].float()
    assert torch.allclose(baked_o, W_new.to(torch.bfloat16).float(), atol=1e-2)

    # Copied-through tensor is byte-identical.
    assert torch.equal(
        out["model.norm.weight"], torch.ones(hidden, dtype=torch.bfloat16)
    )

    # config.json keeps quantization_config (still FP4).
    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["quantization_config"]["quant_method"] == "mxfp4"

    # index.json references every output tensor.
    idx = json.loads((dst / "model.safetensors.index.json").read_text())
    assert set(idx["weight_map"]) == set(out)

    # aux files copied.
    assert (dst / "tokenizer_config.json").exists()


def test_dequant_yields_model_logical_orientation():
    """A 4-D packed MoE tensor is rotated to the orientation the model sees.

    transformers' reference dequant ends with ``transpose(1, 2)``, so the
    in-memory weight is ``(E, K, rows)`` while the packed bytes are
    ``(E, rows, n_blocks, bytes)``. The steering plan records axis semantics
    against the in-memory tensor, so the bake path must match it — otherwise
    EGA projects the wrong axis (invisibly on gpt-oss, where hidden ==
    intermediate makes both axes equal length).
    """
    torch.manual_seed(20)
    E, rows, K = 2, 6, 128
    w_ondisk = torch.randn(E, rows, K) * 0.1
    blocks, scales = f4.quantize_to_mxfp4(w_ondisk, 32)
    assert blocks.dim() == 4  # (E, rows, n_blocks, bytes)

    keys = rp._Fp4KeySet("w_blocks", "w_scales")
    got = rp._dequant_fp4_tensor(
        {"w_blocks": blocks, "w_scales": scales}, keys, f4.Fp4Format("mxfp4", 32)
    )
    # Logical orientation is the transpose of the on-disk element order.
    assert got.shape == (E, K, rows)
    raw = f4.dequantize_mxfp4(blocks, scales, 32, out_dtype=torch.float32)
    assert torch.equal(got, raw.transpose(1, 2).contiguous())


def test_requant_restores_ondisk_orientation_roundtrip():
    """logical → on-disk → logical is a fixed point (no silent axis flip)."""
    torch.manual_seed(21)
    w_ondisk = torch.randn(2, 6, 128) * 0.1
    blocks, scales = f4.quantize_to_mxfp4(w_ondisk, 32)
    src = {"w_blocks": blocks, "w_scales": scales}
    keys = rp._Fp4KeySet("w_blocks", "w_scales")
    fmt = f4.Fp4Format("mxfp4", 32)

    logical = rp._dequant_fp4_tensor(src, keys, fmt)
    repacked = rp._requant_fp4_tensor(logical, keys, fmt, src)
    back = rp._dequant_fp4_tensor({**src, **repacked}, keys, fmt)
    assert back.shape == logical.shape
    assert torch.equal(back, logical)  # on-grid → exact round-trip
    # And the packed bytes keep the on-disk shape.
    assert repacked["w_blocks"].shape == blocks.shape


def test_streaming_bake_scale_search_lowers_requant_error(tmp_path):
    """The bake's default MSE scale search reduces reported requant error."""
    src = tmp_path / "ss_src"
    _build_mxfp4_checkpoint(src, E=8, hidden=16, inter=128)
    edit = rp.TensorEdit(
        "model.layers.0.mlp.experts.down_proj",
        "ega",
        torch.randn(16),
        1.5,
        True,
        hidden_dim=16,
        transposed=False,
    )
    key = "model.layers.0.mlp.experts.down_proj"

    s_off = rp.abliterate_fp4_to_disk(
        src, tmp_path / "ss_off", [edit], use_cuda=False, verbose=False, scale_search=0
    )
    s_on = rp.abliterate_fp4_to_disk(
        src, tmp_path / "ss_on", [edit], use_cuda=False, verbose=False, scale_search=2
    )
    assert s_on.requant_rel_err[key] <= s_off.requant_rel_err[key]


def test_streaming_bake_reports_unmatched_edits(tmp_path):
    src = tmp_path / "src2"
    dst = tmp_path / "dst2"
    _build_mxfp4_checkpoint(src)
    edits = [
        rp.TensorEdit(
            "model.layers.99.does.not.exist",
            "direct",
            torch.randn(8),
            1.0,
            True,
            projection_side="output",
        )
    ]
    stats = rp.abliterate_fp4_to_disk(src, dst, edits, use_cuda=False, verbose=False)
    assert stats.fp4_edited == 0 and stats.dense_edited == 0


# ---------------------------------------------------------------------------
# record_steering_plan via a mock engine
# ---------------------------------------------------------------------------


class _TinyAttn(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class _TinyLayer(nn.Module):
    def __init__(self, hidden, inter, E):
        super().__init__()
        self.self_attn = _TinyAttn(hidden)
        self.experts_down = nn.Parameter(torch.randn(E, hidden, inter))


class _TinyModel(nn.Module):
    def __init__(self, n_layers, hidden, inter, E):
        super().__init__()
        self.layers = nn.ModuleList(
            [_TinyLayer(hidden, inter, E) for _ in range(n_layers)]
        )


def _mock_engine(n_layers=2, hidden=8, inter=64, E=4):
    model = _TinyModel(n_layers, hidden, inter, E)
    layers = list(model.layers)

    def steerable_modules(idx):
        return {"attn.o_proj": [layers[idx].self_attn.o_proj]}

    def locate_fused(layer):
        return layer.experts_down

    return SimpleNamespace(
        model=model,
        transformer_layers=layers,
        steerable_modules=steerable_modules,
        _locate_fused_weights=locate_fused,
        has_expert_routing=lambda: True,
        _fused_down_proj_transposed=False,
    )


def _cfg(kernel=DecayKernel.LINEAR, norm=WeightNorm.FULL, discriminative=False):
    return SimpleNamespace(
        steering=SimpleNamespace(
            decay_kernel=kernel,
            weight_normalization=norm,
            direct_transform="standard",
            discriminative_layer_selection=discriminative,
        )
    )


def test_record_steering_plan_captures_direct_and_ega():
    eng = _mock_engine(n_layers=2, hidden=8, inter=64, E=4)
    svs = torch.randn(3, 8)  # (n_layers+1, hidden)
    profiles = {
        "attn.o_proj": SteeringProfile(2.0, 0.0, 2.0, 100.0),
        "mlp.down_proj": SteeringProfile(3.0, 0.0, 3.0, 100.0),
    }
    edits = rp.record_steering_plan(eng, svs, None, profiles, _cfg(), None)

    ega = [e for e in edits if e.kind == "ega"]
    direct = [e for e in edits if e.kind == "direct"]
    assert len(ega) == 2 and len(direct) == 2  # one per layer each

    # Names resolved to canonical parameter paths.
    assert any(e.logical_name.endswith("experts_down") for e in ega)
    assert any(e.logical_name.endswith("o_proj.weight") for e in direct)

    # EGA edit carries hidden_dim + transposed for offline axis resolution.
    assert all(e.hidden_dim == 8 for e in ega)
    # Direct edit resolved to output-side (square o_proj, engine precedence).
    assert all(e.projection_side == "output" for e in direct)
    # Strengths from the profiles.
    assert all(e.strength == pytest.approx(3.0) for e in ega)
    assert all(e.strength == pytest.approx(2.0) for e in direct)


def test_record_steering_plan_respects_discriminative_layers():
    eng = _mock_engine(n_layers=3, hidden=8, inter=64, E=4)
    svs = torch.randn(4, 8)
    profiles = {"mlp.down_proj": SteeringProfile(1.0, 0.0, 1.0, 100.0)}
    edits = rp.record_steering_plan(eng, svs, None, profiles, _cfg(), {1})
    assert len(edits) == 1  # only layer 1


def test_record_from_trial_resolves_global_vector():
    """record_*_from_trial uses the interpolated global vector for every layer."""
    eng = _mock_engine(n_layers=2, hidden=8, inter=64, E=4)
    svs = torch.randn(3, 8)
    profiles = {"mlp.down_proj": SteeringProfile(2.0, 0.0, 2.0, 100.0)}
    edits = rp.record_steering_plan_from_trial(
        eng, svs, 0.5, profiles, _cfg(), benign_states=None, target_states=None
    )
    from abliterix.core.steering import resolve_global_vector

    gv = resolve_global_vector(svs, 0.5)
    # Every EGA edit uses the single global direction, not per-layer vectors.
    ega = [e for e in edits if e.kind == "ega"]
    assert len(ega) == 2
    for e in ega:
        assert torch.allclose(e.direction, gv.to(torch.float32), atol=1e-6)


def test_cli_end_to_end(tmp_path):
    """abliterate_fp4.main() bakes a plan against a synthetic FP4 checkpoint."""
    from abliterix.scripts import abliterate_fp4

    src = tmp_path / "cli_src"
    dst = tmp_path / "cli_dst"
    _build_mxfp4_checkpoint(src, E=4, hidden=8, inter=64)
    edits = [
        rp.TensorEdit(
            "model.layers.0.mlp.experts.down_proj",
            "ega",
            torch.randn(8),
            1.2,
            True,
            hidden_dim=8,
            transposed=False,
        ),
    ]
    plan_path = tmp_path / "plan.pt"
    rp.save_plan(edits, plan_path)

    rc = abliterate_fp4.main([str(src), str(plan_path), str(dst), "--cpu", "--quiet"])
    assert rc == 0
    assert (dst / "model.safetensors").exists()
    assert (dst / "config.json").exists()
    assert (dst / "model.safetensors.index.json").exists()


def test_cli_missing_inputs_return_error_codes(tmp_path):
    from abliterix.scripts import abliterate_fp4

    # Non-existent src dir.
    rc = abliterate_fp4.main(
        [str(tmp_path / "nope"), str(tmp_path / "p.pt"), str(tmp_path / "o")]
    )
    assert rc == 2
