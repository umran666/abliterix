"""Regression guard for the _apply_ega_steering refactor.

The fused-expert projection math was extracted into
``weight_transforms.apply_ega_projection`` so the offline FP4 repack tool and
the in-engine HF path stay bit-identical. This drives the real
``_apply_ega_steering`` through a mock engine and checks it still matches the
verbatim pre-refactor inline math. Synthetic tensors, no GPU, no model.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from abliterix.core.steering import _apply_direct_steering, _apply_ega_steering
from abliterix.types import DecayKernel, SteeringProfile, WeightNorm


def _ega_reference(W_all, vf, strength, axis_is_in, norm_preserve):
    W_all = W_all.to(torch.float32)
    if axis_is_in:
        proj = torch.matmul(W_all, vf)
        W_new = W_all - strength * (proj.unsqueeze(-1) * vf.view(1, 1, -1))
    else:
        proj = torch.einsum("o,eoi->ei", vf, W_all)
        W_new = W_all - strength * (vf.view(1, -1, 1) * proj.unsqueeze(1))
    if norm_preserve:
        orig = torch.linalg.vector_norm(W_all, dim=2, keepdim=True)
        new = torch.linalg.vector_norm(W_new, dim=2, keepdim=True).clamp(min=1e-8)
        W_new = W_new * (orig / new)
    return W_new


def _make_engine(fused_params, transposed=False):
    layers = [SimpleNamespace(_idx=i) for i in range(len(fused_params))]

    def _locate(layer):
        return fused_params[layer._idx]

    return SimpleNamespace(
        transformer_layers=layers,
        _locate_fused_weights=_locate,
        _fused_down_proj_transposed=transposed,
    )


def _config(kernel=DecayKernel.LINEAR, norm=WeightNorm.FULL):
    return SimpleNamespace(
        steering=SimpleNamespace(
            decay_kernel=kernel,
            weight_normalization=norm,
            direct_transform="standard",
            direct_transform_preserve_row_norm=True,
        )
    )


def test_apply_ega_steering_matches_reference_standard_layout():
    torch.manual_seed(11)
    n_layers, E, hidden, inter = 3, 4, 8, 16
    fused = [
        torch.nn.Parameter(torch.randn(E, hidden, inter), requires_grad=False)
        for _ in range(n_layers)
    ]
    originals = [p.data.clone() for p in fused]
    engine = _make_engine(fused)
    # steering_vectors: (n_layers + 1, hidden); layer i uses index i + 1.
    svs = torch.randn(n_layers + 1, hidden)
    strength = 2.0  # constant: max==min weight
    profiles = {
        "mlp.down_proj": SteeringProfile(
            max_weight=strength,
            max_weight_position=0.0,
            min_weight=strength,
            min_weight_distance=100.0,
        )
    }

    _apply_ega_steering(engine, svs, None, profiles, _config(), None)

    for i in range(n_layers):
        vf = svs[i + 1].to(torch.float32)
        exp = _ega_reference(
            originals[i], vf, strength, axis_is_in=False, norm_preserve=True
        )
        assert torch.allclose(fused[i].data.float(), exp, atol=1e-5)


def test_apply_ega_steering_matches_reference_no_norm_preserve():
    torch.manual_seed(12)
    n_layers, E, hidden, inter = 2, 3, 8, 16
    fused = [
        torch.nn.Parameter(torch.randn(E, hidden, inter), requires_grad=False)
        for _ in range(n_layers)
    ]
    originals = [p.data.clone() for p in fused]
    engine = _make_engine(fused)
    svs = torch.randn(n_layers + 1, hidden)
    profiles = {
        "mlp.down_proj": SteeringProfile(
            max_weight=1.5,
            max_weight_position=0.0,
            min_weight=1.5,
            min_weight_distance=100.0,
        )
    }

    _apply_ega_steering(
        engine, svs, None, profiles, _config(norm=WeightNorm.NONE), None
    )

    for i in range(n_layers):
        vf = svs[i + 1].to(torch.float32)
        exp = _ega_reference(
            originals[i], vf, 1.5, axis_is_in=False, norm_preserve=False
        )
        assert torch.allclose(fused[i].data.float(), exp, atol=1e-5)


def test_apply_ega_steering_caches_originals_for_restore():
    torch.manual_seed(13)
    fused = [torch.nn.Parameter(torch.randn(4, 8, 16), requires_grad=False)]
    engine = _make_engine(fused)
    svs = torch.randn(2, 8)
    profiles = {
        "mlp.down_proj": SteeringProfile(
            max_weight=1.0,
            max_weight_position=0.0,
            min_weight=1.0,
            min_weight_distance=100.0,
        )
    }
    _apply_ega_steering(engine, svs, None, profiles, _config(), None)
    # The engine caches the pristine weight keyed by the fused Parameter so
    # restore_baseline() can roll back.
    assert fused[0] in engine._direct_weight_originals


def test_direct_steering_refuses_bnb_quantized_base_weight():
    """Runtime defense-in-depth: direct edit of a packed 4-bit weight raises."""
    hidden = 8
    o_proj = nn.Linear(hidden, hidden, bias=False)
    # Simulate a bitsandbytes Params4bit: a `quant_state` attribute marks the
    # weight as packed 4-bit storage that .to(float32) would not dequant.
    o_proj.weight.quant_state = object()

    layer = SimpleNamespace(_idx=0)
    engine = SimpleNamespace(
        transformer_layers=[layer],
        steerable_modules=lambda idx: {"attn.o_proj": [o_proj]},
        _direct_weight_originals={},
    )
    svs = torch.randn(2, hidden)
    profiles = {
        "attn.o_proj": SteeringProfile(1.0, 0.0, 1.0, 100.0),
    }
    with pytest.raises(RuntimeError, match="cannot edit quantised base weight"):
        _apply_direct_steering(engine, svs, None, profiles, _config(), None)
