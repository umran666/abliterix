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


# ---------------------------------------------------------------------------
# frozen_experts dispatch: hooks instead of weight mutation
# ---------------------------------------------------------------------------


class _Router(nn.Module):
    def __init__(self, hidden, n_exp, top_k=2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_exp, hidden))
        self.top_k = top_k

    def forward(self, x):
        logits = x @ self.weight.T
        scores, idx = torch.topk(torch.softmax(logits, -1), self.top_k, dim=-1)
        return scores, idx


class _Experts(nn.Module):
    """Stands in for a packed container: W is a buffer, never edited."""

    def __init__(self, n_exp, hidden, inter):
        super().__init__()
        self.register_buffer("W", torch.randn(n_exp, inter, hidden) * 0.05)
        self.register_buffer("down_proj_bias", torch.randn(n_exp, hidden) * 0.1)
        self.register_buffer("up", torch.randn(hidden, inter) * 0.05)

    def forward(self, x, indices, scores):
        acts = x @ self.up
        out = torch.zeros_like(x)
        for k in range(indices.shape[-1]):
            for e in range(self.W.shape[0]):
                m = indices[:, k] == e
                if m.any():
                    out[m] += scores[m, k : k + 1] * (
                        acts[m] @ self.W[e] + self.down_proj_bias[e]
                    )
        return out


class _MoEBlock(nn.Module):
    def __init__(self, hidden, inter, n_exp):
        super().__init__()
        self.router = _Router(hidden, n_exp)
        self.experts = _Experts(n_exp, hidden, inter)

    def forward(self, x):
        scores, idx = self.router(x)
        return self.experts(x, idx, scores)


def _frozen_engine(n_layers=2, hidden=8, inter=12, n_exp=3):
    torch.manual_seed(50)
    layers = [
        SimpleNamespace(mlp=_MoEBlock(hidden, inter, n_exp)) for _ in range(n_layers)
    ]
    return SimpleNamespace(
        transformer_layers=layers,
        _locate_router=lambda layer: layer.mlp.router,
        _angular_hooks=[],
    )


def _frozen_config():
    return SimpleNamespace(
        steering=SimpleNamespace(
            decay_kernel=DecayKernel.LINEAR,
            weight_normalization=WeightNorm.NONE,
            frozen_experts=True,
        ),
        display=SimpleNamespace(print_responses=False),
    )


def test_frozen_ega_installs_hooks_without_touching_weights():
    from abliterix.core.steering import _apply_frozen_ega_steering

    engine = _frozen_engine()
    hidden = 8
    before = [layer.mlp.experts.W.clone() for layer in engine.transformer_layers]
    x = torch.randn(5, hidden)
    baseline = [layer.mlp(x).clone() for layer in engine.transformer_layers]

    svs = torch.randn(len(engine.transformer_layers) + 1, hidden)
    profiles = {"mlp.down_proj": SteeringProfile(2.0, 0.0, 2.0, 100.0)}
    _apply_frozen_ega_steering(engine, svs, None, profiles, _frozen_config(), None)

    assert len(engine._angular_hooks) >= len(engine.transformer_layers)
    # Weights are untouched — that is the entire point.
    for layer, w0 in zip(engine.transformer_layers, before):
        assert torch.equal(layer.mlp.experts.W, w0)
    # ...but the output changed.
    for layer, y0 in zip(engine.transformer_layers, baseline):
        assert not torch.allclose(layer.mlp(x), y0, atol=1e-4)

    # restore_baseline's cleanup removes them and restores the original output.
    for h in engine._angular_hooks:
        h.remove()
    engine._angular_hooks = []
    for layer, y0 in zip(engine.transformer_layers, baseline):
        assert torch.allclose(layer.mlp(x), y0, atol=1e-5)


def test_frozen_ega_matches_the_weight_edit_it_replaces():
    """Hooked frozen block == the same block with its expert weight edited."""
    from abliterix.core.steering import _apply_frozen_ega_steering
    from abliterix.weight_transforms import apply_ega_projection

    hidden, strength = 8, 1.5
    engine = _frozen_engine(n_layers=1, hidden=hidden)
    layer = engine.transformer_layers[0]
    x = torch.randn(6, hidden)

    svs = torch.randn(2, hidden)
    profiles = {"mlp.down_proj": SteeringProfile(strength, 0.0, strength, 100.0)}
    _apply_frozen_ega_steering(engine, svs, None, profiles, _frozen_config(), None)
    hooked = layer.mlp(x)

    for h in engine._angular_hooks:
        h.remove()
    engine._angular_hooks = []

    # Reference: mutate the weight the way _apply_ega_steering would.
    W = layer.mlp.experts.W
    W.copy_(
        apply_ega_projection(
            W, svs[1], strength=strength, axis_is_in=True, preserve_row_norm=False
        ).to(W.dtype)
    )
    edited = layer.mlp(x)
    assert torch.allclose(hooked, edited, atol=1e-4)
