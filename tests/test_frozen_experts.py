"""Tests for abliterix.core.frozen_experts — forward-time EGA on frozen weights.

The load-bearing property: running an expert through the forward path must give
exactly what editing its weights would have given. If that holds, the Optuna
search can run against packed 4-bit weights (13 GB instead of 27 GB for
gpt-oss-20b; 160 GB instead of ~600 GB for DeepSeek-V4-Flash) and the resulting
plan can still be baked into a checkpoint by fp4_repack describing the same
model. Everything here is synthetic and CPU-only.
"""

import pytest
import torch

from abliterix.core.frozen_experts import (
    FrozenEgaPlan,
    apply_frozen_ega,
    build_frozen_plan,
    compute_norm_preserve_scales,
    install_frozen_ega_hook,
    project_out_direction,
    weighted_bias_projection,
)
from abliterix.weight_transforms import apply_ega_projection


def _rand(E=3, d0=12, d1=20, seed=0):
    torch.manual_seed(seed)
    return torch.randn(E, d0, d1)


# ---------------------------------------------------------------------------
# The weight-free projection
# ---------------------------------------------------------------------------


def test_project_out_direction_removes_component():
    """With a unit direction (what abliterix always supplies) strength 1 is
    exactly the orthogonal projection."""
    torch.manual_seed(1)
    y = torch.randn(4, 16)
    d = torch.randn(16)
    d = d / d.norm()
    out = project_out_direction(y, d, strength=1.0)
    assert torch.allclose((out * d).sum(-1), torch.zeros(4), atol=1e-5)


def test_project_out_direction_uses_raw_direction_scaling():
    """The direction is NOT renormalised — it must match apply_ega_projection.

    A silent renormalisation here would make the frozen path disagree with the
    weight-edit path whenever the caller's vector is not unit length.
    """
    torch.manual_seed(2)
    y = torch.randn(3, 12)
    d = torch.randn(12)
    got = project_out_direction(y, d, strength=0.7)
    want = y - 0.7 * (y * d).sum(-1, keepdim=True) * d
    assert torch.allclose(got, want, atol=1e-6)
    # Doubling d changes the result (it would not if we renormalised).
    assert not torch.allclose(got, project_out_direction(y, 2 * d, 0.7), atol=1e-4)


def test_project_out_direction_zero_strength_is_identity():
    y = torch.randn(3, 8)
    assert torch.allclose(project_out_direction(y, torch.randn(8), 0.0), y, atol=1e-6)


def test_project_out_direction_preserves_dtype():
    y = torch.randn(2, 8, dtype=torch.bfloat16)
    assert project_out_direction(y, torch.randn(8), 0.5).dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Equivalence WITHOUT norm preservation — the fully weight-free case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis_is_in", [True, False])
def test_forward_equals_weight_edit_no_norm_preserve(axis_is_in):
    """y = x @ W_edited  ==  project(x @ W_base). No weights touched."""
    E, d0, d1 = 3, 12, 20
    W = _rand(E, d0, d1, seed=2)
    strength = 1.7
    hidden = d1 if axis_is_in else d0
    d = torch.randn(hidden)

    W_edit = apply_ega_projection(
        W, d, strength=strength, axis_is_in=axis_is_in, preserve_row_norm=False
    )
    plan = build_frozen_plan(
        None, d, strength, axis_is_in=axis_is_in, preserve_row_norm=False
    )
    assert plan.scales is None  # never read the weights

    for e in range(E):
        x = torch.randn(d0 if axis_is_in else d1)
        want = x @ W_edit[e] if axis_is_in else W_edit[e] @ x
        got = apply_frozen_ega(x, W, plan, expert_index=e)
        assert torch.allclose(got, want, atol=1e-5), (axis_is_in, e)


# ---------------------------------------------------------------------------
# Equivalence WITH norm preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis_is_in", [True, False])
@pytest.mark.parametrize("strength", [0.5, 1.0, 2.5])
def test_forward_equals_weight_edit_with_norm_preserve(axis_is_in, strength):
    E, d0, d1 = 3, 12, 20
    W = _rand(E, d0, d1, seed=3)
    hidden = d1 if axis_is_in else d0
    d = torch.randn(hidden)

    W_edit = apply_ega_projection(
        W, d, strength=strength, axis_is_in=axis_is_in, preserve_row_norm=True
    )
    plan = build_frozen_plan(
        W, d, strength, axis_is_in=axis_is_in, preserve_row_norm=True
    )
    assert plan.scales is not None and plan.scales.shape == (E, d0)

    for e in range(E):
        x = torch.randn(d0 if axis_is_in else d1)
        want = x @ W_edit[e] if axis_is_in else W_edit[e] @ x
        got = apply_frozen_ega(x, W, plan, expert_index=e)
        assert torch.allclose(got, want, atol=1e-4), (axis_is_in, strength, e)


@pytest.mark.parametrize("axis_is_in", [True, False])
def test_norm_preserve_scales_match_explicit_row_norms(axis_is_in):
    """The closed-form factors equal the ratio of the actual edited row norms."""
    E, d0, d1 = 2, 10, 16
    W = _rand(E, d0, d1, seed=4)
    strength = 1.3
    d = torch.randn(d1 if axis_is_in else d0)

    got = compute_norm_preserve_scales(W, d, strength, axis_is_in=axis_is_in)

    # Explicitly build the un-normalised edit and measure its row norms.
    raw = apply_ega_projection(
        W, d, strength=strength, axis_is_in=axis_is_in, preserve_row_norm=False
    )
    orig = torch.linalg.vector_norm(W.float(), dim=-1)
    new = torch.linalg.vector_norm(raw, dim=-1).clamp(min=1e-8)
    assert torch.allclose(got, orig / new, atol=1e-4)


def test_scales_side_depends_on_layout():
    W = _rand(seed=5)
    d_in = torch.randn(W.shape[2])
    d_out = torch.randn(W.shape[1])
    assert build_frozen_plan(
        W, d_in, 1.0, axis_is_in=True, preserve_row_norm=True
    ).scales_apply_to_input
    assert not build_frozen_plan(
        W, d_out, 1.0, axis_is_in=False, preserve_row_norm=True
    ).scales_apply_to_input


def test_build_plan_rejects_missing_weight_when_norms_requested():
    with pytest.raises(ValueError, match="needs the base weight"):
        build_frozen_plan(
            None, torch.randn(8), 1.0, axis_is_in=True, preserve_row_norm=True
        )


def test_compute_scales_rejects_direction_mismatch():
    W = _rand(seed=6)
    with pytest.raises(ValueError, match="direction len"):
        compute_norm_preserve_scales(W, torch.randn(7), 1.0, axis_is_in=True)
    with pytest.raises(ValueError, match="direction len"):
        compute_norm_preserve_scales(W, torch.randn(7), 1.0, axis_is_in=False)


# ---------------------------------------------------------------------------
# Batched activations + per-token expert routing
# ---------------------------------------------------------------------------


def test_batched_tokens_match_per_token_application():
    """A (tokens, d) activation is handled the same as looping over tokens."""
    E, d0, d1 = 2, 8, 12
    W = _rand(E, d0, d1, seed=7)
    d = torch.randn(d1)
    plan = build_frozen_plan(W, d, 1.1, axis_is_in=True, preserve_row_norm=True)
    x = torch.randn(5, d0)
    batched = apply_frozen_ega(x, W, plan, expert_index=1)
    for t in range(x.shape[0]):
        one = apply_frozen_ega(x[t], W, plan, expert_index=1)
        assert torch.allclose(batched[t], one, atol=1e-5)


def test_per_token_expert_index_selects_matching_scales():
    """Token-major routing: each token gets its own expert's rescale row."""
    E, d0, d1 = 3, 6, 10
    W = _rand(E, d0, d1, seed=8)
    d = torch.randn(d1)
    plan = build_frozen_plan(W, d, 0.9, axis_is_in=True, preserve_row_norm=True)

    x = torch.randn(3, d0)
    idx = torch.tensor([2, 0, 1])
    scaled = plan.prepare_input(x, idx)
    for t, e in enumerate(idx.tolist()):
        assert torch.allclose(scaled[t], x[t] * plan.scales[e], atol=1e-5)


def test_gather_requires_index_when_multiple_experts():
    W = _rand(seed=9)
    plan = build_frozen_plan(
        W, torch.randn(W.shape[2]), 1.0, axis_is_in=True, preserve_row_norm=True
    )
    with pytest.raises(ValueError, match="expert_index required"):
        plan.prepare_input(torch.randn(W.shape[1]), None)


# ---------------------------------------------------------------------------
# The property that makes search-on-frozen-weights safe
# ---------------------------------------------------------------------------


def test_frozen_search_then_bake_describe_the_same_model():
    """Forward-hook search and a weight bake from the same plan agree.

    This is the contract path A relies on: tune strengths against frozen packed
    weights, then hand the winning parameters to fp4_repack, and the baked
    checkpoint behaves as the search measured.
    """
    E, d0, d1 = 4, 16, 24
    W = _rand(E, d0, d1, seed=10)
    d = torch.randn(d1)
    for strength in (0.4, 1.0, 3.0):
        plan = build_frozen_plan(
            W, d, strength, axis_is_in=True, preserve_row_norm=True
        )
        baked = apply_ega_projection(
            W, d, strength=strength, axis_is_in=True, preserve_row_norm=True
        )
        x = torch.randn(7, d0)
        via_hook = apply_frozen_ega(x, W, plan, expert_index=2)
        via_bake = x @ baked[2]
        rel = (via_hook - via_bake).abs().max() / via_bake.abs().max().clamp(min=1e-9)
        assert rel < 1e-4, (strength, float(rel))


def test_plan_is_serialisable_state():
    """A plan is plain tensors/floats — safe to ship to a worker or store."""
    W = _rand(seed=11)
    plan = build_frozen_plan(
        W, torch.randn(W.shape[2]), 1.5, axis_is_in=True, preserve_row_norm=True
    )
    assert isinstance(plan, FrozenEgaPlan)
    assert plan.direction.dtype == torch.float32
    assert isinstance(plan.strength, float)
    assert plan.scales.dtype == torch.float32


# ---------------------------------------------------------------------------
# Hook installation against a synthetic MoE container
# ---------------------------------------------------------------------------


class _FakeExperts(torch.nn.Module):
    """Stands in for a packed expert container: weights are never editable."""

    def __init__(self, W):
        super().__init__()
        self.register_buffer("W", W)  # (E, in, out) — gpt-oss orientation

    def forward(self, x, expert_index=0):
        return x @ self.W[expert_index]


def test_hook_reproduces_weight_edit_without_norm_preserve():
    """A hooked frozen module == the same module with edited weights."""
    torch.manual_seed(20)
    E, d_in, d_out = 3, 10, 14
    W = torch.randn(E, d_in, d_out)
    d = torch.randn(d_out)
    strength = 1.6

    frozen = _FakeExperts(W.clone())
    plan = build_frozen_plan(
        None, d, strength, axis_is_in=True, preserve_row_norm=False
    )
    handles = install_frozen_ega_hook(frozen, plan)

    edited = _FakeExperts(
        apply_ega_projection(
            W, d, strength=strength, axis_is_in=True, preserve_row_norm=False
        )
    )

    x = torch.randn(6, d_in)
    for e in range(E):
        assert torch.allclose(frozen(x, e), edited(x, e), atol=1e-4)

    # The frozen module's weights were never modified.
    assert torch.equal(frozen.W, W)

    for h in handles:
        h.remove()
    assert torch.allclose(frozen(x, 0), x @ W[0], atol=1e-5)  # back to baseline


def test_hook_rejects_fused_norm_preserve_instead_of_guessing():
    """Multi-expert row-norm preservation cannot be done at the container level."""
    W = _rand(E=4, seed=21)
    plan = build_frozen_plan(
        W, torch.randn(W.shape[2]), 1.0, axis_is_in=True, preserve_row_norm=True
    )
    with pytest.raises(NotImplementedError, match="per-token routing"):
        install_frozen_ega_hook(_FakeExperts(W), plan)


def test_hook_handles_tuple_outputs():
    class _TupleExperts(torch.nn.Module):
        def __init__(self, W):
            super().__init__()
            self.register_buffer("W", W)

        def forward(self, x):
            return x @ self.W, "aux"

    torch.manual_seed(22)
    W = torch.randn(8, 12)
    d = torch.randn(12)
    mod = _TupleExperts(W)
    install_frozen_ega_hook(
        mod, build_frozen_plan(None, d, 1.0, axis_is_in=True, preserve_row_norm=False)
    )
    x = torch.randn(3, 8)
    y, aux = mod(x)
    assert aux == "aux"
    assert torch.allclose(y, project_out_direction(x @ W, d, 1.0), atol=1e-5)


def test_single_expert_norm_preserve_hook_is_allowed():
    """Per-expert modules (DeepSeek layout) can preserve norms via the hook."""
    torch.manual_seed(23)
    d_out, d_in = 10, 16
    W = torch.randn(1, d_out, d_in)  # single expert, (out, in)
    d = torch.randn(d_out)
    strength = 1.2

    plan = build_frozen_plan(W, d, strength, axis_is_in=False, preserve_row_norm=True)
    assert plan.scales.shape == (1, d_out)

    x = torch.randn(d_in)
    got = apply_frozen_ega(x, W, plan, expert_index=0)
    edited = apply_ega_projection(
        W, d, strength=strength, axis_is_in=False, preserve_row_norm=True
    )
    assert torch.allclose(got, edited[0] @ x, atol=1e-4)


# ---------------------------------------------------------------------------
# Expert bias: EGA edits the weight but not the bias, so a container-level hook
# over-projects unless told about it. gpt-oss's down_proj_bias is (E, hidden).
# ---------------------------------------------------------------------------


def _moe_reference(x, W, bias, routing, edited_W=None):
    """sum_e routing[t,e] * (x[t] @ W[e] + bias[e]) — the container's job."""
    Wm = W if edited_W is None else edited_W
    per_e = torch.stack([x @ Wm[e] + bias[e] for e in range(W.shape[0])], dim=1)
    return (routing.unsqueeze(-1) * per_e).sum(dim=1)


def test_bias_makes_naive_hook_diverge_and_correction_fixes_it():
    """The whole point of `bias_dot_d`: without it the hook is measurably wrong."""
    torch.manual_seed(40)
    E, d_in, d_out, T = 4, 10, 14, 6
    W = torch.randn(E, d_in, d_out)
    bias = torch.randn(E, d_out) * 0.5
    routing = torch.softmax(torch.randn(T, E), dim=-1)
    x = torch.randn(T, d_in)
    d = torch.randn(d_out)
    d = d / d.norm()
    strength = 2.0

    # Ground truth: edit the WEIGHT only, bias untouched.
    W_edit = apply_ega_projection(
        W, d, strength=strength, axis_is_in=True, preserve_row_norm=False
    )
    want = _moe_reference(x, W, bias, routing, edited_W=W_edit)

    y_container = _moe_reference(x, W, bias, routing)
    plan = build_frozen_plan(
        None, d, strength, axis_is_in=True, preserve_row_norm=False
    )

    naive = plan.finish_output(y_container)
    corr = weighted_bias_projection(bias, d, routing)
    fixed = plan.finish_output(y_container, bias_dot_d=corr)

    naive_err = (naive - want).abs().mean().item()
    fixed_err = (fixed - want).abs().mean().item()
    # The bias really does break the naive hook...
    assert naive_err > 1e-3, naive_err
    # ...and the correction restores exactness.
    assert fixed_err < 1e-5, (naive_err, fixed_err)


def test_bias_correction_is_a_noop_without_bias():
    torch.manual_seed(41)
    E, d_in, d_out, T = 3, 8, 12, 5
    W = torch.randn(E, d_in, d_out)
    bias = torch.zeros(E, d_out)
    routing = torch.softmax(torch.randn(T, E), dim=-1)
    x = torch.randn(T, d_in)
    d = torch.randn(d_out)
    plan = build_frozen_plan(None, d, 1.5, axis_is_in=True, preserve_row_norm=False)

    y = _moe_reference(x, W, bias, routing)
    corr = weighted_bias_projection(bias, d, routing)
    assert torch.allclose(corr, torch.zeros(T), atol=1e-6)
    assert torch.allclose(
        plan.finish_output(y), plan.finish_output(y, bias_dot_d=corr), atol=1e-6
    )


def test_weighted_bias_projection_matches_explicit_sum():
    torch.manual_seed(42)
    E, d_out, T = 5, 9, 4
    bias = torch.randn(E, d_out)
    routing = torch.softmax(torch.randn(T, E), dim=-1)
    d = torch.randn(d_out)
    got = weighted_bias_projection(bias, d, routing)
    want = torch.stack(
        [sum(routing[t, e] * (bias[e] @ d) for e in range(E)) for t in range(T)]
    )
    assert torch.allclose(got, want, atol=1e-5)


def test_installed_hook_applies_bias_correction_from_forward_args():
    """The installer wires bias correction through, recomputed per forward.

    Mirrors gpt-oss: a fused container that takes router scores as an argument
    and adds a per-expert down_proj bias internally.
    """
    torch.manual_seed(43)
    E, d_in, d_out, T = 4, 9, 13, 7
    W = torch.randn(E, d_in, d_out)
    bias = torch.randn(E, d_out) * 0.4
    d = torch.randn(d_out)
    d = d / d.norm()
    strength = 2.5

    class _Container(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("W", W)
            self.register_buffer("down_proj_bias", bias)

        def forward(self, x, routing):
            return _moe_reference(x, self.W, self.down_proj_bias, routing)

    mod = _Container()
    plan = build_frozen_plan(
        None, d, strength, axis_is_in=True, preserve_row_norm=False
    )
    install_frozen_ega_hook(
        mod,
        plan,
        bias_dot_d_fn=lambda args: weighted_bias_projection(bias, d, args[1]),
    )

    x = torch.randn(T, d_in)
    routing = torch.softmax(torch.randn(T, E), dim=-1)
    got = mod(x, routing)

    W_edit = apply_ega_projection(
        W, d, strength=strength, axis_is_in=True, preserve_row_norm=False
    )
    want = _moe_reference(x, W, bias, routing, edited_W=W_edit)
    assert torch.allclose(got, want, atol=1e-4)

    # A second call with different routing must recompute the correction.
    routing2 = torch.softmax(torch.randn(T, E), dim=-1)
    want2 = _moe_reference(x, W, bias, routing2, edited_W=W_edit)
    assert torch.allclose(mod(x, routing2), want2, atol=1e-4)
