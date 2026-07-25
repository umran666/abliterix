# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Apply the EGA edit at *forward time* so quantised expert weights stay packed.

:mod:`abliterix.core.fp4_repack` shrinks the abliteration *output* — it edits an
FP4 checkpoint and writes FP4 back. It does not shrink the *search*: the Optuna
loop still needs a model whose expert weights can be mutated, so MXFP4 is loaded
as BF16 (gpt-oss-20b: 27 GB instead of 13.8 GB; DeepSeek-V4-Flash: ~600 GB
instead of 160 GB). That expansion is what forces multi-GPU sizing.

This module removes the need to mutate weights at all. The observation is that
the EGA projection is *rank-1*, so it can be applied to the expert's activations
instead of its weights — and the two are algebraically identical.

Why it works
------------
EGA edits a fused expert weight as ``W_new = W - s · P(W, d)``. Writing out the
matmul for each of the two layouts abliterix supports:

**gpt-oss layout** — ``W[e]`` is ``(in, out)``, ``d`` lives in ``out``,
``y = x @ W[e]`` (this is ``axis_is_in=True`` in
:func:`~abliterix.weight_transforms.resolve_ega_axis` terms)::

    W_new = W - s (W d) ⊗ d
    y_new = x @ W_new = x@W - s (x @ W @ d) d = y - s (y·d) d

**DeepSeek layout** — ``W[e]`` is ``(out, in)``, ``d`` lives in ``out``,
``y = W[e] @ x`` (``axis_is_in=False``)::

    W_new = W - s d ⊗ (dᵀ W)
    y_new = W_new @ x = W@x - s d (dᵀ W x) = y - s (y·d) d

**Both collapse to the same weight-free expression**: project ``d`` out of the
expert's output. No dequantisation, no writable parameter, nothing cached per
expert — the packed 4-bit weights are read by their native kernel exactly as
shipped.

Row-norm preservation
---------------------
``weight_normalization != "none"`` rescales every row of the edited weight back
to its original L2 norm, which is *not* expressible as a pure activation
transform — but it is still only an elementwise rescale, on a different side per
layout:

* gpt-oss ``(in, out)``, norms taken along ``out``: one factor per **input**
  position, i.e. scale ``x`` before the matmul.
* DeepSeek ``(out, in)``, norms taken along ``in``: one factor per **output**
  position, i.e. scale ``y`` after the projection.

The factors depend on the direction and strength, so they must be recomputed per
trial by :func:`compute_norm_preserve_scales`, which needs one or two matvecs
against the base weight (cheap in FLOPs, but it does read the weights — the
caller can dequantise a single layer transiently rather than the whole model).

Setting ``weight_normalization = "none"`` therefore makes the search completely
weight-free; ``"full"`` costs one per-layer pass per trial.

The forward result is bit-comparable to what :mod:`fp4_repack` bakes, so a
search run through these hooks and a checkpoint baked from the resulting plan
describe the same model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

_EPS = 1e-8


# ---------------------------------------------------------------------------
# The weight-free part: project the refusal direction out of the expert output
# ---------------------------------------------------------------------------


def project_out_direction(y: Tensor, direction: Tensor, strength: float) -> Tensor:
    """``y - strength · (y·d) d`` over the last axis. The whole EGA edit when
    row-norm preservation is off.

    ``y`` is any activation whose last axis is the expert output space;
    ``direction`` is a 1-D vector in that space.

    ``direction`` is used **as given, not renormalised** — deliberately, so this
    matches :func:`~abliterix.weight_transforms.apply_ega_projection` for any
    input rather than only for unit vectors. (abliterix's steering vectors are
    unit-normalised upstream, so in practice the two conventions coincide; the
    equivalence tests pass raw vectors precisely to catch a silent divergence
    here.) Shape and dtype of ``y`` are preserved.
    """
    d = direction.to(dtype=torch.float32, device=y.device)
    y32 = y.to(torch.float32)
    coeff = (y32 * d).sum(dim=-1, keepdim=True)
    return (y32 - strength * coeff * d).to(y.dtype)


# ---------------------------------------------------------------------------
# The weight-dependent part: row-norm preservation factors
# ---------------------------------------------------------------------------


def compute_norm_preserve_scales(
    W: Tensor,
    direction: Tensor,
    strength: float,
    *,
    axis_is_in: bool,
) -> Tensor:
    """Per-row rescale factors reproducing ``preserve_row_norm`` at forward time.

    ``W`` is the fused expert weight ``(E, d0, d1)``; ``direction`` matches
    ``d1`` when ``axis_is_in`` else ``d0``. Returns ``(E, d0)`` factors — to be
    applied to the **input** when ``axis_is_in`` (gpt-oss), or to the
    **output** otherwise (DeepSeek).

    Derived from the closed forms of the edited row norms, so no edited weight
    is ever materialised. ``d`` is used as given (see
    :func:`project_out_direction`), so ``‖d‖²`` appears explicitly rather than
    being assumed to be 1:

    ``axis_is_in`` (row ``W[e,i,:]`` loses its ``d`` component)::

        ‖W_new[e,i,:]‖² = ‖W[e,i,:]‖² - (2s - s²‖d‖²) (W[e,i,:]·d)²

    otherwise (row ``W[e,o,:]`` is perturbed by ``-s·d[o]·q``, ``q = dᵀW[e]``)::

        ‖W_new[e,o,:]‖² = ‖W[e,o,:]‖² - 2s d[o] (W[e,o,:]·q) + s² d[o]² ‖q‖²
    """
    W32 = W.to(torch.float32)
    if W32.dim() != 3:
        raise ValueError(f"expected a fused (E, d0, d1) weight, got {tuple(W.shape)}")
    d = direction.to(dtype=torch.float32, device=W32.device)

    row_sq = (W32 * W32).sum(dim=-1)  # (E, d0) — norms along the last axis
    if axis_is_in:
        if d.shape[0] != W32.shape[2]:
            raise ValueError(f"direction len {d.shape[0]} != last axis {W32.shape[2]}")
        proj = torch.matmul(W32, d)  # (E, d0)
        d_sq = float((d * d).sum())
        new_sq = row_sq - (2.0 * strength - strength * strength * d_sq) * proj * proj
    else:
        if d.shape[0] != W32.shape[1]:
            raise ValueError(f"direction len {d.shape[0]} != axis 1 {W32.shape[1]}")
        q = torch.einsum("o,eoi->ei", d, W32)  # (E, d1)
        wq = torch.einsum("eoi,ei->eo", W32, q)  # (E, d0)
        q_sq = (q * q).sum(dim=-1, keepdim=True)  # (E, 1)
        dv = d.view(1, -1)  # (1, d0)
        new_sq = (
            row_sq - 2.0 * strength * dv * wq + (strength * strength) * (dv * dv) * q_sq
        )

    return torch.sqrt(row_sq.clamp(min=0.0)) / torch.sqrt(new_sq.clamp(min=_EPS**2))


# ---------------------------------------------------------------------------
# Per-layer plan
# ---------------------------------------------------------------------------


@dataclass
class FrozenEgaPlan:
    """Everything a forward hook needs for one MoE layer.

    ``scales`` is ``None`` when row-norm preservation is off — the common case
    that makes the whole layer weight-free.
    """

    direction: Tensor  # (hidden,) in the expert output space
    strength: float
    axis_is_in: bool  # gpt-oss (in, out) layout when True
    scales: Tensor | None = None  # (E, d0) rescale factors

    @property
    def scales_apply_to_input(self) -> bool:
        """gpt-oss scales the matmul input; DeepSeek scales its output."""
        return self.axis_is_in

    def prepare_input(
        self, x: Tensor, expert_index: Tensor | int | None = None
    ) -> Tensor:
        """Apply the input-side rescale (gpt-oss layout) before the expert matmul."""
        if self.scales is None or not self.scales_apply_to_input:
            return x
        r = _gather_expert_scales(self.scales, expert_index, x)
        return (x.to(torch.float32) * r).to(x.dtype)

    def finish_output(
        self,
        y: Tensor,
        expert_index: Tensor | int | None = None,
        bias_dot_d: Tensor | None = None,
    ) -> Tensor:
        """Project the direction out, then apply any output-side rescale.

        ``bias_dot_d`` corrects for an additive bias inside ``y``. EGA edits
        only the weight, so the intended output is ``P(act@W) + bias`` — but a
        hook sees ``y = act@W + bias`` and computes ``P(y) = P(act@W) + P(bias)``,
        over-projecting the bias by ``s·(bias·d)·d``. Passing ``bias·d`` (per
        token, from :func:`weighted_bias_projection`) adds that back, making the
        hook exactly equal to the weight edit. Omit it only when the expert has
        no bias, or when you deliberately want the bias projected too.
        """
        out = project_out_direction(y, self.direction, self.strength)
        if bias_dot_d is not None:
            d = self.direction.to(dtype=torch.float32, device=out.device)
            corr = bias_dot_d.to(dtype=torch.float32, device=out.device)
            out = (out.to(torch.float32) + self.strength * corr.unsqueeze(-1) * d).to(
                y.dtype
            )
        if self.scales is None or self.scales_apply_to_input:
            return out
        r = _gather_expert_scales(self.scales, expert_index, out)
        return (out.to(torch.float32) * r).to(y.dtype)


def weighted_bias_projection(
    bias: Tensor, direction: Tensor, routing_weights: Tensor
) -> Tensor:
    """``B·d`` per token, where ``B = Σ_e w[token,e] · bias[e]``.

    Feed the result to :meth:`FrozenEgaPlan.finish_output` as ``bias_dot_d`` to
    make a container-level hook exactly match the weight edit on architectures
    whose expert down-projection carries a bias (gpt-oss does: ``down_proj_bias``
    is ``(E, hidden)``).

    ``bias`` is ``(E, out)``, ``routing_weights`` is ``(tokens, E)`` — the same
    scores the MoE block uses to combine experts. Returns ``(tokens,)``.
    """
    b = bias.to(torch.float32)
    d = direction.to(dtype=torch.float32, device=b.device)
    per_expert = b @ d  # (E,)
    return routing_weights.to(dtype=torch.float32, device=b.device) @ per_expert


def _gather_expert_scales(
    scales: Tensor, expert_index: Tensor | int | None, like: Tensor
) -> Tensor:
    """Select the row of ``scales`` for the expert(s) this activation belongs to.

    ``expert_index`` may be a single int (one expert's tensor), a per-token
    index tensor (token-major routing), or ``None`` when ``scales`` already has
    no expert axis.
    """
    s = scales.to(device=like.device, dtype=torch.float32)
    if s.dim() == 1:
        return s
    if expert_index is None:
        if s.shape[0] != 1:
            raise ValueError(
                f"expert_index required to select among {s.shape[0]} experts"
            )
        return s[0]
    if isinstance(expert_index, int):
        return s[expert_index]
    idx = expert_index.to(s.device).long()
    return s[idx]


def build_frozen_plan(
    W: Tensor | None,
    direction: Tensor,
    strength: float,
    *,
    axis_is_in: bool,
    preserve_row_norm: bool,
) -> FrozenEgaPlan:
    """Build a :class:`FrozenEgaPlan`, reading ``W`` only if norms are preserved.

    Pass ``W=None`` when ``preserve_row_norm`` is False — that path never
    touches the weights, which is the point of this module.
    """
    scales = None
    if preserve_row_norm:
        if W is None:
            raise ValueError("preserve_row_norm=True needs the base weight")
        scales = compute_norm_preserve_scales(
            W, direction, strength, axis_is_in=axis_is_in
        )
    return FrozenEgaPlan(
        direction=direction.detach().to(torch.float32),
        strength=float(strength),
        axis_is_in=axis_is_in,
        scales=scales,
    )


# ---------------------------------------------------------------------------
# Reference application (what a hook does), used by tests and callers
# ---------------------------------------------------------------------------


def install_frozen_ega_hook(module, plan: FrozenEgaPlan, *, bias_dot_d_fn=None):
    """Attach ``plan`` to an MoE expert container's forward, returning handles.

    The hook post-processes whatever the container returns, so it works with a
    packed 4-bit expert kernel unchanged — that is the point: the weights are
    never dequantised, never copied, never mutated.

    ``bias_dot_d_fn`` is an optional ``callable(args) -> Tensor`` receiving the
    container's forward args and returning the per-token ``bias·d`` correction
    (see :func:`weighted_bias_projection`). **Supply it whenever the expert
    down-projection has a bias** — gpt-oss's does — or the hook will also
    project the bias, which the weight edit does not, and the two paths will
    disagree. Extracting the router scores from ``args`` is architecture-
    specific, which is why it is a caller-supplied hook rather than a guess.

    Requirements the caller owns:

    * The module must return activations whose **last axis is the expert output
      space** (the residual/hidden dim for a ``down_proj``-style container). If
      it returns a tuple, the first element is treated as the activation.
    * Row-norm preservation across a *fused multi-expert* container is rejected
      here rather than approximated: the rescale factors are per expert, and a
      container-level hook cannot see which expert each token was routed to.
      Either run with ``weight_normalization = "none"`` (the fully weight-free
      configuration this module exists for) or hook the per-expert modules
      individually, where the expert identity is unambiguous.

    Remove with ``handle.remove()`` on each returned handle, exactly like the
    angular-mode hooks in :mod:`abliterix.core.steering`.
    """
    if plan.scales is not None and plan.scales.dim() == 2 and plan.scales.shape[0] > 1:
        raise NotImplementedError(
            f"row-norm preservation needs per-expert scales ({plan.scales.shape[0]} "
            "experts), but a fused-container hook cannot see per-token routing. "
            "Use weight_normalization='none' for the frozen-weight search, or "
            "install this on each expert module separately."
        )

    handles = []

    if plan.scales is not None and plan.scales_apply_to_input:

        def _pre(_mod, args):
            if not args:
                return args
            return (plan.prepare_input(args[0], None),) + tuple(args[1:])

        handles.append(module.register_forward_pre_hook(_pre))

    def _post(_mod, args, output):
        # Recomputed per call: the correction is per token, so it depends on
        # this forward's routing, not on anything cached at install time.
        corr = bias_dot_d_fn(args) if bias_dot_d_fn is not None else None
        y = output[0] if isinstance(output, tuple) else output
        flat = y.reshape(-1, y.shape[-1]) if corr is not None else y
        edited = plan.finish_output(flat, None, bias_dot_d=corr)
        if corr is not None:
            edited = edited.reshape(y.shape)
        if isinstance(output, tuple):
            return (edited,) + tuple(output[1:])
        return edited

    handles.append(module.register_forward_hook(_post))
    return handles


def apply_frozen_ega(
    x: Tensor,
    W: Tensor,
    plan: FrozenEgaPlan,
    *,
    expert_index: int | None = None,
) -> Tensor:
    """Run one expert's matmul under ``plan`` without ever editing ``W``.

    Mirrors what a forward hook does around the model's own (possibly packed)
    expert kernel; here the matmul is explicit so the result can be compared
    against the weight-editing path. ``W`` is ``(d0, d1)`` for a single expert
    or ``(E, d0, d1)`` with ``expert_index`` given.
    """
    We = W if W.dim() == 2 else W[expert_index if expert_index is not None else 0]
    We = We.to(torch.float32)
    xi = plan.prepare_input(x, expert_index)
    y = xi.to(torch.float32) @ We if plan.axis_is_in else We @ xi.to(torch.float32)
    return plan.finish_output(y, expert_index)
