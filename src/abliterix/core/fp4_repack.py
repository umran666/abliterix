# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Offline "edit-and-repack" bake path for native FP4 (MXFP4 / NVFP4) models.

Direct/EGA abliteration edits only a *subset* of a model's tensors (fused
expert ``down_proj``, ``attn.o_proj``, dense ``mlp.down_proj``), and each edit
is *local* to one tensor. That is the lever this module pulls: instead of
expanding a 284B FP4 checkpoint to ~568GB of BF16 just to steer it (2×+ the
footprint, 4× B200), it streams the original FP4 checkpoint shard by shard and,
for each edited tensor only, does ``dequant → project → re-pack to the same
FP4 format`` — every other tensor is copied through byte-for-byte. Peak memory
is a single layer's expert block, and the output is a clean standalone FP4
checkpoint that loads in any NVFP4/MXFP4-capable stack with no special support.

This is the counterpart of :func:`abliterix.core.fp8_utils.dequant_model_to_disk`
(FP8 → BF16 on disk) but it keeps the model 4-bit end to end.

Two-phase workflow
------------------
1. **Search / export** (model loaded at any precision): call
   :func:`record_steering_plan` at the moment the engine would apply the best
   trial. It records the resolved per-tensor edits (direction + strength +
   projection geometry) keyed by canonical parameter name — cheap, no extra
   memory beyond the already-loaded model.
2. **Bake** (single GPU, streams disk): :func:`abliterate_fp4_to_disk` replays
   that plan against the original FP4 shards.

Faithfulness
    The projection math is the *shared* :mod:`abliterix.weight_transforms`
    kernels the in-engine HF path uses, so the abliteration fingerprint is
    bit-identical. EGA axis is re-resolved from the on-disk dequantised shape
    (via recorded ``hidden_dim`` + ``transposed``) so a producer that stores
    experts transposed (gpt-oss) still steers the correct axis. Before writing,
    each source FP4 tensor is cross-checked against the model's own dequant
    when a reference is supplied (see :func:`abliterate_fp4_to_disk`).
"""

from __future__ import annotations

import gc
import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor

from . import fp4_utils as f4
from ..types import DecayKernel, DirectTransform, WeightNorm
from ..weight_transforms import (
    apply_direct_transform,
    apply_ega_projection,
    resolve_ega_axis,
)

# ---------------------------------------------------------------------------
# Edit specification
# ---------------------------------------------------------------------------


@dataclass
class TensorEdit:
    """One resolved abliteration edit targeting a single named weight tensor.

    ``logical_name`` is the canonical (base-model, PEFT-stripped) parameter
    name, e.g. ``model.layers.5.mlp.experts.down_proj``. The offline tool maps
    it to the on-disk FP4 sibling keys (``*_blocks`` / ``*_scales`` / global).
    """

    logical_name: str
    kind: str  # "ega" | "direct"
    direction: Tensor  # 1-D float refusal direction
    strength: float
    preserve_row_norm: bool
    # EGA (fused 3-D expert tensor):
    hidden_dim: int | None = None
    transposed: bool = False
    # Direct (2-D Linear weight):
    projection_side: str | None = None  # "output" | "input"
    direct_transform: str = "standard"
    benign_dir: Tensor | None = None


def apply_tensor_edit(W: Tensor, edit: TensorEdit) -> Tensor:
    """Apply one :class:`TensorEdit` to a dequantised float weight.

    Returns a new float32 tensor (caller re-quantises / casts). Raises
    ``ValueError`` if the tensor shape is incompatible with the edit; the
    streaming caller turns that into a skip-with-warning rather than aborting.
    """
    W32 = W.to(torch.float32)
    if edit.kind == "ega":
        if W32.dim() != 3:
            raise ValueError(
                f"EGA edit '{edit.logical_name}' expects a 3-D fused expert "
                f"tensor, got shape {tuple(W32.shape)}"
            )
        hidden = edit.hidden_dim
        if hidden is None:
            hidden = edit.direction.shape[0]
        axis_is_in = resolve_ega_axis(
            tuple(W32.shape), hidden, transposed=edit.transposed
        )
        if axis_is_in is None:
            raise ValueError(
                f"EGA edit '{edit.logical_name}': hidden dim {hidden} matches "
                f"neither axis of {tuple(W32.shape)} (transposed={edit.transposed})"
            )
        return apply_ega_projection(
            W32,
            edit.direction,
            strength=edit.strength,
            axis_is_in=axis_is_in,
            preserve_row_norm=edit.preserve_row_norm,
        )

    # Direct 2-D weight.
    if W32.dim() != 2:
        raise ValueError(
            f"Direct edit '{edit.logical_name}' expects a 2-D weight, got "
            f"shape {tuple(W32.shape)}"
        )
    # Directions come from a plan loaded onto CPU while the weight may already
    # be on GPU (the bake streams shards to CUDA); always follow the weight.
    dev = W32.device
    transform = DirectTransform(edit.direct_transform)
    if transform != DirectTransform.STANDARD:
        # ORBA / biprojected / householder carry their own norm handling and
        # output-side preference — identical to the engine's advanced branch.
        return apply_direct_transform(
            transform,
            W32,
            edit.direction.to(device=dev, dtype=torch.float32),
            None
            if edit.benign_dir is None
            else edit.benign_dir.to(device=dev, dtype=torch.float32),
            strength=edit.strength,
            preserve_row_norm=edit.preserve_row_norm,
        ).to(torch.float32)

    # Standard rank-1 ablation with the engine's exact side precedence
    # (output-side first for square matrices) + optional row-norm post-step.
    v = _normalise(edit.direction.to(device=dev, dtype=torch.float32))
    out_f, in_f = W32.shape
    side = edit.projection_side
    if side is None:
        side = "output" if v.shape[0] == out_f else "input"
    if side == "output":
        if v.shape[0] != out_f:
            raise ValueError(
                f"Direct edit '{edit.logical_name}': output-side direction "
                f"len {v.shape[0]} != out_features {out_f}"
            )
        W_new = W32 - edit.strength * v.unsqueeze(1) * (v @ W32).unsqueeze(0)
    else:
        if v.shape[0] != in_f:
            raise ValueError(
                f"Direct edit '{edit.logical_name}': input-side direction "
                f"len {v.shape[0]} != in_features {in_f}"
            )
        W_new = W32 - edit.strength * (W32 @ v).unsqueeze(1) * v.unsqueeze(0)
    if edit.preserve_row_norm:
        orig = torch.linalg.vector_norm(W32, dim=1, keepdim=True)
        new = torch.linalg.vector_norm(W_new, dim=1, keepdim=True).clamp(min=1e-8)
        W_new = W_new * (orig / new)
    return W_new


def _normalise(v: Tensor, eps: float = 1e-8) -> Tensor:
    return v / torch.linalg.vector_norm(v).clamp(min=eps)


# ---------------------------------------------------------------------------
# Plan recording (engine → list[TensorEdit])
# ---------------------------------------------------------------------------


def _layer_strength(sp, layer_idx: int, kernel: DecayKernel) -> float | None:
    """Reproduce the per-layer decay used by _apply_direct/_apply_ega_steering.

    Returns ``None`` when this layer is out of the profile's falloff window or
    resolves to exactly zero strength (both are skips in the engine).
    """
    distance = abs(layer_idx - sp.max_weight_position)
    if distance > sp.min_weight_distance:
        return None
    t = distance / sp.min_weight_distance if sp.min_weight_distance else 0.0
    if kernel == DecayKernel.GAUSSIAN:
        strength = sp.min_weight + (sp.max_weight - sp.min_weight) * math.exp(
            -2.0 * t * t
        )
    elif kernel == DecayKernel.COSINE:
        strength = sp.min_weight + (sp.max_weight - sp.min_weight) * (
            0.5 * (1.0 + math.cos(math.pi * t))
        )
    else:  # LINEAR
        strength = sp.max_weight + t * (sp.min_weight - sp.max_weight)
    if strength == 0:
        return None
    return float(strength)


def normalize_param_name(name: str) -> str:
    """Strip PEFT decorations so an in-memory name matches the on-disk name.

    ``base_model.model.model.layers.0.self_attn.o_proj.base_layer.weight`` →
    ``model.layers.0.self_attn.o_proj.weight``. Idempotent on already-canonical
    names.
    """
    if name.startswith("base_model.model."):
        name = name[len("base_model.model.") :]
    return name.replace(".base_layer.", ".")


def _steering_vector_for(
    steering_vectors: Tensor, global_vector: Tensor | None, layer_idx: int
) -> Tensor:
    if global_vector is not None:
        return global_vector
    return steering_vectors[layer_idx + 1]


def record_steering_plan(
    engine,
    steering_vectors: Tensor,
    global_vector: Tensor | None,
    profiles: dict,
    config,
    discriminative_layers: set[int] | None = None,
) -> list[TensorEdit]:
    """Record the direct + EGA edits the engine would apply for the best trial.

    Mirrors the walk of ``core.steering._apply_direct_steering`` and
    ``_apply_ega_steering`` but emits :class:`TensorEdit` records instead of
    mutating weights. Call with the SAME arguments those functions receive.

    The model must be loaded (this uses ``engine.steerable_modules`` /
    ``engine._locate_fused_weights`` and the parameter-name maps). It does not
    depend on the weights being any particular precision — the recorded plan is
    then replayed against the original FP4 checkpoint on disk.
    """
    kernel = config.steering.decay_kernel
    preserve = config.steering.weight_normalization != WeightNorm.NONE
    direct_transform = getattr(
        config.steering, "direct_transform", DirectTransform.STANDARD
    )
    transposed = bool(getattr(engine, "_fused_down_proj_transposed", False))

    model = engine.model
    name_by_param_id = {id(p): n for n, p in model.named_parameters()}

    edits: list[TensorEdit] = []
    n_layers = len(engine.transformer_layers)

    # --- Direct (per steerable Linear module) ---
    for layer_idx in range(n_layers):
        if discriminative_layers is not None and layer_idx not in discriminative_layers:
            continue
        for component, modules in engine.steerable_modules(layer_idx).items():
            sp = profiles.get(component)
            if sp is None:
                continue
            strength = _layer_strength(sp, layer_idx, kernel)
            if strength is None:
                continue
            v = _steering_vector_for(steering_vectors, global_vector, layer_idx)
            if v.ndim != 1:
                # Multi-direction subspace direct steering is not represented
                # by a single TensorEdit; skip (caller handles rank-k elsewhere).
                continue
            vf = v.to(torch.float32)
            for mod in modules:
                base_mod = getattr(mod, "base_layer", mod)
                weight = getattr(base_mod, "weight", None)
                if weight is None:
                    continue
                raw_name = name_by_param_id.get(id(weight))
                if raw_name is None:
                    continue
                out_f, in_f = weight.shape[0], weight.shape[1]
                # Engine precedence: output-side first, then input-side.
                if vf.shape[0] == out_f:
                    side = "output"
                elif vf.shape[0] == in_f:
                    side = "input"
                else:
                    continue  # matches neither axis → engine skips it
                edits.append(
                    TensorEdit(
                        logical_name=normalize_param_name(raw_name),
                        kind="direct",
                        direction=vf.detach().cpu().clone(),
                        strength=strength,
                        preserve_row_norm=preserve,
                        projection_side=side,
                        direct_transform=DirectTransform(direct_transform).value,
                    )
                )

    # --- EGA (fused expert down_proj) ---
    if engine.has_expert_routing():
        sp = profiles.get("mlp.down_proj")
        if sp is not None:
            for layer_idx in range(n_layers):
                if (
                    discriminative_layers is not None
                    and layer_idx not in discriminative_layers
                ):
                    continue
                layer = engine.transformer_layers[layer_idx]
                fused = engine._locate_fused_weights(layer)
                if fused is None:
                    continue
                strength = _layer_strength(sp, layer_idx, kernel)
                if strength is None:
                    continue
                v = _steering_vector_for(steering_vectors, global_vector, layer_idx)
                if v.ndim != 1:
                    continue
                raw_name = name_by_param_id.get(id(fused))
                if raw_name is None:
                    continue
                axis_is_in = resolve_ega_axis(
                    tuple(fused.shape), v.shape[0], transposed=transposed
                )
                if axis_is_in is None:
                    continue
                edits.append(
                    TensorEdit(
                        logical_name=normalize_param_name(raw_name),
                        kind="ega",
                        direction=v.to(torch.float32).detach().cpu().clone(),
                        strength=strength,
                        preserve_row_norm=preserve,
                        hidden_dim=int(v.shape[0]),
                        transposed=transposed,
                    )
                )

    return edits


def record_steering_plan_from_trial(
    engine,
    steering_vectors: Tensor,
    vector_index: float | None,
    profiles: dict,
    config,
    *,
    benign_states: Tensor | None = None,
    target_states: Tensor | None = None,
) -> list[TensorEdit]:
    """Record a plan from the same inputs :func:`apply_steering` receives.

    Resolves the global vector and discriminative-layer selection exactly as
    ``apply_steering`` does (via the shared ``core.steering`` helpers), then
    delegates to :func:`record_steering_plan`. This is the entry point the
    export flow uses: it captures what the best trial *would* edit without
    mutating the model, so the plan can be replayed offline against the FP4
    checkpoint.
    """
    from .steering import _detect_discriminative_layers, resolve_global_vector

    global_vector = resolve_global_vector(steering_vectors, vector_index)
    discriminative_layers = None
    if config.steering.discriminative_layer_selection:
        discriminative_layers = _detect_discriminative_layers(
            steering_vectors, benign_states, target_states
        )
    return record_steering_plan(
        engine,
        steering_vectors,
        global_vector,
        profiles,
        config,
        discriminative_layers,
    )


# ---------------------------------------------------------------------------
# On-disk FP4 tensor layout
# ---------------------------------------------------------------------------


@dataclass
class _Fp4KeySet:
    """The safetensors keys that together encode one FP4 logical weight."""

    blocks: str  # packed nibble uint8
    scales: str  # per-block scale (ue8m0 uint8 for MXFP4, e4m3 for NVFP4)
    global_scale: str | None = None  # per-tensor fp32 (NVFP4 only)


# Suffix candidates seen across producers. MXFP4 (gpt-oss) uses
# ``_blocks``/``_scales``; NVFP4 (ModelOpt / llm-compressor) stores the packed
# weight under the bare name plus ``_scale`` (+ a ``_scale_2`` / global).
_MX_BLOCK_SUFFIXES = ("_blocks",)
_MX_SCALE_SUFFIXES = ("_scales",)
_NV_SCALE_SUFFIXES = ("_scale", "_weight_scale")
_NV_GLOBAL_SUFFIXES = (
    "_scale_2",
    "_weight_scale_2",
    "_global_scale",
    "_weight_global_scale",
)


def resolve_fp4_keys(
    logical_name: str, present: set[str], fmt: f4.Fp4Format
) -> _Fp4KeySet | None:
    """Find the FP4 sibling keys for ``logical_name`` among ``present`` keys.

    Returns ``None`` if the tensor is not FP4-encoded under this layout (e.g.
    it is a plain BF16 weight that should be edited without repacking).
    """
    if fmt.kind == "mxfp4":
        for bsuf in _MX_BLOCK_SUFFIXES:
            for ssuf in _MX_SCALE_SUFFIXES:
                b, s = logical_name + bsuf, logical_name + ssuf
                if b in present and s in present:
                    return _Fp4KeySet(b, s)
        return None
    # NVFP4: packed weight under the bare name + a block scale sibling.
    if logical_name not in present:
        return None
    for ssuf in _NV_SCALE_SUFFIXES:
        s = logical_name + ssuf
        if s in present:
            g = None
            for gsuf in _NV_GLOBAL_SUFFIXES:
                if logical_name + gsuf in present:
                    g = logical_name + gsuf
                    break
            return _Fp4KeySet(logical_name, s, g)
    return None


def _packed_moe_is_transposed(blocks: Tensor) -> bool:
    """True when a packed FP4 tensor stores the MoE layout the model transposes.

    **Verified against transformers 5.9's reference dequant**
    (``integrations.mxfp4._convert_moe_packed_tensors``): a 4-D packed tensor
    ``(E, rows, n_blocks, bytes)`` decodes to ``(E, rows, K)`` element-wise and
    is then ``.transpose(1, 2)``-ed, so the weight the *model* sees is
    ``(E, K, rows)``. Our element decoding is bit-identical to the reference;
    only this final axis swap is model-side.

    That matters because the steering plan records axis semantics against the
    **in-memory** tensor. Without undoing the swap, an EGA edit replayed
    offline would project the wrong axis — silently, and undetectably on
    gpt-oss where hidden == intermediate makes both axes the same length.

    2-D/3-D packed tensors (plain Linear weights) carry no such swap.
    """
    return blocks.dim() == 4


def _to_logical(W: Tensor, transposed: bool) -> Tensor:
    """On-disk orientation → the orientation the model/plan uses."""
    return W.transpose(1, 2).contiguous() if transposed else W


def _to_ondisk(W: Tensor, transposed: bool) -> Tensor:
    """Inverse of :func:`_to_logical`."""
    return W.transpose(1, 2).contiguous() if transposed else W


def _dequant_fp4_tensor(
    tensors: dict[str, Tensor], keys: _Fp4KeySet, fmt: f4.Fp4Format
) -> Tensor:
    """Dequantise an on-disk FP4 tensor into the model's logical orientation.

    Siblings are pulled onto ``blocks``' device: a shard's tensors may be a
    mix of freshly-repacked GPU tensors and untouched CPU ones.
    """
    blocks = tensors[keys.blocks]
    scales = tensors[keys.scales].to(blocks.device)
    g = tensors[keys.global_scale].to(blocks.device) if keys.global_scale else None
    W = f4.dequantize_fp4(fmt, blocks, scales, global_scale=g, out_dtype=torch.float32)
    return _to_logical(W, _packed_moe_is_transposed(blocks))


def _requant_fp4_tensor(
    W: Tensor,
    keys: _Fp4KeySet,
    fmt: f4.Fp4Format,
    orig: dict[str, Tensor],
    scale_search: int = 0,
) -> dict[str, Tensor]:
    """Re-pack ``W`` (float) into the same FP4 key layout.

    ``W`` is in the model's **logical** orientation (as produced by
    :func:`_dequant_fp4_tensor` and edited); it is rotated back to the on-disk
    orientation here so the written bytes keep the producer's convention.

    NVFP4 keeps the source tensor's global scale so block scales stay in the
    e4m3 range the producer chose. ``scale_search`` enables the MSE-optimal
    MXFP4 block-scale search (see :func:`fp4_utils.quantize_to_mxfp4`).
    """
    g = orig.get(keys.global_scale) if keys.global_scale else None
    if g is not None:
        g = g.to(W.device)  # plan/shard tensors are CPU; W may be on GPU
    W = _to_ondisk(W, _packed_moe_is_transposed(orig[keys.blocks]))
    blocks, scales, new_g = f4.quantize_fp4(
        fmt, W, global_scale=g, scale_search=scale_search
    )
    out: dict[str, Tensor] = {
        keys.blocks: blocks.to(orig[keys.blocks].dtype),
        keys.scales: scales.to(orig[keys.scales].dtype),
    }
    if keys.global_scale is not None:
        out[keys.global_scale] = (
            new_g if new_g is not None else orig[keys.global_scale]
        ).to(orig[keys.global_scale].dtype)
    return out


# ---------------------------------------------------------------------------
# Streaming bake
# ---------------------------------------------------------------------------


@dataclass
class RepackStats:
    tensors_written: int = 0
    fp4_edited: int = 0
    dense_edited: int = 0
    copied: int = 0
    skipped_edits: list[str] = field(default_factory=list)
    requant_rel_err: dict[str, float] = field(default_factory=dict)


def _iter_shards(src_dir: Path) -> dict[str, list[str]]:
    """Return ``{shard_filename: [tensor_key, ...]}`` for the source model."""
    idx_path = src_dir / "model.safetensors.index.json"
    if idx_path.exists():
        weight_map = json.loads(idx_path.read_text())["weight_map"]
    else:
        from safetensors import safe_open

        weight_map = {}
        for s in sorted(src_dir.glob("*.safetensors")):
            with safe_open(s, framework="pt") as f:
                for k in f.keys():
                    weight_map[k] = s.name
    shards: dict[str, list[str]] = {}
    for key, fname in weight_map.items():
        shards.setdefault(fname, []).append(key)
    return shards


def abliterate_fp4_to_disk(
    src_dir: str | Path,
    dst_dir: str | Path,
    edits: Iterable[TensorEdit],
    *,
    fmt: f4.Fp4Format | None = None,
    use_cuda: bool = True,
    verify_idempotent: bool = True,
    reference_dequant: Any = None,
    scale_search: int = 1,
    verbose: bool = True,
) -> RepackStats:
    """Replay ``edits`` against a native-FP4 checkpoint, streaming shard by shard.

    Edited FP4 tensors are dequantised, projected, and re-packed into the same
    FP4 layout; edited plain-BF16 tensors (e.g. gpt-oss ``attn.o_proj``, which
    is not quantised) are projected and written back in their storage dtype;
    every other tensor is copied through unchanged. ``quantization_config`` is
    preserved in ``config.json`` because the output is still FP4.

    Parameters
    ----------
    fmt
        FP4 format; auto-detected from ``config.json`` ``quantization_config``
        when ``None``.
    verify_idempotent
        Cross-check each source FP4 tensor's pack/unpack self-consistency
        before editing (cheap; catches an internal kernel bug).
    reference_dequant
        Optional callable ``(logical_name, our_W) -> ref_W | None``. When it
        returns a tensor, fp4_utils' dequant is asserted to match it — the real
        layout-faithfulness guard against a producer whose nibble order or
        scale encoding differs from what fp4_utils assumes.
    scale_search
        MSE-optimal MXFP4 block-scale search depth for re-packing edited
        tensors (default 1 = try the amax-tight exponent and one tighter). 0
        restores the plain amax scale. Higher trades a little bake time for
        lower requant error; ignored for NVFP4. See
        :func:`fp4_utils.quantize_to_mxfp4`.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    cfg = json.loads((src_dir / "config.json").read_text())
    if fmt is None:
        qc = cfg.get("quantization_config")
        if qc is None and isinstance(cfg.get("text_config"), dict):
            qc = cfg["text_config"].get("quantization_config")
        fmt = f4.detect_fp4_format(qc)
    if fmt is None:
        raise ValueError(
            f"{src_dir}: no FP4 quantization_config detected and no fmt given. "
            "Use fp8_utils.dequant_model_to_disk for FP8, or pass fmt=."
        )

    edit_by_name = {e.logical_name: e for e in edits}
    remaining = set(edit_by_name)
    stats = RepackStats()
    shards = _iter_shards(src_dir)
    new_weight_map: dict[str, str] = {}
    total_bytes = 0
    t0 = time.time()

    for i, (fname, keys) in enumerate(sorted(shards.items()), 1):
        present = set(keys)
        with safe_open(src_dir / fname, framework="pt") as f:
            src = {k: f.get_tensor(k) for k in keys}

        out: dict[str, Tensor] = {}
        consumed: set[str] = set()

        for name, edit in edit_by_name.items():
            fp4_keys = resolve_fp4_keys(name, present, fmt)
            if fp4_keys is not None:
                W = _dequant_fp4_tensor(src, fp4_keys, fmt).to(device)
                if verify_idempotent:
                    f4.assert_repack_idempotent(
                        src[fp4_keys.blocks].to(device),
                        src[fp4_keys.scales].to(device),
                        fmt,
                        global_scale=(
                            src[fp4_keys.global_scale].to(device)
                            if fp4_keys.global_scale
                            else None
                        ),
                        name=name,
                    )
                if reference_dequant is not None:
                    ref = reference_dequant(name, W)
                    if ref is not None:
                        f4.assert_matches_reference(W, ref.to(device), name=name)
                try:
                    W_new = apply_tensor_edit(W, edit)
                except ValueError as e:
                    if verbose:
                        print(f"  [yellow]skip FP4 edit {name}: {e}[/]")
                    stats.skipped_edits.append(name)
                    continue
                repacked = _requant_fp4_tensor(
                    W_new, fp4_keys, fmt, src, scale_search=scale_search
                )
                # Requant round-trip error (steered vs re-decoded), for the log.
                W_chk = _dequant_fp4_tensor({**src, **repacked}, fp4_keys, fmt).to(
                    device
                )
                denom = W_new.abs().mean().clamp(min=1e-9)
                stats.requant_rel_err[name] = float(
                    (W_chk - W_new).abs().mean() / denom
                )
                out.update({k: v.cpu() for k, v in repacked.items()})
                consumed.update(
                    {fp4_keys.blocks, fp4_keys.scales}
                    | ({fp4_keys.global_scale} if fp4_keys.global_scale else set())
                )
                stats.fp4_edited += 1
                remaining.discard(name)
            elif name in present:
                # Plain (non-quantised) weight, e.g. gpt-oss attn.o_proj.
                W = src[name].to(device)
                try:
                    W_new = apply_tensor_edit(W, edit)
                except ValueError as e:
                    if verbose:
                        print(f"  [yellow]skip dense edit {name}: {e}[/]")
                    stats.skipped_edits.append(name)
                    continue
                out[name] = W_new.to(src[name].dtype).cpu()
                consumed.add(name)
                stats.dense_edited += 1
                remaining.discard(name)

        # Copy through everything not produced by an edit.
        for k in keys:
            if k in consumed or k in out:
                continue
            out[k] = src[k]
            stats.copied += 1

        save_file(out, str(dst_dir / fname), metadata={"format": "pt"})
        for k in out:
            new_weight_map[k] = fname
        shard_bytes = (dst_dir / fname).stat().st_size
        total_bytes += shard_bytes
        if verbose:
            print(
                f"[{i}/{len(shards)}] {fname}: "
                f"{stats.fp4_edited} fp4-edited, {stats.dense_edited} dense-edited "
                f"({total_bytes / 1e9:.1f} GB, {(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
        del src, out
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    stats.tensors_written = len(new_weight_map)

    (dst_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": new_weight_map},
            indent=2,
        )
    )
    # Keep config.json AS-IS (still FP4) but write it out for a standalone dir.
    (dst_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    _copy_aux_files(src_dir, dst_dir)

    if remaining and verbose:
        print(
            f"  [yellow]{len(remaining)} planned edits matched no tensor key "
            f"(first few: {sorted(remaining)[:3]})[/]"
        )
    if verbose:
        rerr = stats.requant_rel_err
        worst = max(rerr.values()) if rerr else 0.0
        print(
            f"\nDone. FP4 model at {dst_dir} ({total_bytes / 1e9:.1f} GB). "
            f"{stats.fp4_edited} FP4 + {stats.dense_edited} dense tensors edited, "
            f"{stats.copied} copied. Worst requant rel-err {worst:.4f}."
        )
    return stats


_AUX_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
    "added_tokens.json",
)


def _copy_aux_files(src_dir: Path, dst_dir: Path) -> None:
    for name in _AUX_FILES:
        p = src_dir / name
        if p.exists():
            shutil.copy2(p, dst_dir / name)
    for p in src_dir.glob("*.py"):
        shutil.copy2(p, dst_dir / p.name)


# ---------------------------------------------------------------------------
# Plan (de)serialisation
# ---------------------------------------------------------------------------


def save_plan(edits: list[TensorEdit], path: str | Path) -> None:
    """Serialise a recorded plan to a ``.pt`` file (directions kept as tensors)."""
    payload = [
        {
            "logical_name": e.logical_name,
            "kind": e.kind,
            "direction": e.direction.detach().cpu(),
            "strength": e.strength,
            "preserve_row_norm": e.preserve_row_norm,
            "hidden_dim": e.hidden_dim,
            "transposed": e.transposed,
            "projection_side": e.projection_side,
            "direct_transform": e.direct_transform,
            "benign_dir": None if e.benign_dir is None else e.benign_dir.detach().cpu(),
        }
        for e in edits
    ]
    torch.save(payload, str(path))


def load_plan(path: str | Path) -> list[TensorEdit]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    return [TensorEdit(**entry) for entry in payload]
