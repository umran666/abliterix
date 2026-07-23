# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared utilities for Abliterix scripts.

setup_io() must be called before importing any library that captures
stdout/stderr (e.g. Rich, which is imported by abliterix.utils).
All other functions use lazy imports so that importing this module
does not trigger heavy library loading.
"""

import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Project root: src/abliterix/scriptlib.py -> src/abliterix -> src -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_TRIAL_REPLAY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrialArtifact:
    """Complete replay metadata extracted from one optimisation trial.

    ``recipe`` is optional for legacy checkpoints.  New checkpoints store a
    versioned snapshot of the steering configuration so replay/export scripts
    do not silently inherit different transforms or kernels from a later TOML.
    """

    vector_index: float | None
    profiles: dict[str, Any]
    routing: Any | None
    direct_transform: Any | None
    steering_variant: str
    recipe: dict[str, Any]


def setup_io():
    """Set up UTF-8 encoding for Windows and load .env file from project root."""
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_trial(checkpoint_dir, model_name, trial_number):
    """Load a specific trial from an Optuna checkpoint.

    Matches by user_attrs["index"] first (matches "Running trial X" display),
    falling back to trial.number (0-indexed).
    """
    import optuna
    from optuna.storages.journal import JournalFileBackend, JournalStorage

    from abliterix.util import slugify_model_name

    slug = slugify_model_name(model_name)
    journal_path = os.path.join(checkpoint_dir, f"{slug}.jsonl")

    if not os.path.exists(journal_path):
        raise FileNotFoundError(f"Checkpoint not found: {journal_path}")

    storage = JournalStorage(JournalFileBackend(journal_path))
    study = optuna.load_study(study_name="abliterix", storage=storage)

    matching = [t for t in study.trials if t.user_attrs.get("index") == trial_number]
    if not matching:
        matching = [t for t in study.trials if t.number == trial_number]
    if not matching:
        available = sorted(
            set(t.user_attrs.get("index", t.number) for t in study.trials)
        )
        raise ValueError(f"Trial {trial_number} not found. Available: {available[:20]}")

    return matching[0]


def build_trial_recipe(
    config,
    *,
    direct_transform,
    steering_variant: str,
    vector_scope: str,
) -> dict[str, Any]:
    """Freeze the effective steering semantics selected for one trial."""
    if vector_scope not in {"global", "per layer"}:
        raise ValueError(
            "vector_scope must be either 'global' or 'per layer', got "
            f"{vector_scope!r}."
        )
    steering = config.steering.model_dump(mode="json")
    steering["direct_transform"] = direct_transform.value
    steering["fixed_vector_scope"] = vector_scope
    return {
        "schema_version": _TRIAL_REPLAY_SCHEMA_VERSION,
        "model": {
            "model_id": config.model.model_id,
            "source_backend": config.model.backend,
        },
        "direction_inputs": {
            "system_prompt": config.system_prompt,
            "benign_prompts": config.benign_prompts.model_dump(mode="json"),
            "target_prompts": config.target_prompts.model_dump(mode="json"),
        },
        "steering": steering,
        "steering_variant": steering_variant,
    }


def extract_trial_artifact(trial) -> TrialArtifact:
    """Extract versioned replay metadata from a completed trial."""
    from abliterix.types import DirectTransform, ExpertRoutingConfig, SteeringProfile

    attrs = trial.user_attrs
    recipe = dict(attrs.get("steering_recipe") or {})
    if recipe:
        version = recipe.get("schema_version")
        if version != _TRIAL_REPLAY_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported trial replay schema {version!r}; "
                f"expected {_TRIAL_REPLAY_SCHEMA_VERSION}."
            )

    vector_index = attrs["vector_index"]
    profiles = {k: SteeringProfile(**v) for k, v in attrs["parameters"].items()}
    routing_dict = attrs.get("moe_parameters")
    routing = ExpertRoutingConfig(**routing_dict) if routing_dict else None

    saved_steering = recipe.get("steering") or {}
    attr_transform = attrs.get("direct_transform")
    recipe_transform = saved_steering.get("direct_transform")
    if (
        attr_transform is not None
        and recipe_transform is not None
        and attr_transform != recipe_transform
    ):
        raise ValueError(
            "Trial direct_transform conflicts with its saved steering recipe: "
            f"{attr_transform!r} != {recipe_transform!r}."
        )
    transform_value = attr_transform or recipe_transform
    direct_transform = (
        DirectTransform(transform_value) if transform_value is not None else None
    )
    attr_variant = attrs.get("steering_variant")
    recipe_variant = recipe.get("steering_variant")
    if (
        attr_variant is not None
        and recipe_variant is not None
        and attr_variant != recipe_variant
    ):
        raise ValueError(
            "Trial steering_variant conflicts with its saved steering recipe: "
            f"{attr_variant!r} != {recipe_variant!r}."
        )
    steering_variant = str(attr_variant or recipe_variant or "single")

    vector_scope = saved_steering.get("fixed_vector_scope")
    if vector_scope is not None:
        if vector_scope not in {"global", "per layer"}:
            raise ValueError(
                "Saved fixed_vector_scope must be 'global' or 'per layer', got "
                f"{vector_scope!r}."
            )
        if vector_scope == "global" and vector_index is None:
            raise ValueError(
                "Saved global vector scope requires a numeric vector_index."
            )
        if vector_scope == "per layer" and vector_index is not None:
            raise ValueError("Saved per-layer vector scope requires vector_index=None.")

    return TrialArtifact(
        vector_index=vector_index,
        profiles=profiles,
        routing=routing,
        direct_transform=direct_transform,
        steering_variant=steering_variant,
        recipe=recipe,
    )


def apply_trial_artifact(config, artifact: TrialArtifact):
    """Apply saved steering semantics to a config before replay/export."""
    if config.model.backend != "hf":
        raise ValueError(
            "Trial materialization currently requires backend='hf'. A vLLM "
            "optimization config cannot be passed directly to the HF replay "
            "path; use an HF twin config for the same base model."
        )
    if artifact.recipe:
        from abliterix.settings import SteeringConfig

        saved_model = artifact.recipe.get("model")
        if isinstance(saved_model, dict):
            saved_model_id = saved_model.get("model_id")
            if saved_model_id and saved_model_id != config.model.model_id:
                raise ValueError(
                    "Trial was optimized for a different base model: "
                    f"{saved_model_id!r} != {config.model.model_id!r}."
                )

        saved = artifact.recipe.get("steering")
        if not isinstance(saved, dict):
            raise ValueError("Trial replay artifact is missing its steering snapshot.")
        expected_fields = set(SteeringConfig.model_fields)
        actual_fields = set(saved)
        if actual_fields != expected_fields:
            missing = sorted(expected_fields - actual_fields)
            unknown = sorted(actual_fields - expected_fields)
            raise ValueError(
                "Trial steering snapshot is incompatible with this Abliterix "
                f"version (missing fields: {missing}; unknown fields: {unknown})."
            )
        config.steering = SteeringConfig.model_validate(saved)

        direction_inputs = artifact.recipe.get("direction_inputs")
        if isinstance(direction_inputs, dict):
            required = {"system_prompt", "benign_prompts", "target_prompts"}
            missing_inputs = sorted(required - set(direction_inputs))
            if missing_inputs:
                raise ValueError(
                    "Trial direction provenance is incomplete; missing "
                    f"{missing_inputs}."
                )
            prompt_source_type = type(config.benign_prompts)
            config.system_prompt = str(direction_inputs["system_prompt"])
            config.benign_prompts = prompt_source_type.model_validate(
                direction_inputs["benign_prompts"]
            )
            config.target_prompts = prompt_source_type.model_validate(
                direction_inputs["target_prompts"]
            )
    elif artifact.direct_transform is not None:
        # Legacy checkpoints with search_direct_transform stored only the
        # sampled transform as a top-level user attribute.
        config.steering.direct_transform = artifact.direct_transform
    return config


def select_trial_vectors(
    artifact: TrialArtifact,
    default_vectors,
    *,
    benign_states,
    target_states,
    config,
):
    """Reconstruct the exact vector variant selected by the trial."""
    if artifact.steering_variant == "single":
        return default_vectors
    if artifact.steering_variant == "harmfulness_pair":
        if benign_states is None or target_states is None:
            raise ValueError(
                "The harmfulness_pair trial requires benign and target residuals "
                "for replay."
            )
        from abliterix.harmfulness import extract_harm_refusal_pair

        return extract_harm_refusal_pair(
            benign_states,
            target_states,
            layer_band=tuple(config.steering.harmfulness_layer_band),
            orthogonal_projection=config.steering.orthogonal_projection,
            projected_abliteration=config.steering.projected_abliteration,
        )
    raise ValueError(
        f"Unknown steering vector variant {artifact.steering_variant!r}; "
        "refusing to replay a different method."
    )


def compute_trial_vectors(
    artifact: TrialArtifact,
    benign_states,
    target_states,
    config,
):
    """Compute the configured base recipe and select the trial's variant."""
    from abliterix.types import SteeringMode

    unsupported_runtime_state: list[str] = []
    if config.iterative.enabled:
        unsupported_runtime_state.append("iterative multi-pass vectors")
    if config.steering.cliff_head_ablation:
        unsupported_runtime_state.append("cliff-head edits and selected head IDs")
    if config.steering.steering_mode == SteeringMode.VECTOR_FIELD:
        unsupported_runtime_state.append("trained vector-field concept scorers")
    if unsupported_runtime_state:
        details = ", ".join(unsupported_runtime_state)
        raise RuntimeError(
            "Exact trial replay requires saved runtime state that TrialArtifact "
            f"v1 does not contain: {details}. Refusing to recompute a different "
            "method; rerun/export from the live optimization session or use a "
            "future vector-sidecar artifact."
        )

    from abliterix.vectors import compute_configured_steering_vectors

    default_vectors = compute_configured_steering_vectors(
        benign_states, target_states, config
    )
    return select_trial_vectors(
        artifact,
        default_vectors,
        benign_states=benign_states,
        target_states=target_states,
        config=config,
    )


def extract_trial_params(trial):
    """Backward-compatible tuple view of :func:`extract_trial_artifact`."""
    artifact = extract_trial_artifact(trial)
    return artifact.vector_index, artifact.profiles, artifact.routing
