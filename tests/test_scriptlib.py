"""Tests for abliterix.scriptlib — trial parameter extraction.

Uses mock trial objects (no checkpoint files needed).
"""

from types import SimpleNamespace

import pytest
import torch

from abliterix.scriptlib import (
    apply_trial_artifact,
    build_trial_recipe,
    compute_trial_vectors,
    extract_trial_artifact,
    extract_trial_params,
    select_trial_vectors,
)
from abliterix.settings import AbliterixConfig
from abliterix.types import (
    DecayKernel,
    DirectTransform,
    ExpertRoutingConfig,
    SteeringProfile,
)


def _trial_with(attrs: dict) -> SimpleNamespace:
    return SimpleNamespace(user_attrs=attrs)


# ---------------------------------------------------------------------------
# extract_trial_params — dense models
# ---------------------------------------------------------------------------


def test_extract_dense_model(sample_trial_attrs):
    trial = _trial_with(sample_trial_attrs)
    vi, profiles, routing = extract_trial_params(trial)
    assert vi == 5.5
    assert isinstance(profiles, dict)
    assert routing is None


def test_profiles_are_steering_profile(sample_trial_attrs):
    trial = _trial_with(sample_trial_attrs)
    _, profiles, _ = extract_trial_params(trial)
    for v in profiles.values():
        assert isinstance(v, SteeringProfile)


def test_extract_per_layer():
    attrs = {
        "vector_index": None,
        "parameters": {
            "q_proj": {
                "max_weight": 1.0,
                "max_weight_position": 5.0,
                "min_weight": 0.1,
                "min_weight_distance": 2.0,
            }
        },
    }
    vi, _, _ = extract_trial_params(_trial_with(attrs))
    assert vi is None


# ---------------------------------------------------------------------------
# extract_trial_params — MoE models
# ---------------------------------------------------------------------------


def test_extract_moe_model(sample_trial_attrs):
    attrs = {
        **sample_trial_attrs,
        "moe_parameters": {
            "n_suppress": 5,
            "router_bias": -3.0,
            "expert_ablation_weight": 1.5,
        },
    }
    _, _, routing = extract_trial_params(_trial_with(attrs))
    assert isinstance(routing, ExpertRoutingConfig)
    assert routing.n_suppress == 5
    assert routing.router_bias == -3.0
    assert routing.expert_ablation_weight == 1.5


def test_missing_moe_parameters(sample_trial_attrs):
    """Without moe_parameters key, routing should be None."""
    trial = _trial_with(sample_trial_attrs)
    _, _, routing = extract_trial_params(trial)
    assert routing is None


# ---------------------------------------------------------------------------
# Versioned replay artifact
# ---------------------------------------------------------------------------


def test_extract_trial_artifact_preserves_sampled_semantics(sample_trial_attrs):
    attrs = {
        **sample_trial_attrs,
        "direct_transform": "householder",
        "steering_variant": "harmfulness_pair",
        "steering_recipe": {
            "schema_version": 1,
            "steering": {
                "direct_transform": "householder",
                "decay_kernel": "cosine",
                "fixed_vector_scope": "global",
            },
        },
    }

    artifact = extract_trial_artifact(_trial_with(attrs))

    assert artifact.direct_transform == DirectTransform.HOUSEHOLDER
    assert artifact.steering_variant == "harmfulness_pair"
    assert artifact.recipe["schema_version"] == 1


def test_apply_trial_artifact_restores_saved_steering_recipe(sample_trial_attrs):
    config = AbliterixConfig(model={"model_id": "dummy/model"})
    saved = config.steering.model_dump(mode="json")
    saved.update(
        direct_transform="biprojected",
        decay_kernel="cosine",
        fixed_vector_scope="global",
    )
    attrs = {
        **sample_trial_attrs,
        "direct_transform": "biprojected",
        "steering_recipe": {"schema_version": 1, "steering": saved},
    }

    artifact = extract_trial_artifact(_trial_with(attrs))
    apply_trial_artifact(config, artifact)

    assert config.steering.direct_transform == DirectTransform.BIPROJECTED
    assert config.steering.decay_kernel == DecayKernel.COSINE
    assert config.steering.fixed_vector_scope == "global"


def test_build_trial_recipe_freezes_actual_sampled_choices():
    config = AbliterixConfig(model={"model_id": "dummy/model"})

    recipe = build_trial_recipe(
        config,
        direct_transform=DirectTransform.HOUSEHOLDER,
        steering_variant="harmfulness_pair",
        vector_scope="per layer",
    )

    assert recipe["schema_version"] == 1
    assert recipe["steering"]["direct_transform"] == "householder"
    assert recipe["steering"]["fixed_vector_scope"] == "per layer"
    assert recipe["steering_variant"] == "harmfulness_pair"
    assert recipe["model"]["model_id"] == "dummy/model"
    assert recipe["direction_inputs"]["system_prompt"] == config.system_prompt


def test_apply_trial_artifact_restores_direction_inputs(sample_trial_attrs):
    source = AbliterixConfig(model={"model_id": "dummy/model"})
    source.system_prompt = "frozen system prompt"
    source.benign_prompts.split = "train[:17]"
    recipe = build_trial_recipe(
        source,
        direct_transform=source.steering.direct_transform,
        steering_variant="single",
        vector_scope="global",
    )
    attrs = {**sample_trial_attrs, "steering_recipe": recipe}
    replay = AbliterixConfig(model={"model_id": "dummy/model"})

    apply_trial_artifact(replay, extract_trial_artifact(_trial_with(attrs)))

    assert replay.system_prompt == "frozen system prompt"
    assert replay.benign_prompts.split == "train[:17]"


def test_apply_trial_artifact_rejects_different_base_model(sample_trial_attrs):
    source = AbliterixConfig(model={"model_id": "original/model"})
    recipe = build_trial_recipe(
        source,
        direct_transform=source.steering.direct_transform,
        steering_variant="single",
        vector_scope="global",
    )
    attrs = {**sample_trial_attrs, "steering_recipe": recipe}
    replay = AbliterixConfig(model={"model_id": "different/model"})

    with pytest.raises(ValueError, match="different base model"):
        apply_trial_artifact(replay, extract_trial_artifact(_trial_with(attrs)))


def test_select_trial_vectors_rejects_unknown_variant(sample_trial_attrs):
    attrs = {**sample_trial_attrs, "steering_variant": "unknown"}
    artifact = extract_trial_artifact(_trial_with(attrs))
    config = AbliterixConfig(model={"model_id": "dummy/model"})

    with pytest.raises(ValueError, match="Unknown steering vector variant"):
        select_trial_vectors(
            artifact,
            torch.zeros(2, 3),
            benign_states=torch.zeros(1, 2, 3),
            target_states=torch.zeros(1, 2, 3),
            config=config,
        )


def test_extract_trial_artifact_rejects_recipe_conflict(sample_trial_attrs):
    attrs = {
        **sample_trial_attrs,
        "direct_transform": "householder",
        "steering_recipe": {
            "schema_version": 1,
            "steering": {"direct_transform": "standard"},
        },
    }

    with pytest.raises(ValueError, match="direct_transform conflicts"):
        extract_trial_artifact(_trial_with(attrs))


def test_extract_trial_artifact_rejects_scope_index_mismatch(sample_trial_attrs):
    attrs = {
        **sample_trial_attrs,
        "vector_index": 4.0,
        "steering_recipe": {
            "schema_version": 1,
            "steering": {"fixed_vector_scope": "per layer"},
        },
    }

    with pytest.raises(ValueError, match="vector_index=None"):
        extract_trial_artifact(_trial_with(attrs))


def test_apply_trial_artifact_rejects_vllm_materialization(sample_trial_attrs):
    config = AbliterixConfig(model={"model_id": "dummy/model", "backend": "vllm"})
    artifact = extract_trial_artifact(_trial_with(sample_trial_attrs))

    with pytest.raises(ValueError, match="requires backend='hf'"):
        apply_trial_artifact(config, artifact)


def test_apply_trial_artifact_rejects_partial_versioned_snapshot(sample_trial_attrs):
    attrs = {
        **sample_trial_attrs,
        "steering_recipe": {
            "schema_version": 1,
            "steering": {"direct_transform": "standard"},
        },
    }
    config = AbliterixConfig(model={"model_id": "dummy/model"})
    artifact = extract_trial_artifact(_trial_with(attrs))

    with pytest.raises(ValueError, match="incompatible"):
        apply_trial_artifact(config, artifact)


@pytest.mark.parametrize(
    "config_patch, expected",
    [
        ({"iterative": {"enabled": True}}, "iterative multi-pass"),
        (
            {"steering": {"cliff_head_ablation": True}},
            "cliff-head edits",
        ),
        (
            {"steering": {"steering_mode": "vector_field"}},
            "concept scorers",
        ),
    ],
)
def test_compute_trial_vectors_rejects_unrecorded_runtime_state(
    sample_trial_attrs, config_patch, expected
):
    config = AbliterixConfig(model={"model_id": "dummy/model"}, **config_patch)
    artifact = extract_trial_artifact(_trial_with(sample_trial_attrs))

    with pytest.raises(RuntimeError, match=expected):
        compute_trial_vectors(
            artifact,
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 3),
            config,
        )
