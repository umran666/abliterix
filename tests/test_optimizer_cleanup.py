"""Regression tests for per-trial optimizer cleanup."""

from __future__ import annotations

import pytest
import torch
from optuna.storages import InMemoryStorage
from types import SimpleNamespace

from abliterix.optimizer import run_search
from abliterix.settings import AbliterixConfig
from abliterix.types import DirectTransform


class _HFEngine:
    def __init__(self) -> None:
        self.restore_calls = 0

    def get_n_layers(self) -> int:
        return 4

    def list_steerable_components(self) -> list[str]:
        return []

    def restore_baseline(self) -> None:
        self.restore_calls += 1


class _InPlaceGenerator:
    def __init__(self) -> None:
        self.attention_editor = SimpleNamespace(
            _attn_layers={0, 1, 2, 3},
            _applied=False,
        )
        self.expert_editor = None
        self.moe_editor = None
        self.attention_restore_calls = 0
        self.expert_restore_calls = 0
        self.router_restore_calls = 0
        self.mutated = False

    def apply_attention_projection(self, *_args, **_kwargs):
        self.mutated = True
        self.attention_editor._applied = True
        return {"applied": 1, "errors": [], "per_layer": []}

    def restore_attention_weights(self) -> int:
        if not self.attention_editor._applied:
            return 0
        self.attention_restore_calls += 1
        self.mutated = False
        self.attention_editor._applied = False
        return 1

    def restore_expert_weights(self) -> int:
        self.expert_restore_calls += 1
        return 0

    def restore_router_suppression(self) -> int:
        self.router_restore_calls += 1
        return 0


class _FailingInPlaceGenerator(_InPlaceGenerator):
    def apply_attention_projection(self, *_args, **_kwargs):
        self.mutated = True
        raise RuntimeError("partial in-place apply")


class _InPlaceEngine:
    def __init__(self, generator=None) -> None:
        self._vllm_gen = generator or _InPlaceGenerator()
        self._projection_cache = None
        self._current_adapter_path = None
        self._cached_n_layers = 4

    def get_n_layers(self) -> int:
        return 4

    def list_steerable_components(self) -> list[str]:
        return ["attn.q_proj"]


class _Scorer:
    target_msgs: list = []
    detector = SimpleNamespace(evaluate_compliance=lambda *_args: 0)

    def measure_kl_and_coherence(self, _engine):
        return 0.1, 0.0

    def _compute_objectives(self, kl, detected, _length_dev):
        return kl, float(detected)


class _ProjectionCache:
    target_modules: list[str] = []

    def build_lora_weights(self, *_args, **_kwargs):
        return {}


class _FailingRouterGenerator:
    attention_editor = None
    expert_editor = None
    _lora_disabled = False

    def __init__(self) -> None:
        self.moe_editor = SimpleNamespace(_applied=False)
        self.mutated = False
        self.router_restore_calls = 0

    def apply_router_suppression(self, **_kwargs):
        self.mutated = True
        raise RuntimeError("partial router apply")

    def restore_router_suppression(self) -> int:
        if not self.moe_editor._applied:
            return 0
        self.router_restore_calls += 1
        self.mutated = False
        self.moe_editor._applied = False
        return 1


class _RouterEngine:
    def __init__(self) -> None:
        self._vllm_gen = _FailingRouterGenerator()
        self._projection_cache = _ProjectionCache()
        self._current_adapter_path = "/stale/adapter"

    def get_n_layers(self) -> int:
        return 4

    def list_steerable_components(self) -> list[str]:
        return []


def test_apply_failure_restores_hf_model_and_sampled_direct_transform(monkeypatch):
    """A failed trial leaves both model and mutable recipe at baseline."""
    config = AbliterixConfig(
        model={"model_id": "dummy/model"},
        steering={
            "steering_mode": "direct",
            "direct_transform": "standard",
            "search_direct_transform": True,
            "search_direct_transform_choices": ["orba"],
        },
        optimization={"num_trials": 1, "num_warmup_trials": 1},
    )
    engine = _HFEngine()
    vectors = torch.randn(5, 8)

    def fail_after_partial_apply(*_args, **_kwargs):
        raise RuntimeError("partial HF apply")

    monkeypatch.setattr(
        "abliterix.optimizer.apply_steering",
        fail_after_partial_apply,
    )

    with pytest.raises(RuntimeError, match="partial HF apply"):
        run_search(
            config,
            engine,
            scorer=object(),
            steering_vectors=vectors,
            safety_experts=None,
            storage=InMemoryStorage(),
        )

    assert config.steering.direct_transform == DirectTransform.STANDARD
    # One reset before apply, then one cleanup after the failed apply.
    assert engine.restore_calls == 2


def test_successful_in_place_trial_runs_each_cleanup_once():
    """The unified trial transaction must not double-restore vLLM editors."""
    config = AbliterixConfig(
        model={"model_id": "dummy/model"},
        steering={"fixed_vector_scope": "per layer"},
        optimization={"num_trials": 1, "num_warmup_trials": 1},
    )
    engine = _InPlaceEngine()

    study = run_search(
        config,
        engine,
        scorer=_Scorer(),
        steering_vectors=torch.randn(5, 8),
        safety_experts=None,
        storage=InMemoryStorage(),
    )

    assert len(study.trials) == 1
    assert engine._vllm_gen.attention_restore_calls == 1
    assert engine._vllm_gen.expert_restore_calls == 1
    assert engine._vllm_gen.router_restore_calls == 1
    assert engine._vllm_gen.mutated is False


def test_partial_in_place_apply_failure_still_restores_every_editor():
    """Cleanup intent is established before the first vLLM RPC mutation."""
    config = AbliterixConfig(
        model={"model_id": "dummy/model"},
        steering={"fixed_vector_scope": "per layer"},
        optimization={"num_trials": 1, "num_warmup_trials": 1},
    )
    generator = _FailingInPlaceGenerator()
    engine = _InPlaceEngine(generator)

    with pytest.raises(RuntimeError, match="partial in-place apply"):
        run_search(
            config,
            engine,
            scorer=object(),
            steering_vectors=torch.randn(5, 8),
            safety_experts=None,
            storage=InMemoryStorage(),
        )

    assert generator.attention_restore_calls == 1
    assert generator.expert_restore_calls == 1
    assert generator.router_restore_calls == 1
    assert generator.mutated is False
    assert engine._current_adapter_path is None


def test_partial_router_apply_failure_restores_router_and_adapter_state():
    """Router cleanup is armed before collective RPC can partially mutate."""
    config = AbliterixConfig(
        model={"model_id": "dummy/model"},
        steering={"fixed_vector_scope": "per layer"},
        optimization={"num_trials": 1, "num_warmup_trials": 1},
    )
    engine = _RouterEngine()

    with pytest.raises(RuntimeError, match="partial router apply"):
        run_search(
            config,
            engine,
            scorer=object(),
            steering_vectors=torch.randn(5, 8),
            safety_experts={0: [(0, 1.0)]},
            storage=InMemoryStorage(),
        )

    assert engine._vllm_gen.router_restore_calls == 1
    assert engine._vllm_gen.mutated is False
    assert engine._current_adapter_path is None
