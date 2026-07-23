"""Regression contracts for canonical trial replay entry points."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("script", "benign_name", "target_name"),
    [
        ("export_model.py", "benign_states", "target_states"),
        ("inspect_trial_judge.py", "benign_states", "target_states"),
        ("test_trial.py", "benign_states", "target_states"),
        ("test_two_trials.py", "benign_states", "target_states"),
        ("upload_model.py", "good_residuals", "bad_residuals"),
        ("inspect_refusals.py", "benign_residuals", "target_residuals"),
    ],
)
def test_replay_script_keeps_and_passes_residuals_until_steering(
    script: str,
    benign_name: str,
    target_name: str,
):
    """Transforms and layer selection must see the training residuals."""
    tree = ast.parse((ROOT / "scripts" / script).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apply_steering"
    ]
    assert calls, f"{script} has no apply_steering call"

    for call in calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert isinstance(keywords.get("benign_states"), ast.Name)
        assert keywords["benign_states"].id == benign_name
        assert isinstance(keywords.get("target_states"), ast.Name)
        assert keywords["target_states"].id == target_name

        for node in ast.walk(tree):
            if not isinstance(node, ast.Delete):
                continue
            deleted = {
                child.id
                for target in node.targets
                for child in ast.walk(target)
                if isinstance(child, ast.Name)
            }
            if {benign_name, target_name} & deleted:
                assert node.lineno > call.lineno, (
                    f"{script} releases replay residuals before apply_steering"
                )


def test_export_script_uses_engine_export_contract():
    tree = ast.parse((ROOT / "scripts" / "export_model.py").read_text(encoding="utf-8"))
    attrs = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]

    assert "export_merged" in attrs
    assert "export_adapter" in attrs
    assert "merge_and_unload" not in attrs


def test_interactive_trial_restore_replays_full_artifact(monkeypatch):
    from abliterix import interactive

    artifact = SimpleNamespace(
        vector_index=None,
        profiles={"q_proj": object()},
        routing=object(),
        steering_variant="harmfulness_pair",
    )
    original_config = Mock()
    trial_config = Mock()
    original_config.model_copy.return_value = trial_config
    engine = SimpleNamespace(config=original_config)
    engine.restore_baseline = Mock()
    trial = object()
    default_vectors = object()
    selected_vectors = object()
    benign_states = object()
    target_states = object()

    extract = Mock(return_value=artifact)
    apply_artifact = Mock()
    select = Mock(return_value=selected_vectors)
    apply = Mock()
    monkeypatch.setattr(interactive, "extract_trial_artifact", extract)
    monkeypatch.setattr(interactive, "apply_trial_artifact", apply_artifact)
    monkeypatch.setattr(interactive, "select_trial_vectors", select)
    monkeypatch.setattr(interactive, "apply_steering", apply)

    restored = interactive._restore_selected_trial(
        trial,
        original_config,
        engine,
        default_vectors,
        safety_experts={0: [(1, 0.5)]},
        benign_states=benign_states,
        target_states=target_states,
        steering_vector_variants={"single": default_vectors},
    )

    assert restored is trial_config
    assert engine.config is trial_config
    original_config.model_copy.assert_called_once_with(deep=True)
    extract.assert_called_once_with(trial)
    apply_artifact.assert_called_once_with(trial_config, artifact)
    select.assert_called_once_with(
        artifact,
        default_vectors,
        benign_states=benign_states,
        target_states=target_states,
        config=trial_config,
    )
    engine.restore_baseline.assert_called_once_with()
    apply.assert_called_once_with(
        engine,
        selected_vectors,
        artifact.vector_index,
        artifact.profiles,
        trial_config,
        safety_experts={0: [(1, 0.5)]},
        routing_config=artifact.routing,
        benign_states=benign_states,
        target_states=target_states,
    )
    assert engine.config is not original_config


def test_interactive_missing_variant_inputs_fail_before_model_mutation(monkeypatch):
    from abliterix import interactive

    artifact = SimpleNamespace(steering_variant="harmfulness_pair")
    original_config = Mock()
    original_config.model_copy.return_value = Mock()
    engine = SimpleNamespace(config=original_config, restore_baseline=Mock())
    monkeypatch.setattr(
        interactive, "extract_trial_artifact", Mock(return_value=artifact)
    )
    monkeypatch.setattr(interactive, "apply_trial_artifact", Mock())
    monkeypatch.setattr(
        interactive,
        "select_trial_vectors",
        Mock(side_effect=ValueError("requires benign and target residuals")),
    )
    apply = Mock()
    monkeypatch.setattr(interactive, "apply_steering", apply)

    with pytest.raises(ValueError, match="requires benign and target residuals"):
        interactive._restore_selected_trial(
            object(),
            original_config,
            engine,
            object(),
            safety_experts=None,
        )

    assert engine.config is original_config
    engine.restore_baseline.assert_not_called()
    apply.assert_not_called()


def test_webui_replays_selected_trial_instead_of_empty_profiles():
    source = (ROOT / "src" / "abliterix" / "webui.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    replay_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_restore_session_trial"
    ]

    assert len(replay_calls) >= 5  # post-search, compare, chat, save, upload
    assert "profiles={}" not in source
    assert "compute_configured_steering_vectors" in source
    assert "selected_trial" in source
