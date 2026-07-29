"""Tests for bnb-4bit / dequant-cache / text_only load helpers."""

from __future__ import annotations

from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from abliterix.core.engine import SteeringEngine, resolve_model_class
from abliterix.settings import AbliterixConfig
from abliterix.types import QuantMode
from abliterix.util import (
    _print_choices,
    _stdin_is_tty,
    ask_choice,
    ask_path,
    ask_secret,
)


def _minimal_bnb_engine() -> SteeringEngine:
    """Lightweight engine shell for unit tests (no real model load)."""
    eng = SteeringEngine.__new__(SteeringEngine)
    eng.config = AbliterixConfig(
        model={"model_id": "org/test-model", "quant_method": QuantMode.BNB_4BIT}
    )
    eng._dequant_cache = {}
    eng._dequant_cache_bytes = 0
    eng._dequant_cache_max_bytes = 4 * 1024**3
    return eng


# ---------------------------------------------------------------------------
# Dequant cache helper (production code path)
# ---------------------------------------------------------------------------


def test_cache_dequant_stores_while_under_budget():
    eng = _minimal_bnb_engine()
    eng._dequant_cache_max_bytes = 4 * 1024
    w = torch.zeros(32, 8, dtype=torch.float32)  # 1 KiB
    eng._cache_dequant(1, w)
    assert 1 in eng._dequant_cache
    assert eng._dequant_cache_bytes == 1024


def test_cache_dequant_stops_once_budget_reached():
    eng = _minimal_bnb_engine()
    eng._dequant_cache_max_bytes = 1024
    eng._cache_dequant(1, torch.zeros(32, 8, dtype=torch.float32))
    eng._cache_dequant(2, torch.zeros(8, 8, dtype=torch.float32))
    assert 1 in eng._dequant_cache
    assert 2 not in eng._dequant_cache
    assert eng._dequant_cache_bytes == 1024


def test_dequant_cache_default_cap_is_4gib():
    eng = _minimal_bnb_engine()
    assert eng._dequant_cache_max_bytes == 4 * 1024**3


# ---------------------------------------------------------------------------
# bnb compute dtype
# ---------------------------------------------------------------------------


def test_build_quant_config_uses_bf16_when_supported(monkeypatch):
    eng = _minimal_bnb_engine()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    cfg = eng._build_quant_config("float16")
    assert cfg is not None
    assert cfg.bnb_4bit_compute_dtype is torch.bfloat16


def test_build_quant_config_falls_back_to_fp16_without_bf16(monkeypatch):
    eng = _minimal_bnb_engine()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    cfg = eng._build_quant_config("float16")
    assert cfg is not None
    assert cfg.bnb_4bit_compute_dtype is torch.float16


# ---------------------------------------------------------------------------
# text_only model class resolution
# ---------------------------------------------------------------------------


def test_resolve_model_class_text_only_forces_causal_lm():
    fake = ({"model_type": "qwen3_5_moe", "vision_config": {"x": 1}},)
    with patch(
        "abliterix.core.engine.PretrainedConfig.get_config_dict",
        return_value=fake,
    ):
        assert resolve_model_class("org/fake", text_only=True) is AutoModelForCausalLM
        assert (
            resolve_model_class("org/fake", text_only=False)
            is AutoModelForImageTextToText
        )


def test_resolve_model_class_default_unchanged_for_text_models():
    fake = ({"model_type": "llama"},)
    with patch(
        "abliterix.core.engine.PretrainedConfig.get_config_dict",
        return_value=fake,
    ):
        assert resolve_model_class("org/llama") is AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Non-TTY prompt helpers
# ---------------------------------------------------------------------------


def test_stdin_is_tty_callable():
    assert isinstance(_stdin_is_tty(), bool)


def test_print_choices_extracts_values(capsys):
    from questionary import Choice

    values = _print_choices(
        "Pick one",
        [Choice(title="Alpha", value="a"), Choice(title="Beta", value="b")],
    )
    assert values == ["a", "b"]
    out = capsys.readouterr().out
    assert "Alpha" in out


def test_ask_choice_non_tty_numeric(monkeypatch):
    monkeypatch.setattr("abliterix.util._stdin_is_tty", lambda: False)
    monkeypatch.setattr("abliterix.util.running_in_notebook", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "2")
    assert ask_choice("Pick", ["first", "second", "third"]) == "second"


def test_ask_path_non_tty_uses_text(monkeypatch):
    monkeypatch.setattr("abliterix.util._stdin_is_tty", lambda: False)
    monkeypatch.setattr("abliterix.util.running_in_notebook", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "/tmp/adapters")
    assert ask_path("Path:") == "/tmp/adapters"


def test_ask_secret_non_tty_uses_getpass(monkeypatch):
    monkeypatch.setattr("abliterix.util._stdin_is_tty", lambda: False)
    monkeypatch.setattr("abliterix.util.running_in_notebook", lambda: False)
    monkeypatch.setattr("getpass.getpass", lambda _prompt="": "tok")
    assert ask_secret("Token:") == "tok"
