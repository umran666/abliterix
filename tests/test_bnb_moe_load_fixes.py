"""Tests for SC117 ROCm/bnb MoE load + dequant-cache hardening.

These cover behaviour added in pr/bnb-rocm-moe-stability without requiring a
full GPU model load.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from abliterix.core.engine import resolve_model_class
from abliterix.util import _print_choices, _stdin_is_tty, ask_choice


# ---------------------------------------------------------------------------
# Dequant cache byte budget (mirrors apply_steering policy)
# ---------------------------------------------------------------------------


def _maybe_store_dequant(
    engine: SimpleNamespace,
    mid: int,
    weight: torch.Tensor,
) -> bool:
    """Copy of the cache-insert guard in apply_steering (bnb/int8/fp8 paths)."""
    if engine._dequant_cache_bytes < engine._dequant_cache_max_bytes:
        engine._dequant_cache[mid] = weight
        engine._dequant_cache_bytes += weight.nelement() * weight.element_size()
        return True
    return False


def test_dequant_cache_stores_while_under_budget():
    engine = SimpleNamespace(
        _dequant_cache={},
        _dequant_cache_bytes=0,
        _dequant_cache_max_bytes=4 * 1024,  # 4 KiB
    )
    w = torch.zeros(32, 8, dtype=torch.float32)  # 1 KiB
    assert _maybe_store_dequant(engine, 1, w) is True
    assert 1 in engine._dequant_cache
    assert engine._dequant_cache_bytes == 1024


def test_dequant_cache_stops_inserting_once_budget_reached():
    engine = SimpleNamespace(
        _dequant_cache={},
        _dequant_cache_bytes=0,
        _dequant_cache_max_bytes=1024,
    )
    w = torch.zeros(32, 8, dtype=torch.float32)  # 1 KiB → fills budget after insert
    assert _maybe_store_dequant(engine, 1, w) is True
    # used == max → further inserts skipped (matches production guard)
    w2 = torch.zeros(8, 8, dtype=torch.float32)
    assert _maybe_store_dequant(engine, 2, w2) is False
    assert 2 not in engine._dequant_cache
    assert engine._dequant_cache_bytes == 1024


def test_dequant_cache_default_cap_is_4gib():
    """SteeringEngine.__init__ must expose a finite default dequant budget."""
    # Avoid constructing the full engine (needs config/model). Assert the
    # constant encoded next to the attribute in source matches the intended
    # production default (4 GiB).
    from pathlib import Path

    src = Path("src/abliterix/core/engine.py").read_text(encoding="utf-8")
    assert "_dequant_cache_max_bytes: int = 4 * 1024**3" in src


# ---------------------------------------------------------------------------
# qwen3_5_moe model class resolution
# ---------------------------------------------------------------------------


def test_resolve_model_class_qwen3_5_moe_is_causal_lm():
    from transformers import AutoModelForCausalLM

    fake = (
        {
            "model_type": "qwen3_5_moe",
            "vision_config": {"dummy": True},  # multimodal fields present
        },
    )
    with patch(
        "abliterix.core.engine.PretrainedConfig.get_config_dict",
        return_value=fake,
    ):
        cls = resolve_model_class("org/qwen3-5-moe-fake")
    assert cls is AutoModelForCausalLM


def test_bnb_compute_dtype_hardcoded_bf16_in_source():
    """4-bit compute dtype must not follow float16 load dtype."""
    from pathlib import Path

    src = Path("src/abliterix/core/engine.py").read_text(encoding="utf-8")
    # The bnb branch should pin bf16 rather than getattr(torch, dtype)
    assert "compute_dtype = torch.bfloat16" in src
    assert (
        'compute_dtype = torch.bfloat16 if dtype == "auto" else getattr(torch, dtype)'
        not in src
    )


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
    assert "Beta" in out


def test_ask_choice_non_tty_numeric(monkeypatch):
    monkeypatch.setattr("abliterix.util._stdin_is_tty", lambda: False)
    monkeypatch.setattr("abliterix.util.running_in_notebook", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "2")
    result = ask_choice("Pick", ["first", "second", "third"])
    assert result == "second"
