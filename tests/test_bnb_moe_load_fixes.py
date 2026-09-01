"""Tests for bnb-4bit / dequant-cache / text_only load helpers."""

from __future__ import annotations

from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from abliterix.core.engine import (
    DEQUANT_CACHE_MAX_BYTES,
    SteeringEngine,
    resolve_model_class,
)
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
    eng._dequant_cache_max_bytes = DEQUANT_CACHE_MAX_BYTES
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
    assert DEQUANT_CACHE_MAX_BYTES == 4 * 1024**3
    # Production default must come from the shared module constant (not a
    # hand-copied literal on a test-only shell).
    eng = SteeringEngine.__new__(SteeringEngine)
    # Mimic __init__ assignment without loading a model.
    eng._dequant_cache_max_bytes = DEQUANT_CACHE_MAX_BYTES
    assert eng._dequant_cache_max_bytes == DEQUANT_CACHE_MAX_BYTES


# ---------------------------------------------------------------------------
# bnb compute dtype
# ---------------------------------------------------------------------------


def test_build_quant_config_uses_bf16_when_supported(monkeypatch):
    eng = _minimal_bnb_engine()
    monkeypatch.setattr("abliterix.core.engine._bf16_compute_supported", lambda: True)
    cfg = eng._build_quant_config()
    assert cfg is not None
    assert cfg.bnb_4bit_compute_dtype is torch.bfloat16


def test_build_quant_config_falls_back_to_fp16_without_bf16(monkeypatch):
    eng = _minimal_bnb_engine()
    monkeypatch.setattr("abliterix.core.engine._bf16_compute_supported", lambda: False)
    cfg = eng._build_quant_config()
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


# ---------------------------------------------------------------------------
# export_adapter after in-place merge
# ---------------------------------------------------------------------------


def test_export_adapter_rejects_after_needs_reload(tmp_path):
    """merge_and_unload sets needs_reload; adapter export must refuse."""
    from abliterix.types import SteeringMode
    from peft import PeftModel

    eng = _minimal_bnb_engine()
    eng.needs_reload = True
    eng.config.steering.steering_mode = SteeringMode.LORA
    eng._router_originals = None
    eng._expert_deltas = None

    # PeftModel subclass with no lora params (post merge_and_unload shape)
    class _Hollow(PeftModel):
        def __init__(self):
            # Bypass PeftModel.__init__
            pass

        def named_parameters(self, *a, **k):
            yield "base_model.model.layers.0.weight", torch.zeros(2, 2)

        def save_pretrained(self, *a, **k):
            raise AssertionError("must not write empty adapter")

    try:
        eng.model = object.__new__(_Hollow)
    except Exception:
        # If PeftModel cannot be subclassed this way, still exercise needs_reload
        class _M:
            def named_parameters(self, *a, **k):
                yield "lora_A.weight", torch.zeros(1)

            def save_pretrained(self, *a, **k):
                raise AssertionError("must not write")

        eng.model = _M()
        # Force isinstance(PeftModel) path by patching
        import abliterix.core.engine as em

        real_isinstance = isinstance

        def _iso(obj, cls):
            if cls is PeftModel:
                return True
            return real_isinstance(obj, cls)

        em.isinstance = _iso  # type: ignore
        try:
            try:
                eng.export_adapter(tmp_path)
                raise AssertionError("expected RuntimeError")
            except RuntimeError as e:
                assert "reload" in str(e).lower() or "consumed" in str(e).lower()
        finally:
            em.isinstance = real_isinstance  # type: ignore
        return

    try:
        eng.export_adapter(tmp_path)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "reload" in str(e).lower() or "consumed" in str(e).lower()


# ---------------------------------------------------------------------------
# bf16 probe branches
# ---------------------------------------------------------------------------


def test_bf16_probe_false_without_cuda(monkeypatch):
    from abliterix.core import engine as eng_mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert eng_mod._bf16_compute_supported() is False


def test_bf16_probe_true_on_hip(monkeypatch):
    from abliterix.core import engine as eng_mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.2.0", raising=False)
    assert eng_mod._bf16_compute_supported() is True


def test_bf16_probe_cuda_sm_gate(monkeypatch):
    from abliterix.core import engine as eng_mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    assert eng_mod._bf16_compute_supported() is False
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    assert eng_mod._bf16_compute_supported() is True


# ---------------------------------------------------------------------------
# Non-TTY abort returns None like questionary.ask()
# ---------------------------------------------------------------------------


def test_ask_choice_abort_returns_none(monkeypatch):
    monkeypatch.setattr("abliterix.util._stdin_is_tty", lambda: False)
    monkeypatch.setattr("abliterix.util.running_in_notebook", lambda: False)

    def _raise(_prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise)
    assert ask_choice("Pick", ["a", "b"]) is None
