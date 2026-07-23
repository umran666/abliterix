from types import SimpleNamespace

import pytest
from huggingface_hub.errors import StrictDataclassClassValidationError
from transformers import PretrainedConfig

from abliterix.core import engine as engine_module


def test_registered_mtp_adapter_truncates_only_the_exact_preflighted_model(
    monkeypatch,
):
    model_id = "org/step-mtp-test"
    monkeypatch.setitem(
        engine_module._MTP_LAYER_TYPE_ADAPTERS,
        model_id,
        (3, 2),
    )

    config = PretrainedConfig(
        name_or_path=model_id,
        num_hidden_layers=2,
        layer_types=["full_attention", "full_attention", "full_attention"],
    )

    assert config.layer_types == ["full_attention", "full_attention"]


def test_registered_mtp_adapter_does_not_hide_changed_config(monkeypatch):
    model_id = "org/step-mtp-changed"
    monkeypatch.setitem(
        engine_module._MTP_LAYER_TYPE_ADAPTERS,
        model_id,
        (3, 2),
    )

    with pytest.raises(StrictDataclassClassValidationError):
        PretrainedConfig(
            name_or_path=model_id,
            num_hidden_layers=2,
            layer_types=["full_attention"] * 4,
        )


def test_unregistered_model_keeps_transformers_validation():
    with pytest.raises(StrictDataclassClassValidationError):
        PretrainedConfig(
            name_or_path="org/unregistered",
            num_hidden_layers=2,
            layer_types=["full_attention"] * 3,
        )


def test_registration_uses_config_metadata_without_touching_cache(monkeypatch):
    model_id = "org/step-mtp-register"
    monkeypatch.setattr(
        PretrainedConfig,
        "get_config_dict",
        lambda *_args, **_kwargs: (
            {
                "num_hidden_layers": 2,
                "layer_types": ["full_attention"] * 3,
            },
            {},
        ),
    )

    engine_module._register_mtp_layer_types_adapter(model_id, True)

    assert engine_module._MTP_LAYER_TYPE_ADAPTERS[model_id] == (3, 2)


def test_adapter_rejects_non_list_layer_types(monkeypatch):
    model_id = "org/step-mtp-tuple"
    monkeypatch.setitem(
        engine_module._MTP_LAYER_TYPE_ADAPTERS,
        model_id,
        (3, 2),
    )
    config = SimpleNamespace(
        _name_or_path=model_id,
        num_hidden_layers=2,
        layer_types=("full_attention",) * 3,
    )

    assert engine_module._adapt_registered_mtp_layer_types(config) is False
    assert len(config.layer_types) == 3
