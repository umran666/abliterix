"""Regression test for method isolation in scripts/ab_test.py."""

from abliterix.settings import AbliterixConfig
from scripts.ab_test import _clone_method_config


def test_ab_method_config_clone_does_not_share_nested_settings():
    original = AbliterixConfig(model={"model_id": "dummy/model"})
    clone = _clone_method_config(original)

    clone.steering.projected_abliteration = True
    clone.steering.discriminative_layer_selection = True

    assert original.steering.projected_abliteration is False
    assert original.steering.discriminative_layer_selection is False
