import json

import pytest
from pydantic import ValidationError

from abliterix.eval.benchmark_result import BenchmarkResult, load_benchmark_results


def _valid_result() -> dict:
    harmful_categories = {
        name: 0.1
        for name in (
            "Violence",
            "CBRNE",
            "Sabotage",
            "Self-Harm",
            "Hate Speech",
            "Harassment",
            "Cybercrime",
            "Fraud",
            "Financial Crimes",
            "Weapons",
            "PII",
            "Political Sensitivity",
            "Radicalization",
            "Illegal Drugs",
        )
    }
    benign_categories = harmful_categories | {
        name: 0.0
        for name in (
            "Business",
            "Cooking",
            "Creative Writing",
            "Education",
            "Health & Fitness",
            "History",
            "Mathematics",
            "Programming",
            "Science",
            "Travel",
        )
    }
    return {
        "spec_version": "1.1",
        "model": "org/abliterated-model",
        "base_model": "org/base-model",
        "tier": "self_reported",
        "refusal_rate_total": 0.1,
        "refusal_rate_en": 0.1,
        "refusal_rate_zh": 0.1,
        "refusal_rate_mixed": 0.1,
        "refusal_rate_by_category": harmful_categories,
        "over_refusal_rate_total": 0.02,
        "over_refusal_rate_en": 0.02,
        "over_refusal_rate_zh": 0.02,
        "over_refusal_rate_mixed": 0.02,
        "over_refusal_rate_by_category": benign_categories,
        "damage_metric": {
            "name": "full_distribution_kl",
            "estimator": "mean_per_token_kl",
            "units": "nats/token",
            "value": 0.001,
        },
        "degenerate_rate": 0.01,
        "mean_response_length": 123.0,
        "length_z_score": 0.3,
        "n_harmful": 500,
        "n_benign": 500,
        "judge_model": "google/gemini-3-flash-preview",
        "judge_prompt_sha256": "b" * 64,
        "judge_cache_sha256": "a" * 64,
        "judge_max_attempts": 3,
        "judge_unknown_harmful": 0,
        "judge_unknown_benign": 0,
        "dataset_sha256_harmful": (
            "249e48f403a8a015a808384ef567c7fa6462aaf7c51d4a78c2caeb0509bd769d"
        ),
        "dataset_sha256_benign": (
            "dfffd37646ddb279429323b360a8190e6d971d0d009a6b5f0b877d472cc0a8a7"
        ),
        "commit": "0123456789abcdef" * 2 + "01234567",
        "gpu": "NVIDIA H100 80GB",
        "runtime_seconds": 3600.0,
        "min_new_tokens": 100,
        "max_new_tokens": 150,
    }


def test_frozen_benchmark_contract_accepts_complete_result():
    result = BenchmarkResult.model_validate(_valid_result())

    assert result.model == "org/abliterated-model"
    assert result.damage_metric.name == "full_distribution_kl"


def test_frozen_benchmark_contract_rejects_other_spec_versions():
    record = _valid_result()
    record["spec_version"] = "0.9"

    with pytest.raises(ValidationError, match="spec_version"):
        BenchmarkResult.model_validate(record)


def test_frozen_benchmark_contract_rejects_out_of_range_rates():
    record = _valid_result()
    record["refusal_rate_total"] = 1.01

    with pytest.raises(ValidationError, match="refusal_rate_total"):
        BenchmarkResult.model_validate(record)


def test_frozen_benchmark_contract_rejects_wrong_dataset_hash():
    record = _valid_result()
    record["dataset_sha256_harmful"] = "0" * 64

    with pytest.raises(ValidationError, match="dataset_sha256_harmful"):
        BenchmarkResult.model_validate(record)


def test_frozen_benchmark_contract_rejects_short_generation_floor():
    record = _valid_result()
    record["min_new_tokens"] = 32

    with pytest.raises(ValidationError, match="min_new_tokens"):
        BenchmarkResult.model_validate(record)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_harmful", 499),
        ("n_benign", 499),
        ("max_new_tokens", 149),
        ("judge_model", "different/judge"),
    ],
)
def test_frozen_benchmark_contract_rejects_changed_protocol_constants(field, value):
    record = _valid_result()
    record[field] = value

    with pytest.raises(ValidationError, match=field):
        BenchmarkResult.model_validate(record)


def test_frozen_benchmark_contract_rejects_missing_category_metrics():
    record = _valid_result()
    del record["refusal_rate_by_category"]["Violence"]

    with pytest.raises(ValidationError, match="refusal_rate_by_category"):
        BenchmarkResult.model_validate(record)


def test_frozen_benchmark_contract_rejects_extra_over_refusal_category():
    record = _valid_result()
    record["over_refusal_rate_by_category"]["Unknown"] = 0.0

    with pytest.raises(ValidationError, match="over_refusal_rate_by_category"):
        BenchmarkResult.model_validate(record)


def test_frozen_benchmark_contract_rejects_non_sha256_provenance():
    record = _valid_result()
    record["judge_cache_sha256"] = "not-a-hash"

    with pytest.raises(ValidationError, match="judge_cache_sha256"):
        BenchmarkResult.model_validate(record)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "continuation_nll_drift"),
        ("estimator", "fixed_continuation_mean_absolute_error"),
        ("units", "nats"),
    ],
)
def test_frozen_benchmark_contract_rejects_mislabeled_damage_metric(field, value):
    record = _valid_result()
    record["damage_metric"][field] = value

    with pytest.raises(ValidationError, match=field):
        BenchmarkResult.model_validate(record)


def test_frozen_benchmark_contract_rejects_unknown_judge_labels():
    record = _valid_result()
    record["judge_unknown_harmful"] = 1

    with pytest.raises(ValidationError, match="judge_unknown_harmful"):
        BenchmarkResult.model_validate(record)


def test_frozen_benchmark_contract_rejects_dry_run():
    record = _valid_result()
    record["dry_run"] = True

    with pytest.raises(ValidationError, match="dry_run"):
        BenchmarkResult.model_validate(record)


def test_result_directory_loader_separates_valid_rows_from_contract_issues(tmp_path):
    (tmp_path / "valid.json").write_text(json.dumps(_valid_result()))
    (tmp_path / "invalid.json").write_text(json.dumps({"model": "incomplete"}))

    report = load_benchmark_results(tmp_path)

    assert [item.model for item in report.results] == ["org/abliterated-model"]
    assert report.issues[0].source.name == "invalid.json"
    assert "spec_version" in report.issues[0].message
