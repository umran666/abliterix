import json
import math

import pytest
import torch

from scripts.benchmark_qwen35_kernels import (
    batch_padding_stats,
    build_workload_samples,
    compare_logprob_probes,
    compare_reports,
    encode_logprob_probe,
    fingerprint_output,
    summarize_seconds,
)


def test_workload_samples_are_deterministic_and_benign_splits_do_not_overlap():
    benign = [f"good-{index}" for index in range(20)]
    harmful = [f"bad-{index}" for index in range(10)]

    first = build_workload_samples(
        benign,
        harmful,
        hidden_per_class=4,
        score_count=3,
        decode_count=5,
        seed=17,
    )
    second = build_workload_samples(
        benign,
        harmful,
        hidden_per_class=4,
        score_count=3,
        decode_count=5,
        seed=17,
    )

    assert first == second
    assert len(first["hidden_benign"]) == 4
    assert len(first["hidden_harmful"]) == 4
    assert len(first["score"]) == 3
    assert len(first["decode"]) == 5
    assert set(first["hidden_benign"]).isdisjoint(first["score"])
    assert set(first["hidden_benign"]).isdisjoint(first["decode"])
    assert set(first["score"]).isdisjoint(first["decode"])


def test_timing_summary_uses_median_throughput_and_linear_p95():
    summary = summarize_seconds([1.0, 2.0, 4.0], token_count=100)

    assert summary["median_seconds"] == 2.0
    assert summary["mean_seconds"] == pytest.approx(7 / 3)
    assert summary["p95_seconds"] == pytest.approx(3.8)
    assert summary["tokens_per_second"] == 50.0


def test_batch_padding_stats_quantifies_length_sorting_savings():
    lengths = [2, 10, 3, 9]

    random_stats = batch_padding_stats(lengths, batch_size=2, sort=False)
    sorted_stats = batch_padding_stats(lengths, batch_size=2, sort=True)

    assert random_stats == {
        "useful_prompt_tokens": 24,
        "padded_prompt_tokens": 38,
        "padding_tokens": 14,
        "padding_fraction": pytest.approx(14 / 38),
        "tensor_token_multiplier": pytest.approx(38 / 24),
    }
    assert sorted_stats["padded_prompt_tokens"] == 26
    assert sorted_stats["padding_tokens"] == 2
    assert sorted_stats["tensor_token_multiplier"] == pytest.approx(26 / 24)


def test_output_fingerprint_is_stable_shape_aware_and_checks_finiteness():
    output = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    first = fingerprint_output(output)
    second = fingerprint_output(output.clone())
    reshaped = fingerprint_output(output.reshape(1, 4))

    assert first == second
    assert first["finite"] is True
    assert first["shape"] == [2, 2]
    assert first["dtype"] == "torch.float32"
    assert len(first["sha256"]) == 64
    assert first["sha256"] != reshaped["sha256"]


def test_compressed_logprob_probe_supports_exact_kl_comparison():
    baseline = torch.log_softmax(
        torch.tensor([[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]]),
        dim=-1,
    )
    candidate = torch.log_softmax(
        torch.tensor([[2.5, 1.5, 0.0], [0.0, 1.0, 2.0]]),
        dim=-1,
    )

    same = compare_logprob_probes(
        encode_logprob_probe(baseline),
        encode_logprob_probe(baseline.clone()),
    )
    changed = compare_logprob_probes(
        encode_logprob_probe(baseline),
        encode_logprob_probe(candidate),
    )

    assert same["kl_base_to_candidate"] == pytest.approx(0.0, abs=1e-8)
    assert same["top1_agreement"] == 1.0
    assert changed["kl_base_to_candidate"] > 0
    assert changed["top1_agreement"] == 0.5


def test_logprob_probe_comparison_accepts_masked_negative_infinity():
    masked = torch.tensor([[-0.25, -2.0, float("-inf")], [-1.0, float("-inf"), -0.5]])

    comparison = compare_logprob_probes(
        encode_logprob_probe(masked),
        encode_logprob_probe(masked.clone()),
    )

    assert comparison["finite"] is True
    assert comparison["raw_valid"] is True
    assert comparison["masked_neginf_accepted"] is True
    assert comparison["raw_nan_count"] == {"baseline": 0, "candidate": 0}
    assert comparison["raw_posinf_count"] == {"baseline": 0, "candidate": 0}
    assert comparison["raw_neginf_count"] == {"baseline": 2, "candidate": 2}
    assert comparison["kl_base_to_candidate"] == pytest.approx(0.0, abs=1e-8)
    assert all(
        math.isfinite(comparison[key])
        for key in (
            "kl_base_to_candidate",
            "top1_agreement",
            "max_abs_logprob_delta",
        )
    )


def test_logprob_probe_comparison_rejects_raw_nan_and_positive_infinity():
    invalid = torch.tensor([[float("nan"), float("inf"), float("-inf"), -0.5]])
    candidate = torch.tensor([[-3.0, -2.0, float("-inf"), -0.5]])

    comparison = compare_logprob_probes(
        encode_logprob_probe(invalid),
        encode_logprob_probe(candidate),
    )

    assert comparison["finite"] is False
    assert comparison["raw_valid"] is False
    assert comparison["raw_nan_count"] == {"baseline": 1, "candidate": 0}
    assert comparison["raw_posinf_count"] == {"baseline": 1, "candidate": 0}
    assert comparison["raw_neginf_count"] == {"baseline": 1, "candidate": 1}
    assert all(
        math.isfinite(comparison[key])
        for key in (
            "kl_base_to_candidate",
            "top1_agreement",
            "max_abs_logprob_delta",
        )
    )
    json.dumps(comparison, allow_nan=False)


def test_report_comparison_calculates_speed_vram_and_correctness():
    probe = encode_logprob_probe(
        torch.log_softmax(torch.tensor([[3.0, 1.0, 0.0]]), dim=-1)
    )
    base = {
        "label": "base",
        "records": [
            {
                "workload": "hidden_extraction",
                "mode": "random",
                "batch_size": 4,
                "cold": {"seconds": 2.0},
                "warm": {
                    "median_seconds": 1.0,
                    "peak_allocated_mib_max": 100.0,
                },
                "correctness": {
                    "finite": True,
                    "warm_output_sha256": "same",
                },
            }
        ],
        "score_logprob_probe": probe,
    }
    candidate = {
        "label": "fla",
        "records": [
            {
                "workload": "hidden_extraction",
                "mode": "random",
                "batch_size": 4,
                "cold": {"seconds": 1.25},
                "warm": {
                    "median_seconds": 0.5,
                    "peak_allocated_mib_max": 120.0,
                },
                "correctness": {
                    "finite": True,
                    "warm_output_sha256": "same",
                },
            }
        ],
        "score_logprob_probe": probe,
    }

    comparison = compare_reports(base, candidate)

    row = comparison["records"][0]
    assert row["warm_speedup"] == 2.0
    assert row["cold_speedup"] == 1.6
    assert row["peak_allocated_mib_delta"] == 20.0
    assert row["output_hash_equal"] is True
    assert comparison["correctness"]["all_finite"] is True
    assert comparison["correctness"]["score_probe"][
        "kl_base_to_candidate"
    ] == pytest.approx(0.0, abs=1e-8)


def test_report_comparison_accepts_masked_neginf_in_scoring_records():
    probe = encode_logprob_probe(torch.tensor([[-0.25, -2.0, float("-inf")]]))
    record = {
        "workload": "shared_continuation_scoring",
        "mode": "random",
        "batch_size": 4,
        "cold": {"seconds": 2.0},
        "warm": {
            "median_seconds": 1.0,
            "peak_allocated_mib_max": 100.0,
        },
        "correctness": {
            "finite": False,
            "warm_output_sha256": "same",
        },
    }

    comparison = compare_reports(
        {"label": "base", "records": [record], "score_logprob_probe": probe},
        {"label": "fla", "records": [record], "score_logprob_probe": probe},
    )

    assert comparison["records"][0]["finite"] is True
    assert comparison["correctness"]["all_finite"] is True
    assert comparison["correctness"]["score_probe"]["finite"] is True
