"""Pure-function contracts for the Qwen3.5 A/B harness."""

import hashlib

import pytest
import torch

from abliterix.eval.scorer import _safe_kl_divergence
from scripts.ab_test_qwen35 import (
    build_split_manifest,
    build_prompt_splits,
    classify_generated_responses,
    deduplicate_entries,
    detect_explicit_refusal,
    file_sha256,
    paired_bootstrap_delta_ci,
    paired_cluster_bootstrap_delta_ci,
    per_prompt_kl,
    prompt_hash,
    recipe_specs,
    select_dev_point,
    wilson_rate_interval,
)


def _entry(idx: int, *, language: str = "en", category: str = "cat") -> dict:
    return {
        "id": idx,
        "prompt": f"prompt-{idx}",
        "language": language,
        "category": category,
    }


def test_build_prompt_splits_excludes_hash_pinned_test_and_is_deterministic():
    all_entries = [
        _entry(i, language=("en", "zh")[i % 2], category=f"cat-{i % 3}")
        for i in range(10)
    ]
    test_entries = all_entries[:4]

    first = build_prompt_splits(
        all_entries,
        test_entries,
        train_size=4,
        dev_size=2,
        test_size=None,
        seed=17,
    )
    second = build_prompt_splits(
        all_entries,
        test_entries,
        train_size=4,
        dev_size=2,
        test_size=None,
        seed=17,
    )

    assert first == second
    assert len(first["train"]) == 4
    assert len(first["dev"]) == 2
    assert first["test"] == test_entries

    hashes = {
        name: {prompt_hash(row["prompt"]) for row in rows}
        for name, rows in first.items()
    }
    assert hashes["train"].isdisjoint(hashes["dev"])
    assert hashes["train"].isdisjoint(hashes["test"])
    assert hashes["dev"].isdisjoint(hashes["test"])


def test_select_dev_point_minimizes_kl_within_preregistered_guardrails():
    points = [
        {
            "strength": 0.8,
            "refusal_rate": 0.08,
            "kl": 0.020,
            "length_z": 0.5,
            "degenerate_rate": 0.01,
        },
        {
            "strength": 1.0,
            "refusal_rate": 0.10,
            "kl": 0.010,
            "length_z": 0.4,
            "degenerate_rate": 0.02,
        },
        {
            "strength": 1.2,
            "refusal_rate": 0.02,
            "kl": 0.005,
            "length_z": 2.5,
            "degenerate_rate": 0.01,
        },
    ]

    selected = select_dev_point(
        points,
        baseline_degenerate_rate=0.01,
        target_refusal_rate=0.10,
        max_degenerate_delta=0.02,
        max_length_z=2.0,
    )

    assert selected["strength"] == 1.0
    assert selected["selection_reason"] == "lowest_kl_within_guardrails"


def test_select_dev_point_does_not_trade_unbounded_kl_for_lower_refusal():
    points = [
        {
            "strength": 0.5,
            "refusal_rate": 0.60,
            "kl": 0.011,
            "length_z": 0.0,
            "degenerate_rate": 0.0,
        },
        {
            "strength": 1.5,
            "refusal_rate": 0.05,
            "kl": 0.050,
            "length_z": 0.0,
            "degenerate_rate": 0.0,
        },
    ]

    selected = select_dev_point(
        points,
        baseline_degenerate_rate=0.0,
        max_kl=0.01,
    )

    assert selected["strength"] == 0.5
    assert selected["selection_reason"] == "fallback_smallest_kl_guardrail_violation"


def test_paired_bootstrap_resamples_prompt_pairs_deterministically():
    reference = [0.0, 2.0, 4.0, 8.0]
    current = [value + 1.5 for value in reference]

    first = paired_bootstrap_delta_ci(
        current,
        reference,
        seed=123,
        n_resamples=500,
    )
    second = paired_bootstrap_delta_ci(
        current,
        reference,
        seed=123,
        n_resamples=500,
    )

    assert first == second
    assert first["point"] == 1.5
    assert first["ci95"] == [1.5, 1.5]


def test_wilson_interval_keeps_boundary_rate_uncertainty_visible():
    result = wilson_rate_interval(0, 500)

    assert result["point"] == 0.0
    assert result["ci95"][0] == 0.0
    assert 0.007 < result["ci95"][1] < 0.008


def test_per_prompt_kl_averages_shared_token_contexts_within_prompt():
    baseline = torch.log_softmax(
        torch.tensor(
            [
                [[3.0, 1.0, 0.0], [1.0, 2.0, 0.0]],
                [[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]],
            ]
        ),
        dim=-1,
    )
    current = torch.log_softmax(
        torch.tensor(
            [
                [[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
                [[1.0, 0.0, 2.0], [1.0, 2.0, 0.0]],
            ]
        ),
        dim=-1,
    )

    values = per_prompt_kl(current, baseline)

    assert values.shape == (2,)
    assert torch.all(values >= 0)
    assert values.mean().item() == pytest.approx(_safe_kl_divergence(current, baseline))


def test_manifest_records_dataset_and_prompt_hashes_without_raw_prompts(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text('[{"prompt":"alpha"}]', encoding="utf-8")
    splits = {
        "train": [_entry(1)],
        "dev": [_entry(2)],
        "test": [_entry(3)],
    }

    manifest = build_split_manifest(
        harmful_splits=splits,
        benign_splits=splits,
        source_sha256={"harmful_1000": file_sha256(dataset)},
        deduplication={"benign_1000": {"dropped_count": 2, "dropped": [7, 9]}},
        seed=17,
    )

    assert manifest["seed"] == 17
    assert (
        manifest["source_sha256"]["harmful_1000"]
        == hashlib.sha256(dataset.read_bytes()).hexdigest()
    )
    assert manifest["sets"]["harmful"]["train"]["count"] == 1
    assert manifest["sets"]["harmful"]["train"]["prompt_sha256"] == [
        prompt_hash("prompt-1")
    ]
    assert manifest["deduplication"]["benign_1000"]["dropped_count"] == 2
    assert "prompt-1" not in str(manifest)
    assert len(manifest["manifest_sha256"]) == 64


def test_generated_responses_are_classified_without_triggering_generation():
    class FakeDetector:
        def __init__(self):
            self.detected = []
            self.degeneracy_checked = []

        def detect_refusal(self, response):
            self.detected.append(response)
            return "refuse" in response

        def _is_degenerate(self, response):
            self.degeneracy_checked.append(response)
            return response == "loop"

    detector = FakeDetector()
    result = classify_generated_responses(detector, ["answer", "refuse", "loop"])

    assert result == {
        "refusal": [False, True, True],
        "degenerate": [False, False, True],
        "explicit_refusal": [False, False, False],
    }
    assert detector.detected == ["answer", "refuse", "loop"]
    assert detector.degeneracy_checked == ["answer", "refuse", "loop"]


def test_duplicate_prompts_are_dropped_by_hash_without_cross_split_leakage():
    all_entries = [_entry(i) for i in range(8)]
    all_entries.append({**_entry(0), "id": 80})
    test_entries = [all_entries[0], all_entries[1], {**_entry(1), "id": 81}]

    unique_all, report = deduplicate_entries(all_entries, label="all")
    splits = build_prompt_splits(
        all_entries,
        test_entries,
        train_size=4,
        dev_size=2,
        test_size=1,
        seed=9,
    )

    assert len(unique_all) == 8
    assert report["dropped_count"] == 1
    assert report["dropped"][0]["id"] == 80
    assert report["dropped"][0]["duplicate_of_id"] == 0
    assert len(splits["test"]) == 1

    hashes = {
        name: {prompt_hash(row["prompt"]) for row in rows}
        for name, rows in splits.items()
    }
    assert hashes["train"].isdisjoint(hashes["dev"])
    assert hashes["train"].isdisjoint(hashes["test"])
    assert hashes["dev"].isdisjoint(hashes["test"])
    all_held_out_hashes = {prompt_hash(row["prompt"]) for row in test_entries}
    assert hashes["train"].isdisjoint(all_held_out_hashes)
    assert hashes["dev"].isdisjoint(all_held_out_hashes)


def test_rank3_recipe_is_an_explicit_optional_challenger():
    without_rank3 = recipe_specs(include_rank3=False)
    with_rank3 = recipe_specs(include_rank3=True)

    assert list(without_rank3) == ["U", "M", "Q"]
    assert list(with_rank3) == ["U", "M", "Q", "R"]
    assert with_rank3["Q"]["steering"]["vector_method"] == "mean"
    assert with_rank3["Q"]["steering"]["projected_abliteration"] is True
    assert with_rank3["R"]["steering"]["vector_method"] == "mean"
    assert with_rank3["R"]["steering"]["projected_abliteration"] is True
    assert with_rank3["R"]["steering"]["winsorize_vectors"] is True
    assert with_rank3["R"]["steering"]["discriminative_layer_selection"] is True
    assert with_rank3["R"]["steering"]["n_directions"] == 3


def test_benign_refusal_marker_ignores_policy_vocabulary_in_real_answers():
    assert (
        detect_explicit_refusal("Illegal activity can be prevented with audits.")
        is False
    )
    assert detect_explicit_refusal("抱歉，我无法协助这个请求。") is True
    assert detect_explicit_refusal("As an AI, I cannot perform that action.") is True


def test_cluster_bootstrap_keeps_topic_rows_together_and_is_deterministic():
    current = [1.0, 1.0, 0.0, 0.0]
    reference = [0.0, 0.0, 0.0, 0.0]
    clusters = ["topic-a", "topic-a", "topic-b", "topic-b"]

    first = paired_cluster_bootstrap_delta_ci(
        current,
        reference,
        clusters,
        seed=7,
        n_resamples=500,
    )
    second = paired_cluster_bootstrap_delta_ci(
        current,
        reference,
        clusters,
        seed=7,
        n_resamples=500,
    )

    assert first == second
    assert first["point"] == 0.5
    assert first["n_clusters"] == 2
    assert first["ci95"] == [0.0, 1.0]
