"""Pure-function contracts for the GSM8K A/B capability guardrail."""

import hashlib
import json

import pytest

from scripts.evaluate_ab_gsm8k import (
    FINAL_ANSWER_INSTRUCTION,
    build_gsm8k_messages,
    canonical_manifest_sha256,
    resolve_ab_contract,
    sample_problems,
    score_responses,
    summarize_accuracy,
    verify_manifest,
)


def _complete_ab_result() -> dict:
    recipes = {
        "U": {"name": "unsteered", "steered": False, "steering": {}},
        "M": {
            "name": "mean_orthogonal_rank1",
            "steered": True,
            "steering": {"vector_method": "mean", "n_directions": 1},
        },
        "Q": {
            "name": "projected_rank1",
            "steered": True,
            "steering": {
                "vector_method": "mean",
                "projected_abliteration": True,
                "n_directions": 1,
            },
        },
        "R": {
            "name": "projected_rank3",
            "steered": True,
            "steering": {"vector_method": "mean", "n_directions": 3},
        },
    }
    return {
        "schema_version": "abliterix-qwen35-ab-v1",
        "status": "complete",
        "model": {
            "source_id": "Qwen/Qwen3.5-2B",
            "revision": "a" * 40,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
        },
        "experiment_contract": {
            "seed": 20260710,
            "train_size_per_domain": 399,
            "dev_size_per_domain": 100,
            "locked_test_size_per_domain": 499,
            "batch_size": 32,
            "sort_by_length": True,
        },
        "recipes": recipes,
        "split_manifest_sha256": "b" * 64,
        "dev": {
            "selection": {
                "selected_by_arm": {
                    "M": {"arm": "M", "strength": 1.2},
                    "Q": {"arm": "Q", "strength": 0.8},
                },
                "challenger": {"arm": "Q", "strength": 0.8},
                "locked_test_arms": ["U", "M", "Q"],
            }
        },
        "test": {"winner": "Q", "arms": {}},
    }


def test_resolve_ab_contract_restores_model_split_and_selected_recipes():
    contract = resolve_ab_contract(_complete_ab_result())

    assert contract["model"] == {
        "source_id": "Qwen/Qwen3.5-2B",
        "revision": "a" * 40,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
    }
    assert contract["seed"] == 20260710
    assert contract["reconstruction_batch_size"] == 32
    assert contract["sort_by_length"] is True
    assert contract["split_sizes"] == {"train": 399, "dev": 100, "test": 499}
    assert contract["winner"] == "Q"
    assert contract["root_n_directions"] == 3
    assert contract["arms"]["U"]["strength"] is None
    assert contract["arms"]["M"]["strength"] == 1.2
    assert contract["arms"]["Q"]["strength"] == 0.8
    assert contract["arms"]["Q"]["recipe"]["name"] == "projected_rank1"


def test_resolve_ab_contract_rejects_failed_or_inconsistently_locked_results():
    failed = _complete_ab_result()
    failed["status"] = "failed"
    with pytest.raises(ValueError, match="status must be 'complete'"):
        resolve_ab_contract(failed)

    inconsistent = _complete_ab_result()
    inconsistent["test"]["winner"] = "R"
    with pytest.raises(ValueError, match="challenger"):
        resolve_ab_contract(inconsistent)


def test_manifest_sha_is_canonical_and_detects_tampering():
    manifest = {
        "schema_version": "abliterix-ab-splits-v1",
        "seed": 7,
        "sets": {"harmful": {}, "benign": {}},
    }
    digest = canonical_manifest_sha256(manifest)
    manifest["manifest_sha256"] = digest

    assert verify_manifest(manifest, expected_sha256=digest) == digest

    manifest["seed"] = 8
    with pytest.raises(ValueError, match="internal SHA256"):
        verify_manifest(manifest, expected_sha256=digest)


def test_manifest_sha_matches_documented_canonical_json_encoding():
    manifest = {"z": "你好", "a": [2, 1]}
    encoded = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert canonical_manifest_sha256(manifest) == hashlib.sha256(encoded).hexdigest()


def test_problem_sampling_is_deterministic_exact_and_non_mutating():
    problems = [
        {"question": f"q-{index}", "answer": f"#### {index}"} for index in range(30)
    ]

    first = sample_problems(problems, n_problems=8, seed=123)
    second = sample_problems(problems, n_problems=8, seed=123)

    assert first == second
    assert len(first) == 8
    assert len({row["source_index"] for row in first}) == 8
    assert all("source_index" not in problem for problem in problems)
    assert sample_problems(problems, n_problems=8, seed=124) != first


def test_messages_share_one_final_numeric_answer_instruction():
    messages = build_gsm8k_messages(
        [{"question": "What is 20 + 22?", "answer": "#### 42"}],
        message_type=lambda system, user: {"system": system, "user": user},
    )

    assert messages == [
        {
            "system": FINAL_ANSWER_INSTRUCTION,
            "user": "What is 20 + 22?",
        }
    ]
    assert "only the final numeric answer" in FINAL_ANSWER_INSTRUCTION.casefold()


def test_response_scoring_reuses_external_eval_numeric_normalisation():
    problems = [
        {"source_index": 3, "question": "q1", "answer": "work\n#### 1,234"},
        {"source_index": 9, "question": "q2", "answer": "#### -2.5"},
        {"source_index": 11, "question": "q3", "answer": "#### 7"},
    ]

    records = score_responses(problems, ["1,234", "answer: -2.5.", "no number"])

    assert [record["correct"] for record in records] == [True, True, False]
    assert records[0]["gold_normalized"] == "1234"
    assert records[1]["predicted_normalized"] == "-2.5"
    assert records[2]["predicted_normalized"] is None


def test_accuracy_summary_has_wilson_interval_and_deterministic_pairable_flags():
    records = [{"correct": value} for value in (True, False, True, True)]

    summary = summarize_accuracy(records)

    assert summary["n"] == 4
    assert summary["n_correct"] == 3
    assert summary["accuracy"]["point"] == 0.75
    assert summary["accuracy"]["ci95"][0] < 0.75
    assert summary["accuracy"]["ci95"][1] > 0.75
    assert summary["correct"] == [True, False, True, True]
