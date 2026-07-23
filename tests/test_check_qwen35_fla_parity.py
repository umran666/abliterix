"""Contracts for the privacy-preserving Qwen3.5 base/FLA parity check."""

import hashlib

import pytest

from scripts.check_qwen35_fla_parity import (
    compare_reports,
    decode_generated_sequences,
    hash_generated_records,
    kernel_provenance,
    model_contract,
    parse_args,
    select_prompt_rows,
)


def test_hash_seed_sampling_is_stable_and_independent_of_input_order():
    rows = [{"prompt": f"prompt-{index}"} for index in range(12)]

    first = select_prompt_rows(rows, count=5, seed=17, domain="harmful")
    second = select_prompt_rows(
        list(reversed(rows)), count=5, seed=17, domain="harmful"
    )

    assert first == second
    assert len(first) == 5
    assert len({row["prompt_sha256"] for row in first}) == 5
    assert all(set(row) == {"domain", "prompt_sha256", "prompt"} for row in first)


def test_generated_records_contain_only_hashes_counts_and_detector_flags():
    class Detector:
        @staticmethod
        def detect_refusal(response: str) -> bool:
            return "cannot" in response.casefold()

        @staticmethod
        def _is_degenerate(response: str) -> bool:
            return response == "broken output"

    selected = [
        {
            "domain": "harmful",
            "prompt_sha256": hashlib.sha256(b"secret prompt").hexdigest(),
            "prompt": "secret prompt",
        },
        {
            "domain": "benign",
            "prompt_sha256": hashlib.sha256(b"private prompt").hexdigest(),
            "prompt": "private prompt",
        },
    ]

    records = hash_generated_records(
        selected,
        ["I cannot assist with that.", "broken output"],
        [6, 2],
        detector=Detector(),
    )

    assert records == [
        {
            "domain": "harmful",
            "prompt_sha256": hashlib.sha256(b"secret prompt").hexdigest(),
            "response_sha256": hashlib.sha256(
                b"I cannot assist with that."
            ).hexdigest(),
            "token_count": 6,
            "heuristic_refusal": True,
            "explicit_refusal": True,
            "degenerate": False,
        },
        {
            "domain": "benign",
            "prompt_sha256": hashlib.sha256(b"private prompt").hexdigest(),
            "response_sha256": hashlib.sha256(b"broken output").hexdigest(),
            "token_count": 2,
            "heuristic_refusal": False,
            "explicit_refusal": False,
            "degenerate": True,
        },
    ]
    serialized = repr(records)
    assert "secret prompt" not in serialized
    assert "private prompt" not in serialized
    assert "broken output" not in serialized


def _record(
    domain: str,
    prompt: str,
    response: str,
    tokens: int,
    *,
    refusal: bool = False,
    explicit: bool = False,
    degenerate: bool = False,
) -> dict:
    return {
        "domain": domain,
        "prompt_sha256": prompt,
        "response_sha256": response,
        "token_count": tokens,
        "heuristic_refusal": refusal,
        "explicit_refusal": explicit,
        "degenerate": degenerate,
    }


def _report(label: str, records: list[dict], *, tok_s: float, peak: float) -> dict:
    return {
        "schema_version": "abliterix-qwen35-fla-parity-v1",
        "status": "complete",
        "label": label,
        "contract": {
            "model": {"requested": "Qwen/Qwen3.5-2B", "revision": "abc"},
            "generation": {"seed": 17, "batch_size": 32},
            "datasets": {"harmful_sha256": "h", "benign_sha256": "b"},
            "ordered_sample_keys": [
                [row["domain"], row["prompt_sha256"]] for row in records
            ],
        },
        "records": records,
        "performance": {
            "tokens_per_second": tok_s,
            "peak_allocated_gib": peak,
            "peak_reserved_gib": peak + 1.0,
        },
    }


def test_compare_reports_measures_exact_parity_paired_behavior_and_speed():
    base_records = [
        _record("harmful", "p1", "a", 2, refusal=True),
        _record("harmful", "p2", "b", 3),
        _record("benign", "p3", "c", 4, explicit=True),
        _record("benign", "p4", "d", 5, degenerate=True),
    ]
    fla_records = [
        _record("harmful", "p1", "a", 2),
        _record("harmful", "p2", "e", 4),
        _record("benign", "p3", "c", 4),
        _record("benign", "p4", "f", 5),
    ]

    comparison = compare_reports(
        _report("base", base_records, tok_s=100.0, peak=20.0),
        _report("fla", fla_records, tok_s=125.0, peak=18.0),
        bootstrap_resamples=200,
        bootstrap_seed=23,
    )

    assert comparison["parity"]["response_exact_match"] == {
        "count": 2,
        "n": 4,
        "rate": 0.5,
    }
    assert comparison["parity"]["token_count_match"]["rate"] == 0.75
    assert comparison["parity"]["token_count_mean_delta"] == 0.25
    assert comparison["behavior"]["harmful_heuristic_refusal"]["delta"] == -0.5
    assert comparison["behavior"]["benign_explicit_refusal"]["delta"] == -0.5
    assert comparison["behavior"]["all_degenerate"]["delta"] == -0.25
    assert comparison["performance"]["tokens_per_second"]["ratio"] == 1.25
    assert comparison["performance"]["peak_allocated_gib"]["ratio"] == 0.9


def test_compare_reports_rejects_a_changed_model_dataset_or_sample_contract():
    rows = [_record("harmful", "p1", "a", 2)]
    base = _report("base", rows, tok_s=100.0, peak=20.0)
    fla = _report("fla", rows, tok_s=120.0, peak=19.0)
    fla["contract"]["model"]["revision"] = "different"

    with pytest.raises(ValueError, match="contracts do not match"):
        compare_reports(base, fla)


def test_decode_generated_sequences_excludes_prompt_eos_and_batch_padding():
    class Tokenizer:
        @staticmethod
        def batch_decode(rows, *, skip_special_tokens: bool):
            assert skip_special_tokens is True
            return ["/".join(str(token) for token in row) for row in rows]

    responses, counts = decode_generated_sequences(
        [
            [0, 10, 11, 20, 21, 99, 0],
            [10, 11, 12, 30, 99, 0, 0],
        ],
        prompt_width=3,
        eos_token_ids={99},
        pad_token_id=0,
        tokenizer=Tokenizer(),
    )

    assert responses == ["20/21", "30"]
    assert counts == [2, 1]


def test_model_contract_records_snapshot_revision_and_small_file_hashes(tmp_path):
    snapshot = tmp_path / "models--Qwen--Qwen3.5-2B" / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type":"qwen3_5"}')
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"weights")

    contract = model_contract(str(snapshot))

    assert contract["revision"] == "deadbeef"
    assert contract["resolved_path"] == str(snapshot.resolve())
    assert len(contract["config.json_sha256"]) == 64
    assert contract["weight_files"] == [
        {"name": "model-00001-of-00001.safetensors", "size_bytes": 7}
    ]


def test_kernel_provenance_proves_bound_fla_callable_not_just_importability():
    def fla_chunk():
        return None

    fla_chunk.__module__ = "fla.ops.gated_delta_rule"
    fla_chunk.__qualname__ = "fla_chunk"

    class FlaNorm:
        pass

    FlaNorm.__module__ = "fla.modules"

    class FakeGatedDeltaNet:
        chunk_gated_delta_rule = staticmethod(fla_chunk)
        recurrent_gated_delta_rule = staticmethod(fla_chunk)
        causal_conv1d_fn = None
        causal_conv1d_update = staticmethod(fla_chunk)
        norm = FlaNorm()

    class Config:
        model_type = "qwen3_5"
        name_or_path = "snapshot"
        _commit_hash = "deadbeef"
        _attn_implementation = "sdpa"

    class Model:
        config = Config()

        @staticmethod
        def named_modules():
            return [("model.layers.0.gated_delta_net", FakeGatedDeltaNet())]

    provenance = kernel_provenance(Model())

    assert provenance["gated_delta_net_layer_count"] == 1
    assert provenance["fla_kernel_binding_present"] is True
    assert provenance["bindings"]["chunk_gated_delta_rule"] == {
        "fla.ops.gated_delta_rule.fla_chunk": 1
    }


def test_cli_exposes_one_process_run_and_compare_modes(tmp_path):
    run = parse_args(
        ["--model", "model-path", "--label", "base", "--output", str(tmp_path / "b")]
    )
    assert run.model == "model-path"
    assert run.label == "base"
    assert run.compare is None
    assert run.batch_size == 32
    assert run.max_new_tokens == 128
    assert run.min_new_tokens == 3

    compare = parse_args(
        [
            "--compare",
            "base.json",
            "fla.json",
            "--output",
            str(tmp_path / "comparison.json"),
        ]
    )
    assert [path.name for path in compare.compare] == ["base.json", "fla.json"]
