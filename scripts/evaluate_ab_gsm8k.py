#!/usr/bin/env python3
"""Rebuild a completed A/B recipe and measure its GSM8K capability tax.

The statistical and provenance helpers intentionally remain importable on a
CPU-only machine.  Model, CUDA, and ``datasets`` imports are deferred to
``main()`` so their contracts can be tested without downloading artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, cast

REPO_ROOT = Path(__file__).resolve().parent.parent
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from abliterix.external_eval import _answers_match, _normalise_answer
from scripts.ab_test_qwen35 import (
    build_prompt_splits,
    build_split_manifest,
    deduplicate_entries,
    file_sha256,
    paired_bootstrap_delta_ci,
    wilson_rate_interval,
)


Entry = dict[str, Any]

DEFAULT_GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
FINAL_ANSWER_INSTRUCTION = (
    "Solve the problem. Return only the final numeric answer, with no reasoning, "
    "explanation, units, or surrounding text."
)


def canonical_manifest_sha256(manifest: dict[str, Any]) -> str:
    """Hash a split manifest using the A/B harness's canonical JSON encoding."""
    payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def verify_manifest(
    manifest: dict[str, Any],
    *,
    expected_sha256: str,
) -> str:
    """Validate both the manifest's self-hash and the A/B result reference."""
    internal = manifest.get("manifest_sha256")
    if not isinstance(internal, str) or len(internal) != 64:
        raise ValueError("Split manifest is missing a valid manifest_sha256")
    actual = canonical_manifest_sha256(manifest)
    if actual != internal:
        raise ValueError(
            "Split manifest internal SHA256 does not match its canonical content: "
            f"recorded={internal}, actual={actual}"
        )
    if expected_sha256 != internal:
        raise ValueError(
            "A/B result split manifest SHA256 does not match the manifest: "
            f"result={expected_sha256}, manifest={internal}"
        )
    return actual


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _strength(value: Any, *, arm: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Selected strength for {arm} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"Selected strength for {arm} must be finite and positive")
    return resolved


def resolve_ab_contract(report: dict[str, Any]) -> dict[str, Any]:
    """Extract the exact model, split, adapter rank, and locked arm recipes."""
    if report.get("status") != "complete":
        raise ValueError("A/B result status must be 'complete'")

    model = report.get("model")
    experiment = report.get("experiment_contract")
    recipes = report.get("recipes")
    dev = report.get("dev")
    test = report.get("test")
    if not isinstance(model, dict):
        raise ValueError("A/B result is missing model data")
    if not isinstance(experiment, dict):
        raise ValueError("A/B result is missing experiment data")
    if not isinstance(recipes, dict):
        raise ValueError("A/B result is missing recipe data")
    if not isinstance(dev, dict):
        raise ValueError("A/B result is missing development data")
    if not isinstance(test, dict):
        raise ValueError("A/B result is missing test data")

    source_id = model.get("source_id")
    revision = model.get("revision")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("A/B model.source_id must be a non-empty string")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("A/B model.revision must be a non-empty string")

    selection = dev.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("A/B result is missing the locked development selection")
    selected_by_arm = selection.get("selected_by_arm")
    challenger = selection.get("challenger")
    if not isinstance(selected_by_arm, dict) or not isinstance(challenger, dict):
        raise ValueError("A/B development selection is incomplete")

    winner = test.get("winner")
    if not isinstance(winner, str) or winner in {"U", "M"}:
        raise ValueError("A/B test winner must identify the locked challenger arm")
    if challenger.get("arm") != winner:
        raise ValueError(
            "A/B test winner does not match the development challenger: "
            f"winner={winner!r}, challenger={challenger.get('arm')!r}"
        )
    locked = selection.get("locked_test_arms")
    if not isinstance(locked, list) or set(locked) != {"U", "M", winner}:
        raise ValueError("A/B locked test arms do not match U/M/challenger")

    for arm in ("U", "M", winner):
        if not isinstance(recipes.get(arm), dict):
            raise ValueError(f"A/B result is missing recipe {arm}")
    if recipes["U"].get("steered") is not False:
        raise ValueError("A/B U recipe is not an unsteered control")
    if (
        recipes["M"].get("steered") is not True
        or recipes[winner].get("steered") is not True
    ):
        raise ValueError("A/B M and challenger recipes must be steered")

    selected_m = selected_by_arm.get("M")
    if not isinstance(selected_m, dict) or selected_m.get("arm") != "M":
        raise ValueError("A/B result is missing M's selected development point")
    selected_winner = selected_by_arm.get(winner)
    if isinstance(selected_winner, dict):
        selected_strength = _strength(selected_winner.get("strength"), arm=winner)
        challenger_strength = _strength(challenger.get("strength"), arm=winner)
        if selected_strength != challenger_strength:
            raise ValueError("A/B challenger strength conflicts with selected_by_arm")

    split_sha = report.get("split_manifest_sha256")
    if not isinstance(split_sha, str) or len(split_sha) != 64:
        raise ValueError("A/B result has no valid split_manifest_sha256")

    root_n_directions = max(
        int(recipe.get("steering", {}).get("n_directions", 1))
        for recipe in recipes.values()
        if isinstance(recipe, dict)
    )
    if root_n_directions < 1:
        raise ValueError("A/B root adapter rank must be positive")

    return {
        "model": {
            "source_id": source_id,
            "revision": revision,
            "dtype": str(model.get("dtype", "bfloat16")),
            "attention_implementation": str(
                model.get("attention_implementation", "sdpa")
            ),
        },
        "seed": _positive_int(experiment.get("seed"), field="experiment seed"),
        "reconstruction_batch_size": _positive_int(
            experiment.get("batch_size"), field="A/B batch size"
        ),
        "sort_by_length": bool(experiment.get("sort_by_length", True)),
        "split_sizes": {
            "train": _positive_int(
                experiment.get("train_size_per_domain"), field="train size"
            ),
            "dev": _positive_int(
                experiment.get("dev_size_per_domain"), field="dev size"
            ),
            "test": _positive_int(
                experiment.get("locked_test_size_per_domain"), field="test size"
            ),
        },
        "split_manifest_sha256": split_sha,
        "winner": winner,
        "root_n_directions": root_n_directions,
        "arms": {
            "U": {"strength": None, "recipe": recipes["U"]},
            "M": {
                "strength": _strength(selected_m.get("strength"), arm="M"),
                "recipe": recipes["M"],
            },
            winner: {
                "strength": _strength(challenger.get("strength"), arm=winner),
                "recipe": recipes[winner],
            },
        },
    }


def sample_problems(
    problems: Iterable[dict[str, Any]],
    *,
    n_problems: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return an exact, deterministic sample without mutating source rows."""
    rows = list(problems)
    if n_problems < 1:
        raise ValueError("n_problems must be positive")
    if n_problems > len(rows):
        raise ValueError(
            f"Requested {n_problems} GSM8K problems but source only has {len(rows)}"
        )
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"GSM8K row {index} is not an object")
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"GSM8K row {index} has no question string")
        if not isinstance(row.get("answer"), str) or not row["answer"].strip():
            raise ValueError(f"GSM8K row {index} has no answer string")

    indices = random.Random(seed).sample(range(len(rows)), n_problems)
    return [{**rows[index], "source_index": index} for index in indices]


def build_gsm8k_messages(
    problems: Iterable[dict[str, Any]],
    *,
    message_type: Callable[..., Any],
) -> list[Any]:
    """Build identical final-answer-only messages for every experiment arm."""
    return [
        message_type(
            system=FINAL_ANSWER_INSTRUCTION,
            user=str(problem["question"]),
        )
        for problem in problems
    ]


def score_responses(
    problems: Iterable[dict[str, Any]],
    responses: Iterable[str],
) -> list[dict[str, Any]]:
    """Score responses with ``abliterix.external_eval`` numeric semantics."""
    problem_rows = list(problems)
    response_rows = list(responses)
    if len(problem_rows) != len(response_rows):
        raise ValueError("Problem and response counts must match")

    records: list[dict[str, Any]] = []
    for problem, response in zip(problem_rows, response_rows, strict=True):
        gold = _normalise_answer(str(problem["answer"]))
        if gold is None:
            raise ValueError(
                f"Unparseable GSM8K gold answer at source index "
                f"{problem.get('source_index')}"
            )
        predicted = _normalise_answer(str(response))
        question = str(problem["question"])
        records.append(
            {
                "source_index": problem.get("source_index"),
                "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "gold_normalized": gold,
                "predicted_normalized": predicted,
                "response": str(response),
                "correct": _answers_match(predicted, gold),
            }
        )
    return records


def summarize_accuracy(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize binary correctness with a Wilson 95% confidence interval."""
    rows = list(records)
    if not rows:
        raise ValueError("At least one scored response is required")
    correct = [bool(row["correct"]) for row in rows]
    return {
        "n": len(correct),
        "n_correct": sum(correct),
        "accuracy": wilson_rate_interval(sum(correct), len(correct)),
        "correct": correct,
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _resolve_manifest_path(ab_result_path: Path, report: dict[str, Any]) -> Path:
    recorded = report.get("split_manifest_path")
    candidates: list[Path] = [ab_result_path.with_suffix(".splits.json")]
    if isinstance(recorded, str) and recorded:
        recorded_path = Path(recorded)
        candidates.insert(0, recorded_path)
        candidates.append(ab_result_path.parent / recorded_path.name)
    checked: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in checked:
            continue
        checked.add(resolved)
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "Could not find the A/B split manifest; checked: "
        + ", ".join(str(path) for path in checked)
    )


def _load_entry_list(path: Path) -> list[Entry]:
    payload = _load_json(path)
    if not isinstance(payload, list) or not all(
        isinstance(row, dict) for row in payload
    ):
        raise ValueError(f"Expected a JSON list of objects in {path}")
    return payload


def _rebuild_and_verify_splits(
    contract: dict[str, Any],
    manifest: dict[str, Any],
    *,
    paths: dict[str, Path],
) -> tuple[dict[str, list[Entry]], dict[str, list[Entry]], dict[str, str]]:
    source_sha256 = {name: file_sha256(path) for name, path in paths.items()}
    if source_sha256 != manifest.get("source_sha256"):
        raise ValueError(
            "Local A/B dataset SHA256 values do not match the split manifest"
        )

    rows = {name: _load_entry_list(path) for name, path in paths.items()}
    sizes = contract["split_sizes"]
    harmful = build_prompt_splits(
        rows["harmful_1000"],
        rows["harmful_500"],
        train_size=sizes["train"],
        dev_size=sizes["dev"],
        test_size=sizes["test"],
        seed=contract["seed"],
    )
    benign = build_prompt_splits(
        rows["benign_1000"],
        rows["benign_500"],
        train_size=sizes["train"],
        dev_size=sizes["dev"],
        test_size=sizes["test"],
        seed=contract["seed"],
    )
    deduplication = {
        name: deduplicate_entries(entries, label=name)[1]
        for name, entries in rows.items()
    }
    rebuilt = build_split_manifest(
        harmful_splits=harmful,
        benign_splits=benign,
        source_sha256=source_sha256,
        deduplication=deduplication,
        seed=contract["seed"],
    )
    if rebuilt["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError(
            "Rebuilt local split manifest does not match the A/B manifest: "
            f"rebuilt={rebuilt['manifest_sha256']}, "
            f"expected={manifest['manifest_sha256']}"
        )
    return harmful, benign, source_sha256


def _method_config(base_config: Any, recipe: dict[str, Any]) -> Any:
    from abliterix.settings import SteeringConfig

    config = base_config.model_copy(deep=True)
    payload = config.steering.model_dump(mode="python")
    payload.update(recipe.get("steering", {}))
    config.steering = SteeringConfig.model_validate(payload)
    return config


def _profiles(engine: Any, strength: float, profile_type: Callable[..., Any]) -> dict:
    n_layers = len(engine.transformer_layers)
    return {
        component: profile_type(
            max_weight=strength,
            max_weight_position=0.7 * n_layers,
            min_weight=0.0,
            min_weight_distance=0.5 * n_layers,
        )
        for component in engine.list_steerable_components()
    }


def _activate_arm(
    engine: Any,
    *,
    arm: str,
    contract: dict[str, Any],
    method_configs: dict[str, Any],
    vectors: dict[str, Any],
    benign_states: Any,
    harmful_states: Any,
    apply_steering: Callable[..., Any],
    profile_type: Callable[..., Any],
) -> None:
    engine.restore_baseline()
    if arm == "U":
        return
    spec = contract["arms"][arm]
    apply_steering(
        engine,
        vectors[arm],
        None,
        _profiles(engine, spec["strength"], profile_type),
        method_configs[arm],
        benign_states=benign_states,
        target_states=harmful_states,
    )


def _generate_arm(
    engine: Any,
    messages: list[Any],
    problems: list[dict[str, Any]],
    *,
    arm: str,
    strength: float | None,
    max_new_tokens: int,
    torch: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    responses = engine.generate_text_batched(
        messages,
        skip_special_tokens=True,
        max_new_tokens=max_new_tokens,
        min_new_tokens=None,
        sort_by_length=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    records = score_responses(problems, responses)
    encoded = engine.tokenizer(
        responses,
        add_special_tokens=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    token_counts = [len(tokens) for tokens in encoded]
    for record, token_count in zip(records, token_counts, strict=True):
        record.update(
            {
                "schema_version": "abliterix-gsm8k-output-v1",
                "arm": arm,
                "strength": strength,
                "output_tokens": token_count,
            }
        )
    total_tokens = sum(token_counts)
    summary = summarize_accuracy(records)
    summary.update(
        {
            "arm": arm,
            "strength": strength,
            "generation_seconds": elapsed,
            "output_tokens": total_tokens,
            "tokens_per_second": total_tokens / elapsed if elapsed else 0.0,
            "sequences_per_second": len(problems) / elapsed if elapsed else 0.0,
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated() / (2**30)
                if torch.cuda.is_available()
                else 0.0
            ),
        }
    )
    return summary, records


def _load_gsm8k_problems(
    *,
    problems_json: Path | None,
    revision: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if problems_json is not None:
        path = _resolve_path(problems_json)
        payload = _load_json(path)
        if not isinstance(payload, list):
            raise ValueError("--problems-json must contain a JSON list")
        return payload, {
            "kind": "local_json",
            "path": str(path),
            "sha256": file_sha256(path),
        }

    from datasets import load_dataset

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="test",
        revision=revision,
    )
    return [dict(cast(dict[str, Any], row)) for row in dataset], {
        "kind": "huggingface_dataset",
        "dataset_id": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "revision": revision,
        "fingerprint": getattr(dataset, "_fingerprint", None),
    }


def _provenance(torch: Any) -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_gib": properties.total_memory / (2**30),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        }
    status = _git_output("status", "--porcelain")
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": _package_version("transformers"),
        "peft": _package_version("peft"),
        "datasets": _package_version("datasets"),
        "huggingface_hub": _package_version("huggingface-hub"),
        "gpu": gpu,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a completed Qwen3.5 A/B and run a GSM8K guardrail"
    )
    parser.add_argument("--ab-result", type=Path, required=True)
    parser.add_argument("--problems-json", type=Path)
    parser.add_argument("--gsm8k-revision", default=DEFAULT_GSM8K_REVISION)
    parser.add_argument("--n-problems", type=int, default=100)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override the A/B batch size (default: reuse the A/B contract)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-name")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument(
        "--harmful-all",
        type=Path,
        default=Path("datasets/harmful_1000/harmful_prompts_1000.json"),
    )
    parser.add_argument(
        "--benign-all",
        type=Path,
        default=Path("datasets/good_1000/good_prompts_1000.json"),
    )
    parser.add_argument(
        "--harmful-test",
        type=Path,
        default=Path("datasets/harmful_500/harmful_prompts_500.json"),
    )
    parser.add_argument(
        "--benign-test",
        type=Path,
        default=Path("datasets/good_500/good_prompts_500.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.n_problems = min(args.n_problems, 8)
        args.bootstrap_resamples = min(args.bootstrap_resamples, 500)
    if min(args.n_problems, args.max_new_tokens, args.bootstrap_resamples) < 1:
        raise ValueError("Problem, token, and bootstrap counts must be positive")

    ab_result_path = _resolve_path(args.ab_result).resolve()
    ab_report = _load_json(ab_result_path)
    if not isinstance(ab_report, dict):
        raise ValueError("--ab-result must contain a JSON object")
    contract = resolve_ab_contract(ab_report)
    if args.batch_size is None:
        args.batch_size = contract["reconstruction_batch_size"]
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    manifest_path = _resolve_manifest_path(ab_result_path, ab_report)
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("A/B split manifest must contain a JSON object")
    verify_manifest(
        manifest,
        expected_sha256=contract["split_manifest_sha256"],
    )
    dataset_paths = {
        "harmful_1000": _resolve_path(args.harmful_all),
        "benign_1000": _resolve_path(args.benign_all),
        "harmful_500": _resolve_path(args.harmful_test),
        "benign_500": _resolve_path(args.benign_test),
    }
    harmful_splits, benign_splits, source_sha256 = _rebuild_and_verify_splits(
        contract,
        manifest,
        paths=dataset_paths,
    )

    all_problems, problem_source = _load_gsm8k_problems(
        problems_json=args.problems_json,
        revision=args.gsm8k_revision,
    )
    problems = sample_problems(
        all_problems,
        n_problems=args.n_problems,
        seed=contract["seed"],
    )

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import torch
    from huggingface_hub import snapshot_download

    from abliterix.core.engine import SteeringEngine
    from abliterix.core.steering import apply_steering
    from abliterix.settings import AbliterixConfig
    from abliterix.types import ChatMessage, SteeringProfile
    from abliterix.vectors import compute_configured_steering_vectors

    torch.set_grad_enabled(False)
    random.seed(contract["seed"])
    torch.manual_seed(contract["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(contract["seed"])

    recorded_model_path = Path(str(ab_report["model"].get("load_path_or_id", "")))
    if (
        recorded_model_path.is_dir()
        and recorded_model_path.name == contract["model"]["revision"]
    ):
        exact_model_path = str(recorded_model_path)
    else:
        exact_model_path = snapshot_download(
            repo_id=contract["model"]["source_id"],
            revision=contract["model"]["revision"],
        )

    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]]
        root_config = AbliterixConfig(
            model={
                "model_id": exact_model_path,
                "dtype_fallback_order": [contract["model"]["dtype"]],
                "device_map": "cuda",
                "attn_implementation": contract["model"]["attention_implementation"],
                "backend": "hf",
            },
            inference={
                "batch_size": args.batch_size,
                "max_gen_tokens": args.max_new_tokens,
            },
            steering={"n_directions": contract["root_n_directions"]},
            system_prompt="",
        )
    finally:
        sys.argv = saved_argv

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or (
        f"qwen35_2b_gsm8k_{'smoke' if args.smoke else 'full'}_{timestamp}"
    )
    output_dir = _resolve_path(args.output_dir)
    result_path = output_dir / f"{run_name}.json"
    raw_path = output_dir / f"{run_name}.outputs.jsonl"
    report: dict[str, Any] = {
        "schema_version": "abliterix-gsm8k-guardrail-v1",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke" if args.smoke else "full",
        "ab_result": {
            "path": str(ab_result_path),
            "sha256": file_sha256(ab_result_path),
            "schema_version": ab_report.get("schema_version"),
        },
        "ab_contract": contract,
        "split_manifest": {
            "path": str(manifest_path),
            "sha256": manifest["manifest_sha256"],
            "rebuilt_from_local_datasets": True,
            "source_sha256": source_sha256,
        },
        "gsm8k_source": problem_source,
        "run_contract": {
            "sample_seed": contract["seed"],
            "n_problems": len(problems),
            "source_indices": [problem["source_index"] for problem in problems],
            "prompt_instruction": FINAL_ANSWER_INSTRUCTION,
            "prompt_version": "only-final-numeric-v1",
            "decoding": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "sort_by_length": True,
            "bootstrap_resamples": args.bootstrap_resamples,
        },
        "raw_outputs_path": str(raw_path),
        "provenance": _provenance(torch),
        "residual_reconstruction": None,
        "vector_reconstruction": {},
        "arms": {},
        "comparisons": {},
        "limitations": [
            "GSM8K measures narrow grade-school math capability, not general utility.",
            "The final-answer-only prompt suppresses chain-of-thought and may understate reasoning capability.",
            "Wilson intervals are per-arm; arm deltas use a paired percentile bootstrap.",
        ],
    }
    _write_json(result_path, report)

    engine = None
    try:
        print(
            f"[GSM8K] Loading {contract['model']['source_id']} at "
            f"{contract['model']['revision']}",
            flush=True,
        )
        load_started = time.perf_counter()
        engine = SteeringEngine(root_config)
        report["provenance"]["model_load_seconds"] = time.perf_counter() - load_started
        report["provenance"]["exact_model_path"] = exact_model_path
        report["provenance"]["transformer_layers"] = len(engine.transformer_layers)
        report["provenance"]["steerable_components"] = (
            engine.list_steerable_components()
        )

        train_benign = [
            ChatMessage(system="", user=str(row["prompt"]))
            for row in benign_splits["train"]
        ]
        train_harmful = [
            ChatMessage(system="", user=str(row["prompt"]))
            for row in harmful_splits["train"]
        ]
        print(
            f"[GSM8K] Rebuilding residuals: {len(train_benign)}+"
            f"{len(train_harmful)} prompts",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        residual_started = time.perf_counter()
        benign_states = engine.extract_hidden_states_batched(
            train_benign,
            sort_by_length=contract["sort_by_length"],
        )
        harmful_states = engine.extract_hidden_states_batched(
            train_harmful,
            sort_by_length=contract["sort_by_length"],
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        report["residual_reconstruction"] = {
            "seconds": time.perf_counter() - residual_started,
            "benign_shape": list(benign_states.shape),
            "harmful_shape": list(harmful_states.shape),
            "all_finite": bool(
                torch.isfinite(benign_states).all()
                and torch.isfinite(harmful_states).all()
            ),
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated() / (2**30)
                if torch.cuda.is_available()
                else 0.0
            ),
        }

        winner = contract["winner"]
        method_configs: dict[str, Any] = {}
        vectors: dict[str, Any] = {}
        for arm in ("M", winner):
            method_configs[arm] = _method_config(
                root_config,
                contract["arms"][arm]["recipe"],
            )
            vector_started = time.perf_counter()
            vectors[arm] = compute_configured_steering_vectors(
                benign_states,
                harmful_states,
                method_configs[arm],
            )
            report["vector_reconstruction"][arm] = {
                "seconds": time.perf_counter() - vector_started,
                "shape": list(vectors[arm].shape),
                "all_finite": bool(torch.isfinite(vectors[arm]).all()),
                "recipe": contract["arms"][arm]["recipe"],
            }
        _write_json(result_path, report)

        messages = build_gsm8k_messages(problems, message_type=ChatMessage)
        all_records: dict[str, list[dict[str, Any]]] = {}
        for arm in ("U", "M", winner):
            spec = contract["arms"][arm]
            _activate_arm(
                engine,
                arm=arm,
                contract=contract,
                method_configs=method_configs,
                vectors=vectors,
                benign_states=benign_states,
                harmful_states=harmful_states,
                apply_steering=apply_steering,
                profile_type=SteeringProfile,
            )
            print(
                f"[GSM8K] Evaluating {arm} strength={spec['strength']}",
                flush=True,
            )
            summary, records = _generate_arm(
                engine,
                messages,
                problems,
                arm=arm,
                strength=spec["strength"],
                max_new_tokens=args.max_new_tokens,
                torch=torch,
            )
            report["arms"][arm] = summary
            all_records[arm] = records
            _write_json(result_path, report)

        report["comparisons"] = {
            "M_minus_U": paired_bootstrap_delta_ci(
                report["arms"]["M"]["correct"],
                report["arms"]["U"]["correct"],
                seed=contract["seed"] + 1000,
                n_resamples=args.bootstrap_resamples,
            ),
            f"{winner}_minus_M": paired_bootstrap_delta_ci(
                report["arms"][winner]["correct"],
                report["arms"]["M"]["correct"],
                seed=contract["seed"] + 2000,
                n_resamples=args.bootstrap_resamples,
            ),
        }
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", encoding="utf-8") as handle:
            for arm in ("U", "M", winner):
                for record in all_records[arm]:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        engine.restore_baseline()
        report["status"] = "complete"
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(result_path, report)
        print(f"RESULT_JSON={result_path}", flush=True)
        print(f"RAW_OUTPUTS={raw_path}", flush=True)
    except BaseException as exc:
        report["status"] = "failed"
        report["failed_at"] = datetime.now(timezone.utc).isoformat()
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(result_path, report)
        raise
    finally:
        if engine is not None:
            try:
                engine.restore_baseline()
            except Exception:
                pass


if __name__ == "__main__":
    main()
