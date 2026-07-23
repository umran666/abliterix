#!/usr/bin/env python3
"""Reproducible Qwen3.5 HF kernel benchmark and report comparator.

Run this script in separate Python processes for the base environment and an
optional FLA ``PYTHONPATH`` overlay.  The benchmark never installs packages.
"""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
import zlib
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "abliterix-qwen35-kernel-benchmark-v1"
DEFAULT_SEED = 20260710
HIDDEN_PER_CLASS = 160
SCORE_PROMPTS = 128
DECODE_PROMPTS = 128
SCORE_TOKENS = 3
DECODE_TOKENS = 64
SHARED_CONTINUATION = "The concise answer is as follows:"
MIB = 1024**2


def build_workload_samples(
    benign_prompts: Sequence[str],
    harmful_prompts: Sequence[str],
    *,
    hidden_per_class: int,
    score_count: int,
    decode_count: int,
    seed: int,
) -> dict[str, list[str]]:
    """Build deterministic, disjoint prompt samples for each workload."""
    counts = (hidden_per_class, score_count, decode_count)
    if any(count < 0 for count in counts):
        raise ValueError("workload sample counts must be non-negative")

    benign_required = hidden_per_class + score_count + decode_count
    if benign_required > len(benign_prompts):
        raise ValueError(
            f"benign workloads require {benign_required} prompts, "
            f"but only {len(benign_prompts)} are available"
        )
    if hidden_per_class > len(harmful_prompts):
        raise ValueError(
            f"hidden extraction requires {hidden_per_class} harmful prompts, "
            f"but only {len(harmful_prompts)} are available"
        )

    benign = list(benign_prompts)
    harmful = list(harmful_prompts)
    random.Random(seed).shuffle(benign)
    random.Random(seed + 1).shuffle(harmful)

    hidden_end = hidden_per_class
    score_end = hidden_end + score_count
    decode_end = score_end + decode_count
    return {
        "hidden_benign": benign[:hidden_end],
        "hidden_harmful": harmful[:hidden_per_class],
        "score": benign[hidden_end:score_end],
        "decode": benign[score_end:decode_end],
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_seconds(
    seconds: Sequence[float],
    *,
    token_count: int,
) -> dict[str, float | list[float]]:
    """Summarize repeated timings with one reproducible throughput basis."""
    samples = [float(value) for value in seconds]
    if not samples:
        raise ValueError("at least one timing sample is required")
    if any(value <= 0 for value in samples):
        raise ValueError("timing samples must be positive")
    if token_count < 0:
        raise ValueError("token_count must be non-negative")

    median = statistics.median(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": median,
        "mean_seconds": statistics.mean(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "p95_seconds": _percentile(samples, 0.95),
        "tokens_per_second": token_count / median,
    }


def batch_padding_stats(
    prompt_lengths: Sequence[int],
    *,
    batch_size: int,
    sort: bool,
) -> dict[str, int | float]:
    """Describe useful and padded prompt tokens for one batching policy."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    lengths = [int(length) for length in prompt_lengths]
    if any(length < 0 for length in lengths):
        raise ValueError("prompt lengths must be non-negative")
    if sort:
        lengths.sort()

    useful = sum(lengths)
    padded = sum(
        max(lengths[start : start + batch_size])
        * len(lengths[start : start + batch_size])
        for start in range(0, len(lengths), batch_size)
    )
    padding = padded - useful
    return {
        "useful_prompt_tokens": useful,
        "padded_prompt_tokens": padded,
        "padding_tokens": padding,
        "padding_fraction": padding / padded if padded else 0.0,
        "tensor_token_multiplier": padded / useful if useful else 0.0,
    }


def fingerprint_output(output: Any) -> dict[str, Any]:
    """Return a content hash and finite check without retaining model output."""
    if isinstance(output, torch.Tensor):
        cpu = output.detach().contiguous().cpu()
        metadata = {
            "kind": "tensor",
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype),
        }
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8"))
        digest.update(cpu.numpy().tobytes(order="C"))
        return {
            **metadata,
            "finite": bool(torch.isfinite(cpu).all().item()),
            "sha256": digest.hexdigest(),
        }

    if isinstance(output, (list, tuple)) and all(
        isinstance(item, str) for item in output
    ):
        metadata = {
            "kind": "text",
            "shape": [len(output)],
            "dtype": "str",
        }
        canonical = json.dumps(
            list(output),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode("utf-8") + canonical
        )
        return {
            **metadata,
            "finite": True,
            "sha256": digest.hexdigest(),
        }

    raise TypeError("benchmark outputs must be a tensor or a sequence of strings")


def encode_logprob_probe(logprobs: torch.Tensor) -> dict[str, Any]:
    """Encode a small full-vocabulary FP32 logprob probe inside JSON."""
    cpu = logprobs.detach().to(dtype=torch.float32, device="cpu").contiguous()
    raw = cpu.numpy().tobytes(order="C")
    return {
        "encoding": "zlib+base64-f32le",
        "shape": list(cpu.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(zlib.compress(raw, level=9)).decode("ascii"),
    }


def _decode_logprob_probe(probe: dict[str, Any]) -> torch.Tensor:
    if probe.get("encoding") != "zlib+base64-f32le":
        raise ValueError(f"unsupported logprob probe encoding: {probe.get('encoding')}")
    raw = zlib.decompress(base64.b64decode(str(probe["data"])))
    if hashlib.sha256(raw).hexdigest() != probe.get("sha256"):
        raise ValueError("logprob probe checksum mismatch")
    shape = [int(dimension) for dimension in probe["shape"]]
    expected_bytes = 4
    for dimension in shape:
        expected_bytes *= dimension
    if len(raw) != expected_bytes:
        raise ValueError(
            f"logprob probe byte count mismatch: expected {expected_bytes}, got {len(raw)}"
        )
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape)


def compare_logprob_probes(
    baseline_probe: dict[str, Any],
    candidate_probe: dict[str, Any],
) -> dict[str, Any]:
    """Compare probes under the scorer's finite-logprob contract.

    Raw ``-inf`` entries are valid masked vocabulary positions. Raw NaN and
    ``+inf`` entries invalidate correctness even though all three are replaced
    before normalization so the emitted JSON metrics remain finite.
    """
    baseline_raw = _decode_logprob_probe(baseline_probe)
    candidate_raw = _decode_logprob_probe(candidate_probe)
    if baseline_raw.shape != candidate_raw.shape:
        raise ValueError(
            "logprob probe shapes differ: "
            f"{list(baseline_raw.shape)} != {list(candidate_raw.shape)}"
        )

    raw_nan_count = {
        "baseline": int(torch.isnan(baseline_raw).sum().item()),
        "candidate": int(torch.isnan(candidate_raw).sum().item()),
    }
    raw_posinf_count = {
        "baseline": int(torch.isposinf(baseline_raw).sum().item()),
        "candidate": int(torch.isposinf(candidate_raw).sum().item()),
    }
    raw_neginf_count = {
        "baseline": int(torch.isneginf(baseline_raw).sum().item()),
        "candidate": int(torch.isneginf(candidate_raw).sum().item()),
    }
    raw_valid = not any(raw_nan_count.values()) and not any(raw_posinf_count.values())
    baseline = F.log_softmax(
        torch.nan_to_num(
            baseline_raw,
            nan=-30.0,
            posinf=0.0,
            neginf=-30.0,
        ),
        dim=-1,
        dtype=torch.float32,
    )
    candidate = F.log_softmax(
        torch.nan_to_num(
            candidate_raw,
            nan=-30.0,
            posinf=0.0,
            neginf=-30.0,
        ),
        dim=-1,
        dtype=torch.float32,
    )
    kl = (
        F.kl_div(
            candidate,
            baseline,
            reduction="none",
            log_target=True,
        )
        .sum(dim=-1)
        .mean()
        .item()
    )
    agreement = (baseline.argmax(dim=-1) == candidate.argmax(dim=-1)).float()
    max_delta = float((baseline - candidate).abs().max().item())
    metrics_finite = all(
        math.isfinite(value)
        for value in (float(kl), float(agreement.mean().item()), max_delta)
    )
    return {
        "finite": bool(raw_valid and metrics_finite),
        "raw_valid": raw_valid,
        "masked_neginf_accepted": True,
        "raw_nan_count": raw_nan_count,
        "raw_posinf_count": raw_posinf_count,
        "raw_neginf_count": raw_neginf_count,
        "kl_base_to_candidate": max(0.0, float(kl)),
        "top1_agreement": float(agreement.mean().item()),
        "max_abs_logprob_delta": max_delta,
    }


def _record_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(record["workload"]),
        str(record["mode"]),
        int(record["batch_size"]),
    )


def _records_by_key(report: dict[str, Any]) -> dict[tuple[str, str, int], dict]:
    indexed: dict[tuple[str, str, int], dict] = {}
    for record in report.get("records", []):
        key = _record_key(record)
        if key in indexed:
            raise ValueError(f"duplicate benchmark record: {key}")
        indexed[key] = record
    return indexed


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Compare two independently produced kernel benchmark JSON reports."""
    base_records = _records_by_key(baseline)
    candidate_records = _records_by_key(candidate)
    common_keys = sorted(set(base_records) & set(candidate_records))
    score_probe_comparison: dict[str, Any] | None = None
    if baseline.get("score_logprob_probe") and candidate.get("score_logprob_probe"):
        score_probe_comparison = compare_logprob_probes(
            baseline["score_logprob_probe"],
            candidate["score_logprob_probe"],
        )

    rows: list[dict[str, Any]] = []
    for key in common_keys:
        base = base_records[key]
        current = candidate_records[key]
        base_warm = float(base["warm"]["median_seconds"])
        current_warm = float(current["warm"]["median_seconds"])
        base_cold_record = base.get("cold", base.get("first_call"))
        current_cold_record = current.get("cold", current.get("first_call"))
        if base_cold_record is None or current_cold_record is None:
            raise ValueError(f"benchmark record lacks cold timing: {key}")
        base_cold = float(base_cold_record["seconds"])
        current_cold = float(current_cold_record["seconds"])
        base_vram = float(base["warm"]["peak_allocated_mib_max"])
        current_vram = float(current["warm"]["peak_allocated_mib_max"])
        base_correctness = base["correctness"]
        current_correctness = current["correctness"]
        generic_finite = bool(
            base_correctness.get("finite") and current_correctness.get("finite")
        )
        masked_score_neginf_accepted = bool(
            key[0] == "shared_continuation_scoring"
            and score_probe_comparison is not None
            and score_probe_comparison["finite"]
            and any(score_probe_comparison["raw_neginf_count"].values())
        )

        rows.append(
            {
                "workload": key[0],
                "mode": key[1],
                "batch_size": key[2],
                "base_warm_seconds": base_warm,
                "candidate_warm_seconds": current_warm,
                "warm_speedup": base_warm / current_warm,
                "base_cold_seconds": base_cold,
                "candidate_cold_seconds": current_cold,
                "cold_speedup": base_cold / current_cold,
                # Kept for compatibility with the first draft of the report.
                "first_call_speedup": base_cold / current_cold,
                "peak_allocated_mib_base": base_vram,
                "peak_allocated_mib_candidate": current_vram,
                "peak_allocated_mib_delta": current_vram - base_vram,
                "finite": generic_finite or masked_score_neginf_accepted,
                "generic_finite": generic_finite,
                "masked_score_neginf_accepted": masked_score_neginf_accepted,
                "output_hash_equal": (
                    base_correctness.get("warm_output_sha256")
                    == current_correctness.get("warm_output_sha256")
                ),
            }
        )

    warm_speedups = [float(row["warm_speedup"]) for row in rows]
    cold_speedups = [float(row["cold_speedup"]) for row in rows]
    correctness: dict[str, Any] = {
        "all_finite": bool(rows) and all(row["finite"] for row in rows),
        "output_hash_matches": sum(row["output_hash_equal"] for row in rows),
        "output_hash_records": len(rows),
        "contract_equal": baseline.get("contract") == candidate.get("contract"),
        "sample_hashes_equal": baseline.get("sample_prompt_sha256")
        == candidate.get("sample_prompt_sha256"),
        "score_probe": score_probe_comparison,
    }

    return {
        "schema_version": "abliterix-qwen35-kernel-comparison-v1",
        "baseline_label": baseline.get("label", "base"),
        "candidate_label": candidate.get("label", "candidate"),
        "records": rows,
        "summary": {
            "record_count": len(rows),
            "warm_speedup_geomean": (
                math.exp(statistics.fmean(math.log(value) for value in warm_speedups))
                if warm_speedups
                else None
            ),
            "cold_speedup_geomean": (
                math.exp(statistics.fmean(math.log(value) for value in cold_speedups))
                if cold_speedups
                else None
            ),
            "peak_allocated_mib_delta_mean": (
                statistics.fmean(float(row["peak_allocated_mib_delta"]) for row in rows)
                if rows
                else None
            ),
        },
        "missing_from_baseline": [
            list(key) for key in sorted(set(candidate_records) - set(base_records))
        ],
        "missing_from_candidate": [
            list(key) for key in sorted(set(base_records) - set(candidate_records))
        ],
        "correctness": correctness,
    }


def file_sha256(path: str | Path) -> str:
    """Hash a file without retaining its contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_sha256(prompt: str) -> str:
    """Return the stable identity used for sampled prompts."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _prompt_rows(payload: Any, *, source: Path) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("prompts", "data", "rows", "examples"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return rows
    raise ValueError(
        f"{source} must contain a JSON list or an object with a prompt list"
    )


def load_prompt_dataset(path: str | Path) -> tuple[list[str], dict[str, Any]]:
    """Load and de-duplicate prompt JSON/JSONL files with a content manifest."""
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        files = sorted(
            candidate
            for candidate in resolved.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in {".json", ".jsonl"}
        )
    elif resolved.is_file():
        files = [resolved]
    else:
        raise FileNotFoundError(f"dataset path does not exist: {resolved}")
    if not files:
        raise ValueError(f"no JSON or JSONL dataset files found under {resolved}")

    prompts: list[str] = []
    duplicate_count = 0
    seen: set[str] = set()
    for file_path in files:
        if file_path.suffix.lower() == ".jsonl":
            rows = [
                json.loads(line)
                for line in file_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            rows = _prompt_rows(
                json.loads(file_path.read_text(encoding="utf-8")),
                source=file_path,
            )
        for index, row in enumerate(rows):
            value = (
                row
                if isinstance(row, str)
                else row.get("prompt")
                if isinstance(row, dict)
                else None
            )
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"invalid prompt at {file_path}:{index + 1}; expected non-empty string"
                )
            digest = prompt_sha256(value)
            if digest in seen:
                duplicate_count += 1
                continue
            seen.add(digest)
            prompts.append(value)

    content_digest = hashlib.sha256()
    for prompt in prompts:
        content_digest.update(prompt_sha256(prompt).encode("ascii"))
        content_digest.update(b"\n")
    manifest = {
        "path": str(resolved),
        "files": [
            {
                "path": str(file_path),
                "sha256": file_sha256(file_path),
            }
            for file_path in files
        ],
        "unique_prompt_count": len(prompts),
        "duplicates_removed": duplicate_count,
        "ordered_prompt_set_sha256": content_digest.hexdigest(),
    }
    return prompts, manifest


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_origin(name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    return None if spec is None or spec.origin is None else str(spec.origin)


def environment_provenance() -> dict[str, Any]:
    """Capture enough runtime state to distinguish base and overlay processes."""
    status = _git_output("status", "--porcelain")
    properties = torch.cuda.get_device_properties(0)
    transformer_checks: dict[str, bool | str] = {}
    try:
        from transformers.utils import import_utils

        for name in (
            "is_fla_available",
            "is_flash_linear_attention_available",
        ):
            checker = getattr(import_utils, name, None)
            if callable(checker):
                try:
                    transformer_checks[name] = bool(checker())
                except Exception as error:  # pragma: no cover - package-specific
                    transformer_checks[name] = f"{type(error).__name__}: {error}"
    except ImportError:
        pass

    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": _distribution_version("transformers"),
        "triton": _distribution_version("triton"),
        "fla_core": _distribution_version("fla-core"),
        "flash_linear_attention": _distribution_version("flash-linear-attention"),
        "causal_conv1d": _distribution_version("causal-conv1d"),
        "fla_module_origin": _module_origin("fla"),
        "causal_conv1d_module_origin": _module_origin("causal_conv1d"),
        "transformers_availability_checks": transformer_checks,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_gib": properties.total_memory / (1024**3),
        },
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def kernel_provenance(engine: Any) -> dict[str, Any]:
    """Describe loaded Qwen/FLA modules without assuming one TF release layout."""
    relevant_classes: Counter[str] = Counter()
    for module_name, module in engine.model.named_modules():
        cls = type(module)
        identity = f"{cls.__module__}.{cls.__qualname__}"
        searchable = f"{module_name} {identity}".lower().replace("_", "")
        if "gateddeltanet" in searchable or cls.__module__.startswith("fla"):
            relevant_classes[identity] += 1

    source_records: list[dict[str, Any]] = []
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5

        for class_name in (
            "Qwen3_5GatedDeltaNet",
            "Qwen3_5RMSNormGated",
        ):
            cls = getattr(modeling_qwen3_5, class_name, None)
            forward = getattr(cls, "forward", None)
            if forward is None:
                continue
            try:
                source = inspect.getsource(forward)
            except (OSError, TypeError):
                source = ""
            source_records.append(
                {
                    "class": class_name,
                    "forward_module": getattr(forward, "__module__", None),
                    "source_file": inspect.getsourcefile(forward),
                    "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                    "source_mentions_fla": "fla" in source.lower(),
                }
            )
    except ImportError:
        pass

    config = getattr(engine.model, "config", None)
    return {
        "relevant_loaded_module_classes": dict(sorted(relevant_classes.items())),
        "qwen_source_records": source_records,
        "model_type": getattr(config, "model_type", None),
        "model_name_or_path": getattr(config, "name_or_path", None),
        "model_commit_hash": getattr(config, "_commit_hash", None),
        "attention_implementation": getattr(config, "_attn_implementation", None),
    }


def rendered_prompt_lengths(tokenizer: Any, prompts: Sequence[str]) -> list[int]:
    """Measure the exact unpadded chat-template lengths used by the engine."""
    chats = [[{"role": "user", "content": prompt}] for prompt in prompts]
    texts = tokenizer.apply_chat_template(
        chats,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    if isinstance(texts, str):
        texts = [texts]
    encoded = tokenizer(
        texts,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    if isinstance(encoded, torch.Tensor):
        rows = encoded.tolist()
    else:
        rows = encoded
    lengths = [len(row) for row in rows]
    if len(lengths) != len(prompts):
        raise RuntimeError(
            "tokenizer returned a different number of rendered prompt rows: "
            f"expected {len(prompts)}, got {len(lengths)}"
        )
    return lengths


def _prepare_cold_call() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _time_cuda_call(
    operation: Callable[[], Any],
) -> tuple[Any, dict[str, float]]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    started = time.perf_counter()
    with torch.inference_mode():
        output = operation()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return output, {
        "seconds": seconds,
        "baseline_allocated_mib": baseline_allocated / MIB,
        "baseline_reserved_mib": baseline_reserved / MIB,
        "peak_allocated_mib": peak_allocated / MIB,
        "peak_reserved_mib": peak_reserved / MIB,
        "peak_allocated_delta_mib": (peak_allocated - baseline_allocated) / MIB,
        "peak_reserved_delta_mib": (peak_reserved - baseline_reserved) / MIB,
    }


def benchmark_operation(
    operation: Callable[[], Any],
    *,
    token_count: int,
    warmups: int,
    repeats: int,
    capture_logprob_probe: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Measure one engine operation with allocator-cold and warmed calls."""
    _prepare_cold_call()
    cold_output, cold = _time_cuda_call(operation)
    cold_fingerprint = fingerprint_output(cold_output)
    cold["tokens_per_second"] = token_count / cold["seconds"]
    del cold_output

    for _ in range(warmups):
        warmup_output, _ = _time_cuda_call(operation)
        del warmup_output

    warm_seconds: list[float] = []
    warm_memory: list[dict[str, float]] = []
    warm_fingerprints: list[dict[str, Any]] = []
    probe: dict[str, Any] | None = None
    for repeat_index in range(repeats):
        output, memory = _time_cuda_call(operation)
        warm_seconds.append(memory["seconds"])
        warm_memory.append(memory)
        warm_fingerprints.append(fingerprint_output(output))
        if capture_logprob_probe and repeat_index == 0:
            if not isinstance(output, torch.Tensor) or output.ndim < 2:
                raise TypeError("logprob probe requires a batched tensor output")
            probe = encode_logprob_probe(output[0])
        del output

    warm = summarize_seconds(warm_seconds, token_count=token_count)
    for key in (
        "peak_allocated_mib",
        "peak_reserved_mib",
        "peak_allocated_delta_mib",
        "peak_reserved_delta_mib",
    ):
        warm[f"{key}_samples"] = [sample[key] for sample in warm_memory]
        warm[f"{key}_max"] = max(sample[key] for sample in warm_memory)

    warm_hashes = [str(item["sha256"]) for item in warm_fingerprints]
    first_warm = warm_fingerprints[0]
    return (
        {
            "cold": {**cold, "output": cold_fingerprint},
            "warm": {
                **warm,
                "warmup_count": warmups,
                "repeat_count": repeats,
                "output": first_warm,
            },
            "correctness": {
                "finite": bool(
                    cold_fingerprint["finite"]
                    and all(item["finite"] for item in warm_fingerprints)
                ),
                "cold_output_sha256": cold_fingerprint["sha256"],
                "warm_output_sha256": first_warm["sha256"],
                "cold_warm_hash_equal": cold_fingerprint["sha256"]
                == first_warm["sha256"],
                "warm_output_hash_stable": len(set(warm_hashes)) == 1,
                "warm_output_sha256_all": warm_hashes,
            },
        },
        probe,
    )


def _construct_engine_config(model: str, *, batch_size: int) -> Any:
    from abliterix.settings import AbliterixConfig

    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]]
        return AbliterixConfig(
            model={
                "model_id": model,
                "dtype_fallback_order": ["bfloat16"],
                "device_map": "cuda",
                "attn_implementation": "sdpa",
                "backend": "hf",
            },
            inference={
                "batch_size": batch_size,
                "max_gen_tokens": DECODE_TOKENS,
                "min_gen_tokens": SCORE_TOKENS,
            },
            steering={"outlier_quantile": 1.0},
            kl={"token_count": SCORE_TOKENS},
            system_prompt="",
        )
    finally:
        sys.argv = saved_argv


def _infer_label(requested: str | None, provenance: dict[str, Any]) -> str:
    if requested:
        return requested
    if provenance.get("fla_module_origin"):
        return "fla-overlay"
    return "base"


def _make_operation(
    engine: Any,
    *,
    workload: str,
    messages: list[Any],
    continuations: list[str],
    sort_by_length: bool,
) -> Callable[[], Any]:
    if workload == "hidden_extraction":
        return lambda: engine.extract_hidden_states_batched(
            messages,
            sort_by_length=sort_by_length,
        )
    if workload == "shared_continuation_scoring":
        return lambda: engine.score_continuation_logprobs_batched(
            messages,
            continuations,
            token_count=SCORE_TOKENS,
            sort_by_length=sort_by_length,
        )
    if workload == "decode_128x64":
        return lambda: engine.generate_text_batched(
            messages,
            skip_special_tokens=True,
            max_new_tokens=DECODE_TOKENS,
            min_new_tokens=DECODE_TOKENS,
            sort_by_length=sort_by_length,
        )
    raise ValueError(f"unknown benchmark workload: {workload}")


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Load one process-local model and run the fixed benchmark matrix."""
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA GPU")

    benign_prompts, benign_manifest = load_prompt_dataset(args.benign_dataset)
    harmful_prompts, harmful_manifest = load_prompt_dataset(args.harmful_dataset)
    samples = build_workload_samples(
        benign_prompts,
        harmful_prompts,
        hidden_per_class=HIDDEN_PER_CLASS,
        score_count=SCORE_PROMPTS,
        decode_count=DECODE_PROMPTS,
        seed=args.seed,
    )
    hidden_prompts = samples["hidden_benign"] + samples["hidden_harmful"]
    random.Random(args.seed + 2).shuffle(hidden_prompts)
    workload_prompts = {
        "hidden_extraction": hidden_prompts,
        "shared_continuation_scoring": samples["score"],
        "decode_128x64": samples["decode"],
    }
    sample_prompt_sha256 = {
        workload: [prompt_sha256(prompt) for prompt in prompts]
        for workload, prompts in workload_prompts.items()
    }

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)

    from abliterix.core.engine import SteeringEngine
    from abliterix.types import ChatMessage

    config = _construct_engine_config(args.model, batch_size=args.batch_sizes[0])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    load_started = time.perf_counter()
    engine = SteeringEngine(config)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    provenance = environment_provenance()
    label = _infer_label(args.label, provenance)
    messages = {
        workload: [ChatMessage(system="", user=prompt) for prompt in prompts]
        for workload, prompts in workload_prompts.items()
    }
    lengths = {
        workload: rendered_prompt_lengths(engine.tokenizer, prompts)
        for workload, prompts in workload_prompts.items()
    }
    continuation_ids = engine.tokenizer(
        SHARED_CONTINUATION,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    if isinstance(continuation_ids, torch.Tensor):
        continuation_ids = continuation_ids.tolist()
    if continuation_ids and isinstance(continuation_ids[0], list):
        continuation_ids = continuation_ids[0]
    if len(continuation_ids) < SCORE_TOKENS:
        raise RuntimeError(
            "shared continuation tokenized to fewer than three tokens: "
            f"{len(continuation_ids)}"
        )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "contract": {
            "seed": args.seed,
            "batch_sizes": args.batch_sizes,
            "modes": args.modes,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "hidden_prompts_per_class": HIDDEN_PER_CLASS,
            "score_prompts": SCORE_PROMPTS,
            "score_tokens": SCORE_TOKENS,
            "decode_prompts": DECODE_PROMPTS,
            "decode_tokens_per_prompt": DECODE_TOKENS,
            "shared_continuation_sha256": prompt_sha256(SHARED_CONTINUATION),
            "shared_continuation_first_token_ids": continuation_ids[:SCORE_TOKENS],
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "sampling": "seeded shuffle; disjoint benign workload slices",
            "cold_definition": (
                "first call for each record after gc.collect and "
                "torch.cuda.empty_cache; process/JIT caches may persist from "
                "engine load and earlier records"
            ),
            "warm_definition": "after configured discarded warmup calls",
        },
        "model": {
            "requested_model_or_path": args.model,
            "load_seconds": load_seconds,
            "load_peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
            "load_peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
        },
        "datasets": {
            "benign": benign_manifest,
            "harmful": harmful_manifest,
        },
        "sample_prompt_sha256": sample_prompt_sha256,
        "environment": provenance,
        "kernels": kernel_provenance(engine),
        "records": [],
        "score_logprob_probe": None,
        "limitations": [
            "Allocator-cold calls are not fresh-process JIT-cold calls.",
            "Token throughput uses useful tokens, while padding statistics report tensor-token overhead separately.",
            "Exact hashes are strict bitwise checks; the score probe provides a numerical KL check when hashes differ.",
            "Generated text is hashed and never stored in the report.",
        ],
    }
    output_path = Path(args.output).expanduser().resolve()
    _write_json(output_path, report)

    continuations = [SHARED_CONTINUATION] * SCORE_PROMPTS
    for workload in (
        "hidden_extraction",
        "shared_continuation_scoring",
        "decode_128x64",
    ):
        for batch_size in args.batch_sizes:
            for mode in args.modes:
                sort_by_length = mode == "sorted"
                engine.config.inference.batch_size = batch_size
                useful_prompt_tokens = sum(lengths[workload])
                if workload == "hidden_extraction":
                    throughput_tokens = useful_prompt_tokens
                    throughput_basis = "useful_prompt_tokens"
                elif workload == "shared_continuation_scoring":
                    throughput_tokens = (
                        useful_prompt_tokens + SCORE_PROMPTS * SCORE_TOKENS
                    )
                    throughput_basis = "useful_prompt_plus_scored_tokens"
                else:
                    throughput_tokens = DECODE_PROMPTS * DECODE_TOKENS
                    throughput_basis = "generated_tokens"
                operation = _make_operation(
                    engine,
                    workload=workload,
                    messages=messages[workload],
                    continuations=continuations,
                    sort_by_length=sort_by_length,
                )
                print(
                    f"[kernel-bench] {label} {workload} {mode} batch={batch_size}",
                    flush=True,
                )
                result, probe = benchmark_operation(
                    operation,
                    token_count=throughput_tokens,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    capture_logprob_probe=(
                        workload == "shared_continuation_scoring"
                        and report["score_logprob_probe"] is None
                    ),
                )
                record = {
                    "status": "ok",
                    "workload": workload,
                    "mode": mode,
                    "batch_size": batch_size,
                    "prompt_count": len(messages[workload]),
                    "throughput_token_count": throughput_tokens,
                    "throughput_basis": throughput_basis,
                    "padding": batch_padding_stats(
                        lengths[workload],
                        batch_size=batch_size,
                        sort=sort_by_length,
                    ),
                    **result,
                }
                report["records"].append(record)
                if probe is not None:
                    report["score_logprob_probe"] = {
                        **probe,
                        "record_key": [workload, mode, batch_size],
                        "prompt_sha256": sample_prompt_sha256[workload][0],
                        "token_count": SCORE_TOKENS,
                    }
                _write_json(output_path, report)

    report["status"] = "complete"
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(output_path, report)
    return report


def _load_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark report must be a JSON object: {path}")
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible Qwen3.5 HF base/FLA kernel benchmark. Run base and "
            "overlay in separate Python processes, then compare their JSON files."
        )
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument(
        "--benign-dataset",
        "--good-dataset",
        dest="benign_dataset",
        type=Path,
        default=REPO_ROOT / "datasets/good_1000/good_prompts_1000.json",
    )
    parser.add_argument(
        "--harmful-dataset",
        dest="harmful_dataset",
        type=Path,
        default=REPO_ROOT / "datasets/harmful_1000/harmful_prompts_1000.json",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("random", "sorted"),
        default=["random", "sorted"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--label")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--compare",
        type=Path,
        nargs=2,
        metavar=("BASE_JSON", "FLA_JSON"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.compare is not None:
        comparison = compare_reports(
            _load_report(args.compare[0]),
            _load_report(args.compare[1]),
        )
        if args.output is not None:
            _write_json(args.output.expanduser().resolve(), comparison)
        print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.output is None:
        raise SystemExit("--output is required for a benchmark run")
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        raise SystemExit("--batch-sizes values must be positive")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        raise SystemExit("--batch-sizes must not contain duplicates")
    if len(set(args.modes)) != len(args.modes):
        raise SystemExit("--modes must not contain duplicates")
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
