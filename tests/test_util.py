"""Tests for abliterix.util — helper functions."""

import gc
import weakref

import torch

from abliterix.util import (
    chunk_batches,
    flush_memory,
    humanize_duration,
    reserved_unallocated_vram,
    slugify_model_name,
)


# ---------------------------------------------------------------------------
# slugify_model_name
# ---------------------------------------------------------------------------


def test_slugify_simple():
    # '.' is not alphanumeric, so it becomes '--'
    assert slugify_model_name("Qwen-0.8B") == "Qwen-0--8B"


def test_slugify_slash():
    assert slugify_model_name("Qwen/Qwen3.5-0.8B") == "Qwen--Qwen3--5-0--8B"


def test_slugify_preserves_alphanumeric():
    assert slugify_model_name("abc123") == "abc123"


def test_slugify_replaces_special_chars():
    result = slugify_model_name("a/b c:d")
    assert "/" not in result
    assert " " not in result
    assert ":" not in result


# ---------------------------------------------------------------------------
# chunk_batches
# ---------------------------------------------------------------------------


def test_chunk_exact_division():
    result = chunk_batches([1, 2, 3, 4], 2)
    assert result == [[1, 2], [3, 4]]


def test_chunk_remainder():
    result = chunk_batches([1, 2, 3, 4, 5], 2)
    assert result == [[1, 2], [3, 4], [5]]


def test_chunk_single_batch():
    result = chunk_batches([1, 2, 3], 10)
    assert result == [[1, 2, 3]]


def test_chunk_empty():
    result = chunk_batches([], 5)
    assert result == []


def test_chunk_size_one():
    result = chunk_batches([1, 2, 3], 1)
    assert result == [[1], [2], [3]]


# ---------------------------------------------------------------------------
# humanize_duration
# ---------------------------------------------------------------------------


def test_humanize_seconds():
    assert humanize_duration(45) == "45s"


def test_humanize_minutes():
    assert humanize_duration(125) == "2m 5s"


def test_humanize_hours():
    assert humanize_duration(3661) == "1h 1m"


def test_humanize_zero():
    assert humanize_duration(0) == "0s"


# ---------------------------------------------------------------------------
# flush_memory
# ---------------------------------------------------------------------------


def test_flush_memory_no_error():
    """flush_memory should not raise even without GPU."""
    flush_memory()


def test_flush_memory_flushes_cache_after_cascading_free(monkeypatch):
    """The backend cache must be emptied after the gc pass that actually
    frees the model, not just once in the middle (issue #83).

    Reproduces the failure shape: the model is only collectable on a
    *second* collector pass, because a finalizer running during the first
    pass drops the last strong reference.  With the old single
    ``collect → empty_cache → collect`` sequence, ``empty_cache()`` ran
    while the weights were still alive, and the VRAM they occupied stayed
    reserved by the caching allocator across the HF → vLLM transition.
    """
    registry = {}

    class Model:
        pass

    class Holder:
        def __init__(self):
            self._cycle = self  # only collectable by the cycle collector

        def __del__(self):
            # Runs during gc pass 1 and drops the last reference to the
            # model, which therefore only becomes garbage in pass 2.
            registry.pop("model", None)

    cache_flushes_after_free = []
    model_ref = None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: cache_flushes_after_free.append(model_ref() is None),
    )

    # Pause automatic collection while the garbage exists: a
    # threshold-triggered gen-0 pass firing between construction and
    # flush_memory() would run Holder.__del__ early and let the OLD buggy
    # code pass this test (silent false negative).
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        model = Model()
        model._cycle = model  # cycle: refcounting alone can't free it
        registry["model"] = model
        model_ref = weakref.ref(model)
        holder = Holder()
        del model, holder

        flush_memory()
    finally:
        if was_enabled:
            gc.enable()

    assert model_ref() is None, "flush_memory did not free the model"
    assert any(cache_flushes_after_free), (
        "empty_cache() never ran after the pass that freed the model — "
        "its memory would stay reserved by the caching allocator"
    )


def test_flush_memory_stops_when_nothing_left(monkeypatch):
    """Once a pass collects nothing, flush_memory stops looping — but the
    cache flush must still run after that final pass.  A model freed purely
    by refcounting (no reference cycles) produces a 0-collect first pass,
    and its VRAM still needs returning to the driver."""
    passes = []
    monkeypatch.setattr(gc, "collect", lambda *a, **k: passes.append(0) or 0)
    flushes = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: flushes.append(True))

    flush_memory(max_passes=10)

    assert len(passes) == 1
    assert len(flushes) == 1, (
        "the cache flush must run even when the first pass collects nothing"
    )


# ---------------------------------------------------------------------------
# reserved_unallocated_vram
# ---------------------------------------------------------------------------


def test_reserved_unallocated_vram_no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert reserved_unallocated_vram() == 0


def test_reserved_unallocated_vram_sums_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d: 100)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d: 30)
    assert reserved_unallocated_vram() == 140
