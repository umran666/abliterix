"""Regression tests for the Hugging Face multi-token KL contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from abliterix.core.engine import SteeringEngine
from abliterix.eval.scorer import _safe_kl_divergence
from abliterix.types import ChatMessage


class _FakeBatch(dict):
    def to(self, _device):
        return self


class _TeacherForcingTokenizer:
    pad_token_id = 0

    _prompt_ids = {
        "prompt-a": [1, 2],
        "prompt-b": [2, 3, 1, 2],
    }
    _continuation_ids = {
        "base-a": [3, 1, 2],
        "base-b": [1, 3, 2],
        "short": [3],
    }

    def apply_chat_template(self, chats, **_kwargs):
        return [chat[-1]["content"] for chat in chats]

    def __call__(self, texts, *, padding=False, **_kwargs):
        ids = [
            list((self._prompt_ids | self._continuation_ids)[text]) for text in texts
        ]
        if not padding:
            return {"input_ids": ids}

        width = max(map(len, ids))
        padded = [[self.pad_token_id] * (width - len(row)) + row for row in ids]
        masks = [[0] * (width - len(row)) + [1] * len(row) for row in ids]
        return _FakeBatch(
            input_ids=torch.tensor(padded, dtype=torch.long),
            attention_mask=torch.tensor(masks, dtype=torch.long),
        )


class _PrefixSensitiveModel:
    device = torch.device("cpu")

    @staticmethod
    def _scores(input_ids, attention_mask):
        # Each position's distribution is a deterministic function of the
        # complete visible prefix.  Selecting the wrong padded position or a
        # different continuation trajectory therefore changes the result.
        prefix_sums = (input_ids * attention_mask).cumsum(dim=1).float() / 100.0
        return torch.stack(
            [prefix_sums, -prefix_sums, prefix_sums / 2, torch.zeros_like(prefix_sums)],
            dim=-1,
        )

    def __call__(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self._scores(input_ids, attention_mask))

    def generate(self, input_ids, attention_mask, **kwargs):
        for _ in range(kwargs["max_new_tokens"]):
            scores = self._scores(input_ids, attention_mask)[:, -1, :]
            for processor in kwargs["logits_processor"]:
                scores = processor(input_ids, scores)
            next_token = scores.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(next_token)), dim=1
            )
        return input_ids


def _make_teacher_forcing_engine(batch_size: int = 2) -> SteeringEngine:
    engine = object.__new__(SteeringEngine)
    engine.config = SimpleNamespace(inference=SimpleNamespace(batch_size=batch_size))
    engine.response_prefix = ""
    engine.tokenizer = _TeacherForcingTokenizer()
    engine.model = _PrefixSensitiveModel()
    return engine


def _make_engine(step_logits: list[torch.Tensor], token_count: int) -> SteeringEngine:
    """Build an engine whose generation emits deterministic score tensors."""
    engine = object.__new__(SteeringEngine)
    engine.config = SimpleNamespace(
        kl=SimpleNamespace(token_count=token_count),
        inference=SimpleNamespace(batch_size=len(step_logits[0])),
    )
    engine.tokenizer = SimpleNamespace(
        batch_decode=lambda outputs, skip_special_tokens=False: [
            "response" for _ in range(outputs.shape[0])
        ]
    )

    captured_kwargs: dict[str, object] = {}

    def fake_generate(messages, **kwargs):
        captured_kwargs.update(kwargs)
        input_ids = torch.zeros((len(messages), 2), dtype=torch.long)
        for scores in step_logits:
            for processor in kwargs["logits_processor"]:
                processor(input_ids, scores)
        outputs = torch.zeros(
            (len(messages), input_ids.shape[1] + len(step_logits)),
            dtype=torch.long,
        )
        return {"input_ids": input_ids}, outputs

    engine._generate = fake_generate  # type: ignore[method-assign]
    engine._captured_generate_kwargs = captured_kwargs  # type: ignore[attr-defined]
    return engine


def _step_logits() -> list[torch.Tensor]:
    return [
        torch.tensor([[3.0, 1.0, -1.0], [0.0, 2.0, 1.0]]),
        torch.tensor([[0.5, 1.5, -0.5], [4.0, 0.0, -2.0]]),
        torch.tensor([[-1.0, 0.0, 2.0], [1.0, 1.5, 2.0]]),
    ]


def test_score_continuation_logprobs_uses_each_shared_teacher_forced_prefix():
    engine = _make_teacher_forcing_engine()
    messages = [
        ChatMessage(system="", user="prompt-a"),
        ChatMessage(system="", user="prompt-b"),
    ]

    actual = engine.score_continuation_logprobs_batched(
        messages,
        ["base-a", "base-b"],
        token_count=3,
    )

    prefix_sums = torch.tensor([[3, 6, 7], [8, 9, 12]]).float() / 100.0
    expected_logits = torch.stack(
        [prefix_sums, -prefix_sums, prefix_sums / 2, torch.zeros_like(prefix_sums)],
        dim=-1,
    )
    expected = F.log_softmax(expected_logits, dim=-1)
    assert actual.shape == (2, 3, 4)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_score_continuation_logprobs_length_sort_keeps_pairs_and_restores_order():
    engine = _make_teacher_forcing_engine(batch_size=1)
    messages = [
        ChatMessage(system="", user="prompt-b"),
        ChatMessage(system="", user="prompt-a"),
    ]

    actual = engine.score_continuation_logprobs_batched(
        messages,
        ["base-b", "base-a"],
        token_count=3,
        sort_by_length=True,
    )

    # Expected rows remain in the caller's original [prompt-b, prompt-a]
    # order even though the shorter prompt-a is evaluated first.
    prefix_sums = torch.tensor([[8, 9, 12], [3, 6, 7]]).float() / 100.0
    expected_logits = torch.stack(
        [prefix_sums, -prefix_sums, prefix_sums / 2, torch.zeros_like(prefix_sums)],
        dim=-1,
    )
    expected = F.log_softmax(expected_logits, dim=-1)
    torch.testing.assert_close(actual, expected)


def test_score_continuation_logprobs_single_token_keeps_legacy_2d_shape():
    engine = _make_teacher_forcing_engine(batch_size=1)

    actual = engine.score_continuation_logprobs_batched(
        [ChatMessage(system="", user="prompt-a")],
        ["base-a"],
        token_count=1,
    )

    assert actual.shape == (1, 4)
    assert actual.dtype == torch.float32


def test_score_continuation_logprobs_rejects_short_continuation():
    engine = _make_teacher_forcing_engine(batch_size=1)

    with pytest.raises(ValueError, match="expected at least 2, got 1"):
        engine.score_continuation_logprobs_batched(
            [ChatMessage(system="", user="prompt-a")],
            ["short"],
            token_count=2,
        )


def test_compute_logprobs_retains_step_axis_for_multiple_tokens():
    logits = _step_logits()
    engine = _make_engine(logits, token_count=3)

    actual = engine.compute_logprobs(["prompt-a", "prompt-b"])

    expected = torch.stack([F.log_softmax(x, dim=-1) for x in logits], dim=1)
    assert actual.shape == (2, 3, 3)
    assert torch.allclose(actual, expected)
    assert engine._captured_generate_kwargs["min_new_tokens"] == 3  # type: ignore[attr-defined]


def test_compute_logprobs_single_token_keeps_legacy_2d_shape():
    engine = _make_engine(_step_logits()[:1], token_count=1)
    engine._logprobs_forward_pass = lambda messages: F.log_softmax(  # type: ignore[method-assign]
        _step_logits()[0], dim=-1
    )

    actual = engine.compute_logprobs(["prompt-a", "prompt-b"])

    assert actual.shape == (2, 3)


def test_generate_and_score_retains_step_axis_and_captures_requested_steps():
    logits = _step_logits()
    engine = _make_engine(logits, token_count=3)

    _, actual = engine.generate_and_score(
        ["prompt-a", "prompt-b"],
        max_new_tokens=5,
        kl_token_count=3,
    )

    expected = torch.stack([F.log_softmax(x, dim=-1) for x in logits], dim=1)
    assert actual.shape == (2, 3, 3)
    assert torch.allclose(actual, expected)
    assert engine._captured_generate_kwargs["min_new_tokens"] == 3  # type: ignore[attr-defined]


def test_generate_and_score_single_token_keeps_legacy_2d_shape():
    engine = _make_engine(_step_logits()[:1], token_count=1)

    _, actual = engine.generate_and_score(
        ["prompt-a", "prompt-b"],
        max_new_tokens=5,
        kl_token_count=1,
    )

    assert actual.shape == (2, 3)


def test_generate_and_score_normalizes_bfloat16_logits_in_float32():
    logits = [
        torch.tensor(
            [[12.0, 3.125, -8.5], [0.125, -2.25, 4.75]],
            dtype=torch.bfloat16,
        )
    ]
    engine = _make_engine(logits, token_count=1)

    _, actual = engine.generate_and_score(
        ["prompt-a", "prompt-b"],
        max_new_tokens=1,
        kl_token_count=1,
    )

    expected = F.log_softmax(logits[0].float(), dim=-1)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_safe_kl_averages_over_batch_and_token_axes():
    baseline = F.log_softmax(
        torch.tensor([[[2.0, 0.0, -1.0], [0.0, 3.0, 1.0]]]),
        dim=-1,
    )
    current = F.log_softmax(
        torch.tensor([[[1.0, 1.0, -2.0], [2.0, 0.0, 1.0]]]),
        dim=-1,
    )
    expected = (baseline.exp() * (baseline - current)).sum(dim=-1).mean().item()

    actual = _safe_kl_divergence(current, baseline)

    assert actual == pytest.approx(expected)


def test_safe_kl_single_token_matches_legacy_batchmean_definition():
    baseline = F.log_softmax(
        torch.tensor([[2.0, 0.0, -1.0], [0.0, 3.0, 1.0]]),
        dim=-1,
    )
    current = F.log_softmax(
        torch.tensor([[1.0, 1.0, -2.0], [2.0, 0.0, 1.0]]),
        dim=-1,
    )
    expected = F.kl_div(
        current,
        baseline,
        reduction="batchmean",
        log_target=True,
    ).item()

    actual = _safe_kl_divergence(current, baseline)

    assert actual == pytest.approx(expected)


def test_safe_kl_normalizes_finite_log_masses_and_never_returns_negative():
    distribution = F.log_softmax(torch.tensor([[2.0, 1.0, -1.0]]), dim=-1)
    current = distribution - 2.0
    baseline = distribution - 5.0

    actual = _safe_kl_divergence(current, baseline)

    assert actual == pytest.approx(0.0, abs=1e-7)
    assert actual >= 0.0
