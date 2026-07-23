"""Behavior tests for SGLang probability capture.

The SGLang package is an external GPU boundary, so these tests use
``meta_info``-shaped stand-ins and do not import SGLang.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from abliterix.core.sglang_backend import SGLangGenerator
from abliterix.types import ChatMessage


def _top(*items: tuple[float, int]) -> list[tuple[float, int, None]]:
    return [(logprob, token_id, None) for logprob, token_id in items]


class _FakeTokenizer:
    def __init__(self, encodings: dict[str, list[int]], vocab_size: int = 7) -> None:
        self.encodings = encodings
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(self.encodings[text])


class _FakeEngine:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = outputs
        self.last_prompt: list[str] | None = None
        self.last_input_ids: list[list[int]] | None = None
        self.last_sampling_params: dict[str, Any] | None = None
        self.last_kwargs: dict[str, Any] = {}

    def generate(
        self,
        prompt: list[str] | None = None,
        sampling_params: dict[str, Any] | None = None,
        input_ids: list[list[int]] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self.last_prompt = prompt
        self.last_input_ids = input_ids
        self.last_sampling_params = sampling_params
        self.last_kwargs = kwargs
        return self.outputs


def _generator(
    outputs: list[dict[str, Any]],
    encodings: dict[str, list[int]] | None = None,
    vocab_size: int = 7,
) -> SGLangGenerator:
    generator = SGLangGenerator.__new__(SGLangGenerator)
    generator.engine = _FakeEngine(outputs)
    generator.tokenizer = _FakeTokenizer(encodings or {}, vocab_size=vocab_size)
    generator._format_prompts = lambda messages: [msg.user for msg in messages]
    return generator


def test_generate_and_score_preserves_each_kl_step() -> None:
    outputs = [
        {
            "text": "answer",
            "meta_info": {
                "output_top_logprobs": [
                    _top((-0.1, 1), (-2.0, 4)),
                    _top((-0.2, 3), (-2.5, 0)),
                ]
            },
        }
    ]
    generator = _generator(outputs)

    _, logprobs = generator.generate_and_score(
        [ChatMessage(system="", user="prompt")],
        max_new_tokens=2,
        kl_token_count=2,
    )

    assert logprobs.shape == (1, 2, 7)
    assert torch.argmax(logprobs[0, 0]).item() == 1
    assert torch.argmax(logprobs[0, 1]).item() == 3
    assert generator.engine.last_sampling_params == {
        "temperature": 0,
        "max_new_tokens": 2,
        "min_new_tokens": 2,
    }


def test_generate_and_score_single_token_keeps_2d_shape() -> None:
    generator = _generator(
        [
            {
                "text": "answer",
                "meta_info": {"output_top_logprobs": [_top((-0.1, 2), (-2.0, 1))]},
            }
        ]
    )

    _, logprobs = generator.generate_and_score(
        [ChatMessage(system="", user="prompt")],
        max_new_tokens=1,
        kl_token_count=1,
    )

    assert logprobs.shape == (1, 7)
    assert torch.argmax(logprobs[0]).item() == 2


def test_generate_and_score_densifies_and_normalizes_once() -> None:
    outputs = [
        {
            "text": "answer",
            "meta_info": {
                "output_top_logprobs": [
                    _top((-0.1, 1)),
                    _top((-0.2, 3)),
                ]
            },
        }
        for _ in range(3)
    ]
    generator = _generator(outputs)

    with patch(
        "abliterix.core.sglang_backend.F.log_softmax", wraps=F.log_softmax
    ) as log_softmax:
        _, logprobs = generator.generate_and_score(
            [ChatMessage(system="", user=f"prompt-{i}") for i in range(3)],
            max_new_tokens=2,
            kl_token_count=2,
        )

    assert logprobs.shape == (3, 2, 7)
    assert log_softmax.call_count == 1
    assert log_softmax.call_args.args[0].shape == (3, 2, 7)


def test_score_continuation_logprobs_uses_teacher_forced_input_positions() -> None:
    outputs = [
        {
            "text": "ignored",
            "meta_info": {
                # SGLang returns one leading None for logprob_start_len,
                # followed by distributions predicting each later input token.
                "input_top_logprobs": [
                    None,
                    _top((-0.1, 2), (-2.0, 1)),
                    _top((-0.2, 3), (-2.5, 0)),
                ],
                "output_top_logprobs": [_top((-0.01, 6))],
            },
        },
        {
            "text": "ignored",
            "meta_info": {
                "input_top_logprobs": [
                    None,
                    _top((-0.3, 4), (-2.0, 2)),
                    _top((-0.4, 5), (-2.5, 1)),
                ],
                "output_top_logprobs": [_top((-0.01, 0))],
            },
        },
    ]
    generator = _generator(
        outputs,
        encodings={
            "prompt-a": [10, 11],
            "prompt-a continuation": [10, 11, 2, 3, 6],
            "prompt-b": [20, 21, 22],
            "prompt-b baseline": [20, 21, 22, 4, 5, 0],
        },
    )

    with patch(
        "abliterix.core.sglang_backend.F.log_softmax", wraps=F.log_softmax
    ) as log_softmax:
        logprobs = generator.score_continuation_logprobs_batched(
            [
                ChatMessage(system="", user="prompt-a"),
                ChatMessage(system="", user="prompt-b"),
            ],
            [" continuation", " baseline"],
            token_count=2,
            adapter_path="adapter-name",
        )

    assert logprobs.shape == (2, 2, 7)
    assert torch.argmax(logprobs[0, 0]).item() == 2
    assert torch.argmax(logprobs[0, 1]).item() == 3
    assert torch.argmax(logprobs[1, 0]).item() == 4
    assert torch.argmax(logprobs[1, 1]).item() == 5
    assert log_softmax.call_count == 1
    assert log_softmax.call_args.args[0].shape == (2, 2, 7)

    assert generator.engine.last_prompt is None
    assert generator.engine.last_input_ids == [
        [10, 11, 2, 3],
        [20, 21, 22, 4, 5],
    ]
    assert generator.engine.last_sampling_params == {
        "temperature": 0,
        "max_new_tokens": 1,
    }
    assert generator.engine.last_kwargs == {
        "return_logprob": True,
        "logprob_start_len": [1, 2],
        "top_logprobs_num": 100,
        "lora_path": ["adapter-name", "adapter-name"],
    }


def test_score_continuation_logprobs_single_token_keeps_2d_shape() -> None:
    generator = _generator(
        [
            {
                "text": "ignored",
                "meta_info": {"input_top_logprobs": [None, _top((-0.1, 2))]},
            }
        ],
        encodings={
            "prompt": [10, 11],
            "prompt continuation": [10, 11, 2],
        },
    )

    logprobs = generator.score_continuation_logprobs_batched(
        [ChatMessage(system="", user="prompt")],
        [" continuation"],
        token_count=1,
    )

    assert logprobs.shape == (1, 7)
    assert torch.argmax(logprobs[0]).item() == 2


def test_score_continuation_logprobs_fails_loud_without_input_distributions() -> None:
    generator = _generator(
        [
            {
                "text": "ignored",
                "meta_info": {
                    "output_top_logprobs": [
                        _top((-0.1, 2)),
                        _top((-0.2, 3)),
                    ]
                },
            }
        ],
        encodings={
            "prompt": [10, 11],
            "prompt continuation": [10, 11, 2, 3],
        },
    )

    with pytest.raises(RuntimeError, match="input_top_logprobs.*teacher-forced"):
        generator.score_continuation_logprobs_batched(
            [ChatMessage(system="", user="prompt")],
            [" continuation"],
            token_count=2,
        )


def test_score_continuation_logprobs_rejects_ambiguous_token_boundary() -> None:
    generator = _generator(
        [],
        encodings={
            "prompt": [10, 11],
            # Appending text retokenized the prompt suffix, so there is no
            # trustworthy continuation offset in the full input.
            "prompt continuation": [10, 12, 2, 3],
        },
    )

    with pytest.raises(ValueError, match="token boundary"):
        generator.score_continuation_logprobs_batched(
            [ChatMessage(system="", user="prompt")],
            [" continuation"],
            token_count=2,
        )


@pytest.mark.parametrize("token_count", [0, -1])
def test_score_continuation_logprobs_requires_positive_token_count(
    token_count: int,
) -> None:
    generator = _generator([])

    with pytest.raises(ValueError, match="at least 1"):
        generator.score_continuation_logprobs_batched([], [], token_count=token_count)
