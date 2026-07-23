"""Behavior tests for vLLM next-token probability capture.

The vLLM package is an external GPU boundary, so these tests use the public
``VLLMGenerator.generate_and_score`` interface with RequestOutput-shaped
stand-ins. They intentionally do not import vLLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from abliterix.core.vllm_backend import VLLMGenerator
from abliterix.types import ChatMessage


@dataclass
class _Logprob:
    logprob: float


class _SamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeLLM:
    def __init__(self, outputs: list[Any], vocab_size: int = 6) -> None:
        self.outputs = outputs
        self.last_prompts: list[str] | None = None
        self.last_params: _SamplingParams | None = None
        self.last_lora_request: Any | None = None
        self.llm_engine = SimpleNamespace(
            model_config=SimpleNamespace(get_vocab_size=lambda: vocab_size)
        )

    def generate(self, prompts, params, lora_request=None):
        self.last_prompts = prompts
        self.last_params = params
        self.last_lora_request = lora_request
        return self.outputs


def _request_output(
    *,
    generated_steps: list[dict[int, _Logprob]],
    prompt_steps: list[dict[int, _Logprob] | None] | None = None,
    prompt_token_ids: list[int] | None = None,
    text: str = "answer",
) -> Any:
    completion = SimpleNamespace(text=text, logprobs=generated_steps)
    return SimpleNamespace(
        outputs=[completion],
        prompt_logprobs=prompt_steps,
        prompt_token_ids=prompt_token_ids,
    )


def _generator(outputs: list[Any], vocab_size: int = 6) -> VLLMGenerator:
    generator = VLLMGenerator.__new__(VLLMGenerator)
    generator._SamplingParams = _SamplingParams
    generator._lora_disabled = True
    generator._adapter_id = 1
    generator.llm = _FakeLLM(outputs, vocab_size=vocab_size)
    generator._format_prompts = lambda messages: [msg.user for msg in messages]
    generator.tokenizer = SimpleNamespace(
        encode=lambda _text, add_special_tokens=False: [10, 11]
    )
    return generator


def test_generate_and_score_uses_generated_next_token_distribution():
    """The KL tensor must describe the first sampled token, not the final
    prompt token distribution returned by ``prompt_logprobs``."""
    output = _request_output(
        generated_steps=[{2: _Logprob(-0.1), 1: _Logprob(-2.0)}],
        prompt_steps=[None, {4: _Logprob(-0.01), 3: _Logprob(-3.0)}],
    )
    generator = _generator([output])

    _, logprobs = generator.generate_and_score(
        [ChatMessage(system="", user="prompt")],
        max_new_tokens=1,
        kl_token_count=1,
    )

    assert logprobs.shape == (1, 6)
    assert torch.argmax(logprobs[0]).item() == 2
    assert "prompt_logprobs" not in generator.llm.last_params.kwargs


def test_generate_and_score_preserves_each_kl_step():
    """Multi-token KL keeps a step axis so the scorer can average actual
    per-step divergences instead of averaging log distributions first."""
    output = _request_output(
        generated_steps=[
            {1: _Logprob(-0.1), 4: _Logprob(-2.0)},
            {3: _Logprob(-0.2), 0: _Logprob(-2.5)},
        ],
    )
    generator = _generator([output])

    _, logprobs = generator.generate_and_score(
        [ChatMessage(system="", user="prompt")],
        max_new_tokens=2,
        kl_token_count=2,
    )

    assert logprobs.shape == (1, 2, 6)
    assert torch.argmax(logprobs[0, 0]).item() == 1
    assert torch.argmax(logprobs[0, 1]).item() == 3


def test_generate_and_score_has_fixed_shape_when_provider_returns_too_few_steps():
    """Request at least N generated tokens and deterministically fill a
    defensive short response, so every batch has shape ``(B, N, V)``."""
    output = _request_output(
        generated_steps=[{2: _Logprob(-0.1), 1: _Logprob(-2.0)}],
    )
    generator = _generator([output], vocab_size=5)

    _, logprobs = generator.generate_and_score(
        [ChatMessage(system="", user="prompt")],
        max_new_tokens=3,
        kl_token_count=3,
    )

    assert generator.llm.last_params.kwargs["min_tokens"] == 3
    assert logprobs.shape == (1, 3, 5)
    expected_uniform = torch.full((5,), torch.log(torch.tensor(1 / 5)).item())
    assert torch.allclose(logprobs[0, 1], expected_uniform)
    assert torch.allclose(logprobs[0, 2], expected_uniform)


def test_generate_and_score_normalizes_the_whole_batch_in_one_operation():
    """Do not allocate and normalise one full-vocabulary tensor per item/step."""
    outputs = [
        _request_output(
            generated_steps=[
                {1: _Logprob(-0.1), 4: _Logprob(-2.0)},
                {3: _Logprob(-0.2), 0: _Logprob(-2.5)},
            ]
        )
        for _ in range(3)
    ]
    generator = _generator(outputs, vocab_size=6)

    with patch(
        "abliterix.core.vllm_backend.F.log_softmax", wraps=F.log_softmax
    ) as log_softmax:
        _, logprobs = generator.generate_and_score(
            [ChatMessage(system="", user=f"prompt-{i}") for i in range(3)],
            max_new_tokens=2,
            kl_token_count=2,
        )

    assert logprobs.shape == (3, 2, 6)
    assert log_softmax.call_count == 1
    assert log_softmax.call_args.args[0].shape == (3, 2, 6)


def test_score_continuation_logprobs_uses_teacher_forced_continuation_positions():
    """Each KL step is the distribution predicting the corresponding fixed
    continuation token, not a prompt-tail or generated-token distribution."""
    output = _request_output(
        generated_steps=[{5: _Logprob(-0.01)}],
        prompt_token_ids=[10, 11, 2, 3],
        prompt_steps=[
            None,
            {4: _Logprob(-0.01), 0: _Logprob(-4.0)},
            {2: _Logprob(-0.1), 1: _Logprob(-2.0)},
            {3: _Logprob(-0.2), 0: _Logprob(-2.5)},
        ],
    )
    generator = _generator([output])

    logprobs = generator.score_continuation_logprobs_batched(
        [ChatMessage(system="", user="prompt")],
        [" baseline continuation"],
        token_count=2,
    )

    assert logprobs.shape == (1, 2, 6)
    assert torch.argmax(logprobs[0, 0]).item() == 2
    assert torch.argmax(logprobs[0, 1]).item() == 3
    assert generator.llm.last_params.kwargs == {
        "temperature": 0.0,
        "max_tokens": 1,
        "prompt_logprobs": 100,
    }
    assert generator.llm.last_prompts == ["prompt baseline continuation"]


def test_score_continuation_logprobs_passes_adapter_request(monkeypatch):
    class _LoRARequest:
        def __init__(self, name: str, adapter_id: int, path: str) -> None:
            self.name = name
            self.adapter_id = adapter_id
            self.path = path

    request_module = SimpleNamespace(LoRARequest=_LoRARequest)
    monkeypatch.setitem(__import__("sys").modules, "vllm", SimpleNamespace())
    monkeypatch.setitem(__import__("sys").modules, "vllm.lora", SimpleNamespace())
    monkeypatch.setitem(
        __import__("sys").modules,
        "vllm.lora.request",
        request_module,
    )

    output = _request_output(
        generated_steps=[],
        prompt_token_ids=[10, 11, 2],
        prompt_steps=[None, None, {2: _Logprob(-0.1)}],
    )
    generator = _generator([output])
    generator._lora_disabled = False
    generator._adapter_id = 7

    generator.score_continuation_logprobs_batched(
        [ChatMessage(system="", user="prompt")],
        [" continuation"],
        token_count=1,
        adapter_path="/tmp/adapter",
    )

    request = generator.llm.last_lora_request
    assert (request.name, request.adapter_id, request.path) == (
        "steering_7",
        7,
        "/tmp/adapter",
    )


def test_score_continuation_logprobs_rejects_short_continuation():
    output = _request_output(
        generated_steps=[],
        prompt_token_ids=[10, 11, 2],
        prompt_steps=[None, None, {2: _Logprob(-0.1)}],
    )
    generator = _generator([output])

    with pytest.raises(
        ValueError,
        match=r"only 1 token\(s\).*token_count=2",
    ):
        generator.score_continuation_logprobs_batched(
            [ChatMessage(system="", user="prompt")],
            [" too short"],
            token_count=2,
        )


def test_score_continuation_logprobs_rejects_ambiguous_token_boundary():
    output = _request_output(
        generated_steps=[],
        # The separately tokenized prompt is [10, 11]. Appending the
        # continuation retokenized its suffix to 12, so offset 2 is unsafe.
        prompt_token_ids=[10, 12, 2, 3],
        prompt_steps=[
            None,
            None,
            {2: _Logprob(-0.1)},
            {3: _Logprob(-0.2)},
        ],
    )
    generator = _generator([output])

    with pytest.raises(ValueError, match="token boundary"):
        generator.score_continuation_logprobs_batched(
            [ChatMessage(system="", user="prompt")],
            [" continuation"],
            token_count=2,
        )


def test_score_continuation_logprobs_rejects_missing_provider_data():
    output = _request_output(
        generated_steps=[],
        prompt_token_ids=[10, 11, 2],
        prompt_steps=None,
    )
    generator = _generator([output])

    with pytest.raises(RuntimeError, match="did not return prompt token IDs"):
        generator.score_continuation_logprobs_batched(
            [ChatMessage(system="", user="prompt")],
            [" continuation"],
            token_count=1,
        )


def test_score_continuation_logprobs_one_token_is_2d_float32():
    outputs = [
        _request_output(
            generated_steps=[],
            prompt_token_ids=[10, 11, token_id],
            prompt_steps=[None, None, {token_id: _Logprob(-0.1)}],
        )
        for token_id in (2, 3)
    ]
    generator = _generator(outputs)

    logprobs = generator.score_continuation_logprobs_batched(
        [
            ChatMessage(system="", user="prompt-a"),
            ChatMessage(system="", user="prompt-b"),
        ],
        [" continuation-a", " continuation-b"],
        token_count=1,
    )

    assert logprobs.shape == (2, 6)
    assert logprobs.dtype is torch.float32


def test_score_continuation_logprobs_normalizes_the_whole_batch_once():
    outputs = [
        _request_output(
            generated_steps=[],
            prompt_token_ids=[10, 11, 2, 3],
            prompt_steps=[
                None,
                None,
                {2: _Logprob(-0.1)},
                {3: _Logprob(-0.2)},
            ],
        )
        for _ in range(3)
    ]
    generator = _generator(outputs)

    with patch(
        "abliterix.core.vllm_backend.F.log_softmax", wraps=F.log_softmax
    ) as log_softmax:
        logprobs = generator.score_continuation_logprobs_batched(
            [ChatMessage(system="", user=f"prompt-{i}") for i in range(len(outputs))],
            [" continuation" for _ in outputs],
            token_count=2,
        )

    assert logprobs.shape == (3, 2, 6)
    assert log_softmax.call_count == 1
    assert log_softmax.call_args.args[0].shape == (3, 2, 6)
