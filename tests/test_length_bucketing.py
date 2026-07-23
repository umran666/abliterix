from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from abliterix.core.engine import SteeringEngine
from abliterix.types import ChatMessage


class _FakeBatch(dict):
    def to(self, _device):
        return self


class _LengthAwareTokenizer:
    pad_token_id = 0

    _rows = {
        # Character lengths deliberately disagree with token lengths.  A
        # raw-string heuristic would produce a different ordering.
        "alpha": [11],
        "b": [22, 22, 22, 22],
        "charlie": [33, 33],
        "dd": [44, 44, 44, 44],
    }

    def apply_chat_template(self, chats, **_kwargs):
        return [chat[-1]["content"] for chat in chats]

    def __call__(self, texts, *, padding=False, **_kwargs):
        rows = [list(self._rows[text]) for text in texts]
        if not padding:
            return {"input_ids": rows}

        width = max(map(len, rows))
        padded = [[self.pad_token_id] * (width - len(row)) + row for row in rows]
        masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        return _FakeBatch(
            input_ids=torch.tensor(padded, dtype=torch.long),
            attention_mask=torch.tensor(masks, dtype=torch.long),
        )

    def batch_decode(self, outputs, **_kwargs):
        return [f"response-{int(row[-1])}" for row in outputs]


class _RecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generate_batches: list[list[tuple[int, int]]] = []
        self.forward_batches: list[list[tuple[int, int]]] = []

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def generate(self, input_ids, attention_mask, **_kwargs):
        self.generate_batches.append(
            [
                (int(row[-1]), int(mask.sum()))
                for row, mask in zip(input_ids, attention_mask)
            ]
        )
        return torch.cat((input_ids, input_ids[:, -1:]), dim=1)

    def forward(
        self,
        input_ids,
        attention_mask,
        *,
        output_hidden_states=False,
        logits_to_keep=0,
    ):
        assert output_hidden_states
        assert logits_to_keep == 1
        self.forward_batches.append(
            [
                (int(row[-1]), int(mask.sum()))
                for row, mask in zip(input_ids, attention_mask)
            ]
        )
        hidden = input_ids.to(torch.float32).unsqueeze(-1)
        return SimpleNamespace(hidden_states=(hidden, hidden + 100))


def _engine() -> tuple[SteeringEngine, _RecordingModel]:
    engine = object.__new__(SteeringEngine)
    model = _RecordingModel()
    engine.model = model
    engine.tokenizer = _LengthAwareTokenizer()
    engine.response_prefix = ""
    engine.config = SimpleNamespace(
        inference=SimpleNamespace(
            batch_size=2,
            max_gen_tokens=1,
            min_gen_tokens=None,
        ),
        steering=SimpleNamespace(outlier_quantile=1.0),
    )
    return engine, model


def _messages() -> list[ChatMessage]:
    return [
        ChatMessage(system="", user="alpha"),
        ChatMessage(system="", user="b"),
        ChatMessage(system="", user="charlie"),
        ChatMessage(system="", user="dd"),
    ]


def test_generate_text_batched_sorts_stably_by_rendered_token_length_and_restores_order():
    engine, model = _engine()

    responses = engine.generate_text_batched(_messages(), sort_by_length=True)

    assert model.generate_batches == [
        [(11, 1), (33, 2)],
        [(22, 4), (44, 4)],
    ]
    assert responses == [
        "response-11",
        "response-22",
        "response-33",
        "response-44",
    ]


def test_generate_text_batched_keeps_original_batch_order_by_default():
    engine, model = _engine()

    responses = engine.generate_text_batched(_messages())

    assert model.generate_batches == [
        [(11, 1), (22, 4)],
        [(33, 2), (44, 4)],
    ]
    assert responses == [
        "response-11",
        "response-22",
        "response-33",
        "response-44",
    ]


def test_extract_hidden_states_batched_sorts_by_length_and_restores_tensor_rows():
    engine, model = _engine()

    residuals = engine.extract_hidden_states_batched(
        _messages(),
        sort_by_length=True,
    )

    assert model.forward_batches == [
        [(11, 1), (33, 2)],
        [(22, 4), (44, 4)],
    ]
    assert residuals[:, 0, 0].tolist() == [11, 22, 33, 44]
    assert residuals[:, 1, 0].tolist() == [111, 122, 133, 144]


def test_extract_hidden_states_batched_keeps_original_batch_order_by_default():
    engine, model = _engine()

    residuals = engine.extract_hidden_states_batched(_messages())

    assert model.forward_batches == [
        [(11, 1), (22, 4)],
        [(33, 2), (44, 4)],
    ]
    assert residuals[:, 0, 0].tolist() == [11, 22, 33, 44]


def test_empty_input_contracts_remain_compatible_with_length_sorting():
    engine, model = _engine()

    assert engine.generate_text_batched([], sort_by_length=True) == []
    assert model.generate_batches == []

    with pytest.raises(ValueError):
        engine.extract_hidden_states_batched([], sort_by_length=True)

    with pytest.raises(ValueError, match="must not be empty"):
        engine.score_continuation_logprobs_batched(
            [],
            [],
            token_count=1,
            sort_by_length=True,
        )
