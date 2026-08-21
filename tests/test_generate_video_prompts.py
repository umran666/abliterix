"""Offline validation tests for the T2V prompt generator."""

import asyncio
from types import SimpleNamespace

import pytest

from scripts.generate_video_prompts import (
    RateLimiter,
    generate_one,
    normalize_prompt,
    sanitize_entries,
    validate_prompt_text,
)


@pytest.mark.parametrize(
    "text",
    [
        "I cannot generate prompts depicting self-harm.",
        "Sorry, but I can't create that video prompt.",
        "抱歉，我无法生成这个视频。",
    ],
)
def test_provider_refusals_are_rejected(text):
    with pytest.raises(ValueError, match="refusal"):
        validate_prompt_text(text, "bad")


@pytest.mark.parametrize(
    "text",
    [
        "A reckless teenager performs rooftop parkour without a safety line.",
        "一个未成年人站在危险的屋顶边缘。",
    ],
)
def test_unsafe_prompts_mentioning_minors_are_rejected(text):
    with pytest.raises(ValueError, match="minor"):
        validate_prompt_text(text, "bad")


def test_benign_prompt_can_mention_children_safely():
    text = "Children fly colorful kites with their parents in a sunny park."
    assert validate_prompt_text(text, "good") == text


def test_normalize_prompt_ignores_case_and_whitespace():
    assert normalize_prompt("  A  Quiet\nPark ") == normalize_prompt("a quiet park")


def test_resume_sanitization_drops_invalid_and_duplicate_rows():
    entries = [
        {"id": 1, "prompt": "A fictional adult performs a dangerous stunt."},
        {"id": 2, "prompt": " a FICTIONAL adult performs a dangerous stunt. "},
        {"id": 3, "prompt": "I cannot generate this prompt."},
        {"id": 4, "prompt": "A teenager performs a dangerous stunt."},
    ]

    clean, dropped = sanitize_entries(entries, "bad")

    assert [entry["id"] for entry in clean] == [1]
    assert dropped == 3


def test_non_retryable_http_error_is_not_retried():
    class UnauthorizedError(Exception):
        status_code = 401

    class Completions:
        calls = 0

        async def create(self, **_kwargs):
            self.calls += 1
            raise UnauthorizedError("invalid credentials")

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    task = {
        "kind": "bad",
        "category": "Dangerous Stunts",
        "language": "en",
        "shot_type": "wide/establishing shot",
        "visual_style": "documentary realism",
    }

    result = asyncio.run(
        generate_one(
            client,
            "test/model",
            task,
            asyncio.Semaphore(1),
            RateLimiter(1_000_000),
        )
    )

    assert result is None
    assert completions.calls == 1
