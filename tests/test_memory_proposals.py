from types import SimpleNamespace

import pytest

from core import llm
from memory import manager
from memory.confirmation import save_proposals
from memory.proposals import extract_proposals


def _chunks(text: str) -> list[object]:
    return [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
        )
    ]


def _fake_model(monkeypatch: pytest.MonkeyPatch, responses: list[str]) -> list[dict[str, object]]:
    requests: list[dict[str, object]] = []

    class FakeCompletions:
        def create(self, **kwargs: object) -> list[object]:
            requests.append(kwargs)
            return _chunks(responses.pop(0))

    class FakeClient:
        def __init__(self, **kwargs: str) -> None:
            assert kwargs["base_url"] == "https://example.test/v1"
            assert kwargs["api_key"] == "test-key"
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(llm, "OpenAI", FakeClient)
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    return requests


def test_invalid_json_retries_once_then_returns_no_proposals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = _fake_model(monkeypatch, ["not JSON", '{"not":"a list"}'])

    proposals = extract_proposals(
        [{"role": "user", "content": "I like gardening."}],
        llm.stream_strict_answer,
    )

    assert proposals == []
    assert len(requests) == 2
    assert all(request["stream"] is True for request in requests)
    assert all(
        request["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        for request in requests
    )


def test_invalid_json_retries_and_accepts_a_valid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid_json = (
        '[{"file":"preferences.md","text":"- Likes short answers.",'
        '"reason":"The user stated this preference."}]'
    )
    _fake_model(monkeypatch, ["not JSON", valid_json])

    proposals = extract_proposals(
        [{"role": "user", "content": "Please keep answers short."}],
        llm.stream_strict_answer,
    )

    assert proposals == [
        {
            "file": "preferences.md",
            "text": "- Likes short answers.",
            "reason": "The user stated this preference.",
        }
    ]


def test_confirmed_duplicate_memories_are_skipped(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager, "DATA_ROOT", tmp_path)
    monkeypatch.setenv("LIFEOS_USER", "test-user")
    proposals = [
        {
            "file": "preferences.md",
            "text": "- Prefers concise answers.",
            "reason": "Stated directly.",
        },
        {
            "file": "preferences.md",
            "text": "Prefers concise answers.",
            "reason": "Repeated in the conversation.",
        },
    ]

    saved, skipped = save_proposals(proposals)

    assert (saved, skipped) == (1, 1)
    assert manager.read("preferences.md") == ["- Prefers concise answers."]
