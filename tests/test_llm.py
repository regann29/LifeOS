from types import SimpleNamespace

import pytest

from core import llm


def test_stream_chat_reads_configuration_and_yields_text(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    chunks = [
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=" there"))]),
    ]

    class FakeCompletions:
        def create(self, **kwargs: object) -> list[object]:
            captured.update(kwargs)
            return chunks

    class FakeClient:
        def __init__(self, **kwargs: str) -> None:
            captured.update(kwargs)
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(llm, "OpenAI", FakeClient)
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    result = list(llm.stream_chat([{"role": "user", "content": "Hi"}]))

    assert result == ["Hello", " there"]
    assert captured["base_url"] == "https://example.test/v1"
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "test-model"
    assert captured["stream"] is True


def test_stream_chat_reports_missing_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="LLM_BASE_URL, LLM_API_KEY, LLM_MODEL"):
        list(llm.stream_chat([{"role": "user", "content": "Hi"}]))
