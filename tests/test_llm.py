from types import SimpleNamespace

import pytest
from openai import OpenAIError

from core import llm
from tools.registry import TOOL_DEFINITIONS


def _chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    choices: bool = True,
) -> object:
    if not choices:
        return SimpleNamespace(choices=[])
    delta = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _setup_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[object] | None = None,
    *,
    provider: str = "nvidia",
    error: Exception | None = None,
) -> dict[str, object]:
    captured: dict[str, object] = {"calls": []}

    class FakeCompletions:
        def create(self, **kwargs: object) -> list[object]:
            captured.update(kwargs)
            captured["calls"].append(kwargs)
            if error:
                raise error
            return chunks or []

    class FakeClient:
        def __init__(self, **kwargs: str) -> None:
            captured.update(kwargs)
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(llm, "OpenAI", FakeClient)
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    return captured


def test_stream_chat_separates_reasoning_and_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _setup_fake_client(
        monkeypatch,
        [
            _chunk(choices=False),
            _chunk(reasoning="Thinking. "),
            _chunk(reasoning="More.", content="Hello"),
            _chunk(content=" there"),
        ],
    )

    events = list(llm.stream_chat([{"role": "user", "content": "Hi"}]))

    assert events == [
        {"type": "reasoning", "text": "Thinking. "},
        {"type": "reasoning", "text": "More."},
        {"type": "answer", "text": "Hello"},
        {"type": "answer", "text": " there"},
    ]
    assert captured["base_url"] == "https://example.test/v1"
    assert captured["api_key"] == "x"
    assert captured["model"] == "test-model"
    assert captured["temperature"] == 1
    assert captured["top_p"] == 0.95
    assert captured["max_tokens"] == 4096
    assert captured["stream"] is True
    assert captured["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True},
        "reasoning_budget": 2048,
    }


def test_nvidia_strict_json_mode_disables_thinking_without_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _setup_fake_client(monkeypatch, [_chunk(content="[]")])

    assert list(
        llm.stream_chat(
            [{"role": "user", "content": "Return JSON."}],
            enable_thinking=False,
        )
    ) == [{"type": "answer", "text": "[]"}]

    assert captured["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_nebius_request_omits_nvidia_extra_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _setup_fake_client(monkeypatch, [_chunk(content="Hello")], provider="nebius")

    list(llm.stream_chat([{"role": "user", "content": "Hi"}]))

    assert "extra_body" not in captured


def test_agent_tool_calls_disable_nvidia_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _setup_fake_client(monkeypatch, [])

    assert list(
        llm.stream_agent_completion(
            [{"role": "user", "content": "Find a note."}],
            TOOL_DEFINITIONS,
        )
    ) == []

    assert captured["tools"] == TOOL_DEFINITIONS
    assert captured["tool_choice"] == "auto"
    assert captured["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_placeholder_key_fails_before_client_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_fake_client(monkeypatch, [])
    monkeypatch.setenv("LLM_API_KEY", "<paste your NEW key here, nowhere else>")
    requested = False

    def unexpected_client(**_kwargs: str) -> object:
        nonlocal requested
        requested = True
        raise AssertionError("No client should be created for a placeholder key.")

    monkeypatch.setattr(llm, "OpenAI", unexpected_client)

    with pytest.raises(RuntimeError, match="The API key is missing or wrong"):
        list(llm.stream_chat([{"role": "user", "content": "Hi"}]))

    assert requested is False


def test_provider_error_does_not_leak_key_or_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_marker = "sensitive-error-marker"
    _setup_fake_client(
        monkeypatch,
        error=OpenAIError(f"provider echoed {private_marker}"),
    )

    with pytest.raises(RuntimeError) as error:
        list(llm.stream_chat([{"role": "user", "content": "Hi"}]))

    assert str(error.value) == llm.GENERIC_MODEL_ERROR
    assert private_marker not in str(error.value)


@pytest.mark.parametrize(
    ("error_name", "message"),
    [
        ("AuthenticationError", llm.API_KEY_ERROR),
        ("NotFoundError", llm.MODEL_ERROR),
        ("APIConnectionError", llm.CONNECTION_ERROR),
    ],
)
def test_provider_errors_use_friendly_messages(
    monkeypatch: pytest.MonkeyPatch,
    error_name: str,
    message: str,
) -> None:
    error_type = type(error_name, (OpenAIError,), {})
    monkeypatch.setattr(llm, error_name, error_type)
    _setup_fake_client(monkeypatch, error=error_type("private-error-marker"))

    with pytest.raises(RuntimeError) as error:
        list(llm.stream_chat([{"role": "user", "content": "Hi"}]))

    assert str(error.value) == message
    assert "private-error-marker" not in str(error.value)


def test_startup_configuration_check_returns_friendly_key_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_fake_client(monkeypatch, [])
    monkeypatch.setenv("LLM_API_KEY", "replace-with-your-new-key")

    assert llm.validate_llm_configuration() == llm.API_KEY_ERROR


def test_strict_answer_stream_discards_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_fake_client(
        monkeypatch,
        [_chunk(reasoning="must not be saved", content="[]")],
    )

    assert list(
        llm.stream_strict_answer([{"role": "user", "content": "Return JSON."}])
    ) == ["[]"]
