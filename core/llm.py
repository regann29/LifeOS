import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    OpenAIError,
)
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from core import hosted

API_KEY_ERROR = "The API key is missing or wrong. Check LLM_API_KEY in .env, then restart the server."
MODEL_ERROR = "Model name not found. Check LLM_MODEL."
CONNECTION_ERROR = "Cannot reach the model. Check LLM_BASE_URL."
GENERIC_MODEL_ERROR = "The model request failed. Check provider settings and try again."


class StreamEvent(TypedDict):
    type: str
    text: str


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    base_url: str
    api_key: str
    model: str
    enable_thinking: bool
    max_tokens: int
    reasoning_budget: int


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().casefold()
    return (
        not normalized
        or normalized.startswith("<")
        or "replace-with" in normalized
        or "paste your" in normalized
        or "your-" in normalized
    )


def _positive_integer(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a positive integer.") from error
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer.")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().casefold()
    if raw in {"true", "1", "yes", "on"}:
        return True
    if raw in {"false", "0", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true or false.")


def _settings() -> LLMSettings:
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "nebius").strip().casefold()
    if provider not in {"nvidia", "nebius"}:
        raise RuntimeError("LLM_PROVIDER must be one of: nvidia, nebius.")

    api_key = os.getenv("LLM_API_KEY", "").strip()
    if _is_placeholder(api_key):
        raise RuntimeError(API_KEY_ERROR)

    base_url = os.getenv("LLM_BASE_URL", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    if not base_url:
        raise RuntimeError("LLM_BASE_URL is missing.")
    if not model:
        raise RuntimeError("LLM_MODEL is missing.")

    return LLMSettings(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        enable_thinking=_boolean("LLM_ENABLE_THINKING", True),
        max_tokens=min(_positive_integer("LLM_MAX_TOKENS", 4096), hosted.HOSTED_MAX_TOKENS)
        if hosted.is_hosted()
        else _positive_integer("LLM_MAX_TOKENS", 4096),
        reasoning_budget=_positive_integer("LLM_REASONING_BUDGET", 2048)
        if provider == "nvidia"
        else 2048,
    )


def validate_llm_configuration() -> str | None:
    try:
        _settings()
    except RuntimeError as error:
        return str(error)
    return None


def friendly_provider_error(error: OpenAIError) -> str:
    if isinstance(error, AuthenticationError):
        return API_KEY_ERROR
    if isinstance(error, NotFoundError):
        return MODEL_ERROR
    if isinstance(error, APIConnectionError):
        return CONNECTION_ERROR
    status_code = getattr(error, "status_code", None)
    if status_code == 401:
        return API_KEY_ERROR
    if status_code == 404:
        return MODEL_ERROR
    return GENERIC_MODEL_ERROR


def _request_body(settings: LLMSettings, enable_thinking: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "temperature": 1,
        "top_p": 0.95,
        "max_tokens": settings.max_tokens,
        "stream": True,
    }
    if settings.provider == "nvidia":
        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }
        if enable_thinking:
            extra_body["reasoning_budget"] = settings.reasoning_budget
        body["extra_body"] = extra_body
    return body


def _create_stream(
    messages: Sequence[ChatCompletionMessageParam],
    *,
    tools: list[ChatCompletionToolParam] | None = None,
    enable_thinking: bool | None = None,
) -> Iterator[ChatCompletionChunk]:
    settings = _settings()
    thinking = (
        settings.enable_thinking
        if enable_thinking is None
        else enable_thinking
    )
    body = _request_body(settings, thinking)
    if tools is not None:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    try:
        hosted.consume_model_call()
        client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=settings.model,
            messages=list(messages),
            **body,
        )
        yield from response
    except OpenAIError as error:
        raise RuntimeError(friendly_provider_error(error)) from None


def stream_chat(
    messages: Sequence[ChatCompletionMessageParam],
    *,
    enable_thinking: bool | None = None,
) -> Iterator[StreamEvent]:
    for chunk in _create_stream(messages, enable_thinking=enable_thinking):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning:
            yield {"type": "reasoning", "text": reasoning}
        content = delta.content
        if isinstance(content, str) and content:
            yield {"type": "answer", "text": content}


def stream_answer(
    messages: Sequence[ChatCompletionMessageParam],
    *,
    enable_thinking: bool | None = None,
) -> Iterator[str]:
    for event in stream_chat(messages, enable_thinking=enable_thinking):
        if event["type"] == "answer":
            yield event["text"]


def stream_strict_answer(
    messages: Sequence[ChatCompletionMessageParam],
) -> Iterator[str]:
    yield from stream_answer(messages, enable_thinking=False)


def stream_agent_completion(
    messages: Sequence[ChatCompletionMessageParam],
    tools: list[ChatCompletionToolParam],
) -> Iterator[ChatCompletionChunk]:
    yield from _create_stream(messages, tools=tools or None, enable_thinking=False)
