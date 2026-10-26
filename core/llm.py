import os
from collections.abc import Iterator, Sequence

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam, ChatCompletionToolParam


def stream_chat(
    messages: Sequence[ChatCompletionMessageParam],
) -> Iterator[str]:
    load_dotenv()
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    if not base_url or not api_key or not model:
        missing = [
            name
            for name, value in (
                ("LLM_BASE_URL", base_url),
                ("LLM_API_KEY", api_key),
                ("LLM_MODEL", model),
            )
            if not value
        ]
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=list(messages),
        stream=True,
        temperature=0.7,
        max_tokens=1024,
    )
    for chunk in response:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if isinstance(content, str) and content:
            yield content


def stream_agent_completion(
    messages: Sequence[ChatCompletionMessageParam],
    tools: list[ChatCompletionToolParam],
) -> Iterator[ChatCompletionChunk]:
    load_dotenv()
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    if not base_url or not api_key or not model:
        missing = [
            name
            for name, value in (
                ("LLM_BASE_URL", base_url),
                ("LLM_API_KEY", api_key),
                ("LLM_MODEL", model),
            )
            if not value
        ]
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=list(messages),
        tools=tools,
        tool_choice="auto",
        stream=True,
        temperature=0.7,
        max_tokens=1024,
    )
    yield from response
