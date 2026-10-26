import json
import logging
import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any

from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

from core.llm import stream_agent_completion
from tools.activity import log_tool_call
from tools.registry import TOOL_DEFINITIONS, execute_tool, parse_arguments, prepare_tool

logger = logging.getLogger(__name__)
_pending_lock = Lock()
CallableCompletion = Callable[
    [Sequence[ChatCompletionMessageParam], list[dict[str, Any]]],
    Iterable[ChatCompletionChunk],
]


@dataclass
class PendingAction:
    messages: list[ChatCompletionMessageParam]
    calls: list[dict[str, Any]]
    index: int
    arguments: dict[str, Any]
    description: str


_pending: dict[str, PendingAction] = {}


def _tool_result(
    name: str,
    arguments: dict[str, Any],
    risk: bool,
    approved: bool,
) -> str:
    try:
        result = execute_tool(name, arguments, allow_risky=approved)
        log_tool_call(name, arguments, risk, approved, result)
        return json.dumps(result, ensure_ascii=False)
    except (RuntimeError, ValueError, OSError) as error:
        log_tool_call(name, arguments, risk, approved, {"error": str(error)})
        return json.dumps({"error": str(error)}, ensure_ascii=False)
    except Exception as error:
        logger.exception("Tool %s failed", name)
        log_tool_call(name, arguments, risk, approved, {"error": "Tool failed unexpectedly."})
        return json.dumps({"error": "Tool failed unexpectedly."}, ensure_ascii=False)


def _append_tool_message(
    messages: list[ChatCompletionMessageParam],
    call: dict[str, Any],
    result: str,
) -> None:
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": result,
        }
    )


def agent_events(
    messages: list[ChatCompletionMessageParam],
    complete: CallableCompletion = stream_agent_completion,
) -> Iterator[dict[str, Any]]:
    yield from _run_agent(messages, complete)


def _run_agent(
    messages: list[ChatCompletionMessageParam],
    complete: CallableCompletion,
    start_calls: list[dict[str, Any]] | None = None,
    start_index: int = 0,
) -> Iterator[dict[str, Any]]:
    calls_to_process = start_calls
    for _ in range(5):
        calls: list[dict[str, Any]] = []
        if calls_to_process is None:
            for chunk in complete(messages, TOOL_DEFINITIONS):
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                content = delta.content
                if isinstance(content, str) and content:
                    yield {"type": "delta", "text": content}
                for partial in delta.tool_calls or []:
                    index = partial.index
                    while len(calls) <= index:
                        calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    call = calls[index]
                    if partial.id:
                        call["id"] = partial.id
                    function = partial.function
                    if function is not None:
                        if function.name:
                            call["function"]["name"] += function.name
                        if function.arguments:
                            call["function"]["arguments"] += function.arguments
            if not calls:
                yield {"type": "done"}
                return
            calls_to_process = calls
            start_index = 0
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": calls,
                }
            )

        assert calls_to_process is not None
        for index in range(start_index, len(calls_to_process)):
            call = calls_to_process[index]
            name = call["function"]["name"]
            raw_arguments = call["function"]["arguments"]
            try:
                arguments = parse_arguments(raw_arguments)
                plan = prepare_tool(name, arguments)
            except (ValueError, json.JSONDecodeError) as error:
                log_tool_call(name or "unknown", {"raw": raw_arguments}, name == "tasks", False, {"error": str(error)})
                _append_tool_message(messages, call, json.dumps({"error": str(error)}))
                continue

            if plan["risk"]:
                approval_id = uuid.uuid4().hex
                with _pending_lock:
                    _pending[approval_id] = PendingAction(
                        messages=messages,
                        calls=calls_to_process,
                        index=index,
                        arguments=arguments,
                        description=plan["description"],
                    )
                yield {
                    "type": "approval_required",
                    "approval_id": approval_id,
                    "tool": name,
                    "input": arguments,
                    "description": plan["description"],
                }
                return

            result = _tool_result(name, arguments, False, True)
            _append_tool_message(messages, call, result)

        calls_to_process = None
        start_index = 0
    yield {"type": "error", "message": "Stopped after too many consecutive tool steps."}


def resume_approval(
    approval_id: str,
    approved: bool,
    complete: CallableCompletion = stream_agent_completion,
) -> Iterator[dict[str, Any]]:
    with _pending_lock:
        pending_action = _pending.pop(approval_id, None)
    if pending_action is None:
        yield {"type": "error", "message": "This approval request has expired or was already handled."}
        return

    call = pending_action.calls[pending_action.index]
    name = call["function"]["name"]
    if approved:
        result = _tool_result(name, pending_action.arguments, True, True)
    else:
        result = json.dumps({"error": "The user denied this action."})
        log_tool_call(name, pending_action.arguments, True, False, {"denied": True})
    _append_tool_message(pending_action.messages, call, result)
    yield {"type": "approval_result", "approved": approved}
    yield from _run_agent(
        pending_action.messages,
        complete,
        pending_action.calls,
        pending_action.index + 1,
    )
