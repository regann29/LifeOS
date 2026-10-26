import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from openai.types.chat import ChatCompletionMessageParam

from core.llm import stream_agent_completion, stream_chat
from tools.activity import log_tool_call
from tools.registry import TOOL_DEFINITIONS, execute_tool, parse_arguments, prepare_tool

logger = logging.getLogger(__name__)
MAX_TOOL_STEPS = 5
APPROVAL_TTL_SECONDS = 120
_clock = time.monotonic


@dataclass
class AgentState:
    messages: list[dict[str, Any]]
    answer_stream: Any | None = None
    tool_steps: int = 0
    invalid_attempts: dict[str, int] = field(default_factory=dict)
    denied_actions: set[str] = field(default_factory=set)


@dataclass
class PendingApproval:
    approval_id: str
    state: AgentState
    tool_calls: list[dict[str, Any]]
    call_index: int
    expires_at: float


_pending_approvals: dict[str, PendingApproval] = {}
_pending_lock = Lock()


def _progress_label(name: str) -> str:
    return {
        "notes_search": "Searching your notes...",
        "tasks": "Checking your task list...",
        "web_search": "Searching the web...",
    }.get(name, "Preparing a tool...")


def _append_tool_result(
    state: AgentState,
    call_id: str,
    result: object,
) -> None:
    state.messages.append(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result, ensure_ascii=False),
        }
    )


def _action_key(name: str, arguments: dict[str, Any]) -> str:
    return f"{name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"


def _tool_error(
    state: AgentState,
    call: dict[str, Any],
    name: str,
    arguments: dict[str, Any],
    error: ValueError,
) -> tuple[bool, str]:
    attempts = state.invalid_attempts.get(name, 0) + 1
    state.invalid_attempts[name] = attempts
    message = str(error)
    try:
        plan = prepare_tool(name, arguments)
        risk = plan["risk"] == "write"
    except ValueError:
        risk = name == "tasks" and arguments.get("action") in {"add", "complete"}
    log_tool_call(name, arguments, risk, False, {"error": message})
    call_id = str(call.get("id", "invalid-call"))
    if attempts >= 2:
        return False, f"I couldn't use the {name} tool because its arguments were invalid: {message}"
    _append_tool_result(
        state,
        call_id,
        {"error": message, "instruction": "Correct the arguments and retry once."},
    )
    return True, ""


def _stream_agent(
    state: AgentState,
    complete: Any,
    tool_calls: list[dict[str, Any]] | None = None,
    call_index: int = 0,
):
    while True:
        if tool_calls is None:
            gathered: dict[int, dict[str, Any]] = {}
            content = ""
            reasoning_parts: list[str] = []
            for chunk in complete(state.messages, TOOL_DEFINITIONS):
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None)
                if isinstance(reasoning, str) and reasoning:
                    reasoning_parts.append(reasoning)
                text = getattr(delta, "content", None)
                if isinstance(text, str) and text:
                    content += text
                for partial in getattr(delta, "tool_calls", None) or []:
                    index = partial.index
                    call = gathered.setdefault(
                        index,
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if partial.id:
                        call["id"] = partial.id
                    function = partial.function
                    if function.name:
                        call["function"]["name"] += function.name
                    if function.arguments:
                        call["function"]["arguments"] += function.arguments

            if not gathered:
                if state.answer_stream is not None:
                    for event in state.answer_stream(state.messages):
                        yield event
                else:
                    for reasoning in reasoning_parts:
                        yield {"type": "reasoning", "text": reasoning}
                    if content:
                        yield {"type": "answer", "text": content}
                yield {"type": "done"}
                return

            tool_calls = [gathered[index] for index in sorted(gathered)]
            state.messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": tool_calls,
                }
            )
            call_index = 0

        while call_index < len(tool_calls):
            call = tool_calls[call_index]
            function = call.get("function", {})
            name = function.get("name", "")
            call_id = str(call.get("id") or f"invalid-{uuid.uuid4().hex}")
            state.tool_steps += 1
            if state.tool_steps > MAX_TOOL_STEPS:
                try:
                    blocked_arguments = parse_arguments(function.get("arguments", ""))
                except (ValueError, json.JSONDecodeError):
                    blocked_arguments = {"raw_arguments": function.get("arguments", "")}
                try:
                    blocked_risk = prepare_tool(name, blocked_arguments)["risk"] == "write"
                except ValueError:
                    blocked_risk = name == "tasks" and blocked_arguments.get("action") in {"add", "complete"}
                log_tool_call(
                    name,
                    blocked_arguments,
                    blocked_risk,
                    False,
                    {"error": "The five-tool-step limit was reached."},
                )
                yield {
                    "type": "answer",
                    "text": "I stopped because this request reached the five-tool-step limit.",
                }
                yield {"type": "done"}
                return

            yield {"type": "tool_step", "tool": name, "label": _progress_label(name)}
            try:
                arguments = parse_arguments(function.get("arguments", ""))
            except (ValueError, json.JSONDecodeError) as error:
                invalid, explanation = _tool_error(
                    state,
                    call,
                    name,
                    {"raw_arguments": function.get("arguments", "")},
                    ValueError(str(error)),
                )
                if not invalid:
                    yield {"type": "answer", "text": explanation}
                    yield {"type": "done"}
                    return
                call_index += 1
                continue
            try:
                plan = prepare_tool(name, arguments)
            except ValueError as error:
                invalid, explanation = _tool_error(state, call, name, arguments, error)
                if not invalid:
                    yield {"type": "answer", "text": explanation}
                    yield {"type": "done"}
                    return
                call_index += 1
                continue

            state.invalid_attempts.pop(name, None)
            if plan["risk"] == "write":
                action_key = _action_key(name, arguments)
                if action_key in state.denied_actions:
                    denied = {"denied": True, "reason": "This action was already denied."}
                    log_tool_call(name, arguments, True, False, denied)
                    _append_tool_result(state, call_id, denied)
                    call_index += 1
                    continue
                approval_id = uuid.uuid4().hex
                now = _clock()
                with _pending_lock:
                    for old_id, pending in list(_pending_approvals.items()):
                        if pending.expires_at <= now:
                            del _pending_approvals[old_id]
                    _pending_approvals[approval_id] = PendingApproval(
                        approval_id=approval_id,
                        state=state,
                        tool_calls=tool_calls,
                        call_index=call_index,
                        expires_at=now + APPROVAL_TTL_SECONDS,
                    )
                yield {
                    "type": "approval_required",
                    "approval_id": approval_id,
                    "tool": name,
                    "input": arguments,
                    "description": plan["description"],
                }
                return

            try:
                result = execute_tool(name, arguments)
            except (OSError, RuntimeError, ValueError) as error:
                result = {"error": str(error)}
            log_tool_call(name, arguments, False, False, result)
            _append_tool_result(state, call_id, result)
            call_index += 1

        tool_calls = None
        call_index = 0


def agent_events(
    messages: list[ChatCompletionMessageParam],
    complete: Any = stream_agent_completion,
):
    answer_stream = stream_chat if complete is stream_agent_completion else None
    state = AgentState(
        messages=[dict(message) for message in messages],
        answer_stream=answer_stream,
    )
    yield from _stream_agent(state, complete)


def resume_approval(
    approval_id: str,
    approved: bool,
    complete: Any = stream_agent_completion,
):
    with _pending_lock:
        pending = _pending_approvals.pop(approval_id, None)
    if pending is None:
        raise RuntimeError("This approval is no longer available.")

    call = pending.tool_calls[pending.call_index]
    name = call.get("function", {}).get("name", "")
    call_id = str(call.get("id") or f"invalid-{uuid.uuid4().hex}")
    try:
        arguments = parse_arguments(call.get("function", {}).get("arguments", ""))
    except (ValueError, json.JSONDecodeError):
        arguments = {}

    expired = pending.expires_at <= _clock()
    if expired:
        approved = False
        result: object = {"error": "Approval expired after two minutes."}
    elif approved:
        try:
            result = execute_tool(name, arguments, allow_risky=True)
        except (OSError, RuntimeError, ValueError) as error:
            result = {"error": str(error)}
    else:
        result = {"denied": True}

    if not approved:
        pending.state.denied_actions.add(_action_key(name, arguments))
    log_tool_call(name, arguments, True, approved, result)
    _append_tool_result(
        pending.state,
        call_id,
        {
            **result,
            "denied": True,
        } if isinstance(result, dict) and not approved else result,
    )
    yield from _stream_agent(
        pending.state,
        complete,
        pending.tool_calls,
        pending.call_index + 1,
    )
