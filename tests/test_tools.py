import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import core.agent as agent
from core.agent import agent_events, resume_approval
from main import app
from memory import manager
from tools import activity
from tools.registry import TOOL_DEFINITIONS, TOOL_METADATA, execute_tool, prepare_tool
from tools.notes_search import search_notes
from tools.tasks import list_tasks
from tools import web_search


@pytest.fixture(autouse=True)
def isolated_tool_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager, "DATA_ROOT", tmp_path)
    monkeypatch.setenv("LIFEOS_USER", "tool-test")


def _chunk(*, text: str | None = None, tool_call: object | None = None) -> object:
    delta = SimpleNamespace(
        content=text,
        tool_calls=[tool_call] if tool_call is not None else None,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=None)
    return SimpleNamespace(choices=[choice])


def _tool_call(name: str, arguments: str) -> object:
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=0, id="call-1", function=function)


def _completion_with_task_call(_messages, _tools):
    return [_chunk(tool_call=_tool_call("tasks", '{"action":"add","title":"Plan garden"}'))]


def _completion_final(_messages, _tools):
    return [_chunk(text="The task is ready.")]


def test_risky_task_waits_for_approval_and_logs_result() -> None:
    messages = [{"role": "user", "content": "Add a garden task."}]
    events = list(agent_events(messages, _completion_with_task_call))

    assert events[0]["type"] == "approval_required"
    assert list_tasks() == []
    assert activity.list_activity() == []

    approval_event = next(event for event in events if event["type"] == "approval_required")
    approval_id = str(approval_event["approval_id"])
    resumed = list(
        resume_approval(
            approval_id,
            True,
            complete=_completion_final,
        )
    )

    assert {"type": "delta", "text": "The task is ready."} in resumed
    assert resumed[-1]["type"] == "done"
    assert [task["title"] for task in list_tasks()] == ["Plan garden"]
    entry = activity.list_activity()[-1]
    assert entry["tool"] == "tasks"
    assert entry["risk"] is True
    assert entry["approved"] is True
    assert json.loads(entry["result"])["title"] == "Plan garden"


def test_denied_task_is_not_executed_and_is_logged() -> None:
    messages = [{"role": "user", "content": "Add a garden task."}]
    events = list(agent_events(messages, _completion_with_task_call))
    approval_id = str(next(event for event in events if event["type"] == "approval_required")["approval_id"])

    resumed = list(resume_approval(approval_id, False, complete=_completion_final))

    assert list_tasks() == []
    assert activity.list_activity()[-1]["approved"] is False
    assert json.loads(activity.list_activity()[-1]["result"]) == {"denied": True}
    assert resumed[-1]["type"] == "done"


def test_task_mutations_are_risky_but_listing_is_not() -> None:
    assert prepare_tool("tasks", {"action": "list"})["risk"] == "read"
    assert prepare_tool("tasks", {"action": "add", "title": "Read"})["risk"] == "write"
    assert prepare_tool(
        "tasks",
        {"action": "complete", "task_id": "seed-001"},
    )["risk"] == "write"
    with pytest.raises(ValueError, match="approved"):
        execute_tool("tasks", {"action": "add", "title": "Read"})


def test_notes_search_reads_local_notes_only() -> None:
    notes = manager.user_data_dir() / "notes"
    notes.mkdir()
    (notes / "garden.md").write_text("Plant basil near the window.\n", encoding="utf-8")

    matches = execute_tool("notes_search", {"query": "basil"})

    assert matches == [
        {
            "file": "garden.md",
            "line": "1",
            "snippet": "Plant basil near the window.",
        }
    ]


def test_notes_search_returns_only_five_short_snippets() -> None:
    notes = manager.user_data_dir() / "notes"
    notes.mkdir()
    long_line = "basil " + ("garden " * 80)
    (notes / "garden.md").write_text("\n".join([long_line] * 8), encoding="utf-8")

    matches = search_notes("basil")

    assert len(matches) == 5
    assert all(len(match["snippet"]) <= 220 for match in matches)
    assert all("snippet" in match and "text" not in match for match in matches)


def test_user_folder_rejects_traversal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIFEOS_USER", "..\\other-user")

    with pytest.raises(ValueError, match="LIFEOS_USER"):
        execute_tool("tasks", {"action": "list"})


def test_registry_exposes_exactly_three_native_tools() -> None:
    names = [item["function"]["name"] for item in TOOL_DEFINITIONS]
    assert names == ["notes_search", "tasks", "web_search"]
    assert [TOOL_METADATA[name]["risk"] for name in names] == [
        "read",
        "read or write depending on action",
        "read",
    ]


def test_web_search_uses_tavily_and_returns_limited_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeTavilyClient:
        def __init__(self, api_key: str) -> None:
            captured["api_key"] = api_key

        def search(self, **kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return {
                "results": [
                    {
                        "title": "Garden guide",
                        "url": "https://example.test/garden",
                        "content": "Useful result",
                        "raw_content": "not returned",
                    }
                ]
            }

    monkeypatch.setattr(web_search, "TavilyClient", FakeTavilyClient)
    monkeypatch.setattr(web_search, "load_dotenv", lambda: None)
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")

    results = web_search.search_web("garden planning")

    assert captured["api_key"] == "test-tavily-key"
    assert captured["query"] == "garden planning"
    assert results == [
        {
            "title": "Garden guide",
            "url": "https://example.test/garden",
            "summary": "Useful result",
        }
    ]


def test_web_search_reports_missing_tavily_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_search, "load_dotenv", lambda: None)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    assert web_search.search_web("garden planning") == [
        {
            "title": "Web search is not set up",
            "url": "",
            "summary": "Add TAVILY_API_KEY to your local .env file to enable web search.",
        }
    ]


def test_invalid_tool_arguments_retry_once_then_explain() -> None:
    response_count = 0

    def complete(_messages, _tools):
        nonlocal response_count
        response_count += 1
        return [_chunk(tool_call=_tool_call("notes_search", '{"unexpected":"field"}'))]

    events = list(agent_events([{"role": "user", "content": "Search my notes."}], complete))

    assert response_count == 2
    assert "arguments were invalid" in "".join(
        str(event.get("text", "")) for event in events if event["type"] == "delta"
    )


def test_agent_caps_tool_steps_across_model_turns() -> None:
    response_count = 0

    def complete(_messages, _tools):
        nonlocal response_count
        response_count += 1
        return [_chunk(tool_call=_tool_call("tasks", '{"action":"list"}'))]

    events = list(agent_events([{"role": "user", "content": "List tasks repeatedly."}], complete))
    delta_text = "".join(
        str(event.get("text", "")) for event in events if event["type"] == "delta"
    )

    assert response_count == 6
    assert delta_text == "I stopped because this request reached the five-tool-step limit."
    assert len(activity.list_activity()) == 5


def test_approval_expires_after_two_minutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [10.0]
    monkeypatch.setattr(agent, "_clock", lambda: clock[0])
    events = list(agent_events(
        [{"role": "user", "content": "Add a garden task."}],
        _completion_with_task_call,
    ))
    approval_id = str(next(event for event in events if event["type"] == "approval_required")["approval_id"])
    clock[0] = 131.0

    resumed = list(resume_approval(approval_id, True, complete=_completion_final))

    assert list_tasks() == []
    assert activity.list_activity()[-1]["approved"] is False
    assert "expired" in activity.list_activity()[-1]["result"]
    assert resumed[-1]["type"] == "done"


def test_denied_action_is_not_retried(
) -> None:
    response_count = 0

    def complete(_messages, _tools):
        nonlocal response_count
        response_count += 1
        if response_count <= 2:
            return [_chunk(tool_call=_tool_call("tasks", '{"action":"add","title":"Plan garden"}'))]
        return [_chunk(text="I will leave it unchanged.")]

    events = list(agent_events([{"role": "user", "content": "Add a garden task."}], complete))
    approval_id = str(next(event for event in events if event["type"] == "approval_required")["approval_id"])
    resumed = list(resume_approval(approval_id, False, complete=complete))

    assert not any(event["type"] == "approval_required" for event in resumed)
    assert list_tasks() == []
    assert resumed[-1]["type"] == "done"


def test_chat_and_approval_endpoints_stream_agent_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "main.agent_events",
        lambda messages: iter([{"type": "approval_required", "approval_id": "a" * 32}]),
    )
    monkeypatch.setattr(
        "main.resume_approval",
        lambda approval_id, approved: iter([{"type": "done"}]),
    )
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "Add a task."}]},
    )
    assert response.status_code == 200
    assert json.loads(response.text) == {
        "type": "approval_required",
        "approval_id": "a" * 32,
    }

    decision = client.post(
        "/api/approval",
        json={"approval_id": "a" * 32, "approved": False},
    )
    assert decision.status_code == 200
    assert json.loads(decision.text) == {"type": "done"}
