import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agent import agent_events, resume_approval
from memory import manager
from tools import activity
from tools.registry import TOOL_DEFINITIONS, execute_tool, prepare_tool
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

    approval_id = str(events[0]["approval_id"])
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
    approval_id = str(events[0]["approval_id"])

    resumed = list(resume_approval(approval_id, False, complete=_completion_final))

    assert list_tasks() == []
    assert activity.list_activity()[-1]["approved"] is False
    assert json.loads(activity.list_activity()[-1]["result"]) == {"denied": True}
    assert resumed[-1]["type"] == "done"


def test_task_mutations_are_risky_but_listing_is_not() -> None:
    assert prepare_tool("tasks", {"action": "list"})["risk"] is False
    assert prepare_tool("tasks", {"action": "add", "title": "Read"})["risk"] is True
    assert prepare_tool(
        "tasks",
        {"action": "complete", "task_id": "seed-001"},
    )["risk"] is True
    with pytest.raises(ValueError, match="approved"):
        execute_tool("tasks", {"action": "add", "title": "Read"})


def test_notes_search_reads_local_notes_only() -> None:
    notes = manager.user_data_dir() / "notes"
    notes.mkdir()
    (notes / "garden.md").write_text("Plant basil near the window.\n", encoding="utf-8")

    matches = execute_tool("notes_search", {"query": "basil"})

    assert matches == [{"file": "garden.md", "line": "1", "text": "Plant basil near the window."}]


def test_registry_exposes_exactly_three_native_tools() -> None:
    names = [item["function"]["name"] for item in TOOL_DEFINITIONS]
    assert names == ["notes_search", "tasks", "web_search"]


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
            "content": "Useful result",
        }
    ]


def test_web_search_reports_missing_tavily_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_search, "load_dotenv", lambda: None)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        web_search.search_web("garden planning")
