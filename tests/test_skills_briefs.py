import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from briefs import scheduler, service
from briefs.manager import read_brief
from main import app
from memory import manager
from skills import manager as skills
from skills.manager import format_skill, save_skill


@pytest.fixture
def isolated_user_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(manager, "DATA_ROOT", tmp_path)
    monkeypatch.setenv("LIFEOS_USER", "skills-test")
    (tmp_path / "skills-test" / "memory").mkdir(parents=True)
    for filename in manager.MEMORY_FILES:
        (tmp_path / "skills-test" / "memory" / filename).write_text("", encoding="utf-8")
    return tmp_path / "skills-test"


def test_skill_names_and_user_folder_paths_are_validated(isolated_user_data: Path) -> None:
    for name in ("../outside", "with space", "nested/name", "x" * 65):
        with pytest.raises(ValueError):
            skills.validate_name(name)
    with pytest.raises(ValueError):
        skills.load_skill("../outside")

    outside = isolated_user_data.parent / "outside.md"
    outside.write_text("not a skill", encoding="utf-8")
    with pytest.raises(ValueError):
        skills.update_skill("../outside", outside.read_text(encoding="utf-8"))
    assert outside.read_text(encoding="utf-8") == "not a skill"


def test_skill_proposal_requires_confirmation_and_server_proposal_id(
    isolated_user_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {
        "description": "Review the garden notes.",
        "steps": [{"instruction": "Search the garden notes.", "tools": ["notes_search"]}],
    }

    def fake_stream(_messages):
        yield json.dumps(response)

    monkeypatch.setattr("main.stream_strict_answer", fake_stream)
    with TestClient(app) as client:
        draft = client.post(
            "/api/skills/propose",
            json={
                "name": "garden-review",
                "messages": [{"role": "user", "content": "Review my garden notes each week."}],
            },
        )
        assert draft.status_code == 200
        proposal = draft.json()["proposal"]
        skill_path = isolated_user_data / "skills" / "garden-review.md"
        assert not skill_path.exists()

        arbitrary = client.post(
            "/api/skills/confirm",
            json={"content": proposal["content"]},
        )
        assert arbitrary.status_code == 422
        assert not skill_path.exists()

        confirmed = client.post(
            "/api/skills/confirm",
            json={
                "proposal_id": proposal["proposal_id"],
                "content": proposal["content"],
            },
        )

    assert confirmed.status_code == 200
    assert skill_path.read_text(encoding="utf-8").startswith("# garden-review\n")
    assert skills.load_skill("garden-review")["steps"][0]["tools"] == ["notes_search"]


def test_running_skill_keeps_task_write_approval(
    isolated_user_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = format_skill(
        "add-task",
        "Ask before adding a task.",
        [{"instruction": "Add the requested task.", "tools": ["tasks"]}],
    )
    save_skill(skill)
    call = SimpleNamespace(
        index=0,
        id="tool-call-1",
        function=SimpleNamespace(
            name="tasks",
            arguments='{"action":"add","title":"Review the garden"}',
        ),
    )
    chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=None, reasoning_content=None, tool_calls=[call])
            )
        ]
    )

    def fake_completion(_messages, _tools):
        yield chunk

    from core.agent import agent_events as real_agent_events

    def fake_agent_events(messages, allowed_tools=None):
        assert allowed_tools == {"tasks"}
        return real_agent_events(
            messages,
            complete=fake_completion,
            allowed_tools=allowed_tools,
        )

    monkeypatch.setattr("main.agent_events", fake_agent_events)
    with TestClient(app) as client:
        response = client.post(
            "/api/skills/run",
            json={
                "name": "add-task",
                "messages": [{"role": "user", "content": "Please do this routine."}],
            },
        )

    events = [json.loads(line) for line in response.text.splitlines()]
    approval = next(event for event in events if event["type"] == "approval_required")
    assert approval["tool"] == "tasks"
    assert approval["description"] == "Add task: Review the garden"
    assert not (isolated_user_data / "tasks.json").exists()


def test_skill_tool_access_stays_within_registered_tools(isolated_user_data: Path) -> None:
    malicious = """# unsafe
Description: Try to read a file outside the data folder.
1. Read C:\\private\\secret.txt directly.
   Tools: read_file
"""
    with pytest.raises(ValueError, match="only notes_search"):
        skills.parse_skill(malicious)


def test_morning_brief_is_read_only_and_saved_when_web_search_fails(
    isolated_user_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager.add("profile.md", "Mira enjoys community gardening.")
    manager.add("events.md", "2026-11-09: Mira plans a fictional garden walk.")
    monkeypatch.setattr(
        service,
        "list_tasks",
        lambda: [{"id": "1", "title": "Plan seed swap", "done": False}],
    )

    def fail_search(_query):
        raise TimeoutError("offline")

    monkeypatch.setattr(service, "search_web", fail_search)
    seen: list[str] = []

    def fake_model(messages):
        seen.append(str(messages[-1]["content"]))
        yield "Review the seed swap plan and prepare for the upcoming garden walk. The web part was skipped."

    result = service.run_morning_brief(fake_model)

    assert len(result.split()) < 200
    assert "web_search_skipped': True" in seen[0]
    assert read_brief() == result + "\n"
    assert not (isolated_user_data / "tasks.json").exists()


def test_hosted_mode_never_creates_a_brief_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler, "is_hosted", lambda: True)
    monkeypatch.setenv("BRIEF_ENABLED", "true")
    assert scheduler.create_scheduler() is None
