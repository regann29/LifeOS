from pathlib import Path

import pytest

from core.prompts import chat_system_prompt
from main import (
    ChatMessage,
    ChatRequest,
    MemoryChange,
    MemoryLineChange,
    MemoryLineReference,
    MemoryProposalInput,
    MemoryProposalRequest,
    add_memory,
    confirm_memories,
    delete_memory,
    end_session,
    get_memory,
    update_memory,
)
from memory import manager


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager, "DATA_ROOT", tmp_path)
    monkeypatch.setenv("LIFEOS_USER", "api-test")
    memory_dir = tmp_path / "api-test" / "memory"
    memory_dir.mkdir(parents=True)
    for file in manager.MEMORY_FILES:
        (memory_dir / file).touch()


def test_memory_api_lists_and_changes_lines() -> None:
    initial = get_memory()
    assert [item["name"] for item in initial["files"]] == [
        "profile.md",
        "preferences.md",
        "events.md",
    ]
    assert initial["files"][0]["lines"] == []

    assert add_memory(MemoryChange(file="profile.md", text="Mira grows herbs.")) == {
        "added": True
    }
    assert add_memory(MemoryChange(file="profile.md", text="Mira grows herbs.")) == {
        "added": False
    }
    assert update_memory(
        MemoryLineChange(
            file="profile.md",
            line_number=1,
            text="Mira coordinates a garden.",
        )
    ) == {"updated": True}
    assert delete_memory(
        MemoryLineReference(file="profile.md", line_number=1)
    ) == {"deleted": True}


def test_confirm_endpoint_saves_only_submitted_items_and_skips_duplicates() -> None:
    result = confirm_memories(
        MemoryProposalRequest(
            proposals=[
                MemoryProposalInput(
                    file="preferences.md",
                    text="- Likes short answers.",
                    reason="Stated directly.",
                ),
                MemoryProposalInput(
                    file="preferences.md",
                    text="Likes short answers.",
                    reason="Said again.",
                ),
            ]
        )
    )

    assert result == {"saved": 1, "skipped_duplicates": 1}
    assert get_memory()["files"][1]["lines"] == [
        {"line_number": 1, "text": "- Likes short answers."}
    ]


def test_chat_prompt_includes_memory_and_usage_guidance() -> None:
    manager.add("profile.md", "Mira coordinates a garden.")

    prompt = chat_system_prompt()

    assert "What you know about the user:" in prompt
    assert "Mira coordinates a garden." in prompt
    assert "Never recite memories unprompted" in prompt
    assert "If a memory seems outdated, ask" in prompt


def test_end_session_uses_model_suggestions_without_saving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_stream(messages):
        assert messages[0]["role"] == "system"
        yield (
            '[{"file":"profile.md","text":"- Mira likes gardening.",'
            '"reason":"The user stated this."}]'
        )

    monkeypatch.setattr("main.stream_strict_answer", fake_stream)
    result = end_session(
        ChatRequest(messages=[ChatMessage(role="user", content="I like gardening.")])
    )

    assert result["message"] == "Review each suggestion and select the memories to save."
    assert len(result["proposals"]) == 1
    assert manager.read("profile.md") == []
