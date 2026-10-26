from datetime import date, timedelta
from pathlib import Path

import pytest

from memory import manager


@pytest.fixture(autouse=True)
def memory_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(manager, "DATA_ROOT", tmp_path)
    monkeypatch.setenv("LIFEOS_USER", "test-user")
    memory_dir = tmp_path / "test-user" / "memory"
    memory_dir.mkdir(parents=True)
    for file in manager.MEMORY_FILES:
        (memory_dir / file).touch()
    return memory_dir


def test_first_use_creates_only_the_three_memory_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh_root = tmp_path / "fresh-data"
    monkeypatch.setattr(manager, "DATA_ROOT", fresh_root)
    monkeypatch.setenv("LIFEOS_USER", "test-user")

    assert manager.list_files() == ["profile.md", "preferences.md", "events.md"]
    memory_dir = fresh_root / "test-user" / "memory"
    assert sorted(path.name for path in memory_dir.iterdir()) == [
        "events.md",
        "preferences.md",
        "profile.md",
    ]


@pytest.mark.parametrize("file", ["../profile.md", "..", "C:\\private.md", "/tmp/profile.md"])
def test_rejects_unsafe_memory_paths(file: str) -> None:
    with pytest.raises(ValueError):
        manager.read(file)


def test_rejects_unsafe_user_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIFEOS_USER", "../other-user")

    with pytest.raises(ValueError, match="LIFEOS_USER"):
        manager.list_files()


def test_add_update_and_delete_memory_lines() -> None:
    assert manager.add("profile.md", "Mira enjoys gardening.")
    assert manager.read("profile.md") == ["- Mira enjoys gardening."]

    manager.update("profile.md", 1, "Mira coordinates a garden.")
    assert manager.read("profile.md") == ["- Mira coordinates a garden."]

    manager.delete("profile.md", 1)
    assert manager.read("profile.md") == []

    with pytest.raises(IndexError):
        manager.delete("profile.md", 1)


def test_events_require_valid_dates() -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        manager.add("events.md", "The garden team meets soon.")
    with pytest.raises(ValueError, match="valid calendar date"):
        manager.add("events.md", "2026-02-30: Garden planning.")


def test_summary_is_capped_and_keeps_newest_events() -> None:
    start = date(2026, 1, 1)
    manager.list_files()
    events_path = manager.DATA_ROOT / "test-user" / "memory" / "events.md"
    events = [
        f"- {(start + timedelta(days=offset)).isoformat()}: Community garden planning event number {offset:03d}."
        for offset in range(100)
    ]
    events_path.write_text("\n".join(events) + "\n", encoding="utf-8")

    summary = manager.load_summary()

    assert len(summary) <= manager.SUMMARY_LIMIT
    assert "2026-04-10" in summary
    assert "2026-01-01" not in summary
    assert summary.index("2026-04-10") < summary.index("2026-04-09")


def test_summary_includes_profile_and_preferences() -> None:
    manager.add("profile.md", "Mira coordinates a neighborhood garden.")
    manager.add("preferences.md", "Mira likes short, practical answers.")

    summary = manager.load_summary()

    assert "Profile:" in summary
    assert "Preferences:" in summary
    assert "Mira coordinates a neighborhood garden." in summary
    assert "Mira likes short, practical answers." in summary
