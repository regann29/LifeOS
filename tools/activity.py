import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from memory.manager import user_data_dir


class ActivityEntry(TypedDict):
    timestamp: str
    tool: str
    input: dict[str, Any]
    risk: bool
    approved: bool
    result: str


def _activity_path() -> Path:
    path = user_data_dir() / "activity.jsonl"
    if path.is_symlink() or not path.resolve().is_relative_to(path.parent):
        raise ValueError("Activity log must stay inside the user's data folder.")
    return path


def list_activity() -> list[ActivityEntry]:
    path = _activity_path()
    if not path.exists():
        return []
    entries: list[ActivityEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-100:]:
        if not line.strip():
            continue
        entry = json.loads(line)
        if not isinstance(entry, dict):
            raise ValueError("Activity log contains a malformed entry.")
        entries.append(entry)
    return entries


def log_tool_call(
    tool: str,
    arguments: dict[str, Any],
    risk: bool,
    approved: bool,
    result: object,
) -> None:
    path = _activity_path()
    entry: ActivityEntry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "input": arguments,
        "risk": risk,
        "approved": approved,
        "result": json.dumps(result, ensure_ascii=False),
    }
    with path.open("a", encoding="utf-8") as activity_file:
        activity_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        activity_file.flush()
        os.fsync(activity_file.fileno())
