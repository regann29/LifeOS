import json
import os
import tempfile
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
    path = user_data_dir() / "activity.json"
    if path.is_symlink() or not path.resolve().is_relative_to(path.parent):
        raise ValueError("Activity log must stay inside the user's data folder.")
    return path


def list_activity() -> list[ActivityEntry]:
    path = _activity_path()
    if not path.exists():
        return []
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("Activity log must contain a JSON list.")
    return entries[-100:]


def log_tool_call(
    tool: str,
    arguments: dict[str, Any],
    risk: bool,
    approved: bool,
    result: object,
) -> None:
    path = _activity_path()
    entries = list_activity()
    entries.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "input": arguments,
            "risk": risk,
            "approved": approved,
            "result": json.dumps(result, ensure_ascii=False),
        }
    )
    entries = entries[-100:]
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".activity.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(entries, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
