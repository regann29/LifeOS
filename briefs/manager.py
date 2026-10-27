import os
import re
import tempfile
from datetime import date
from pathlib import Path

from memory.manager import user_data_dir


def _brief_path(day: date) -> Path:
    if not isinstance(day, date):
        raise ValueError("Brief date must be a calendar date.")
    root = user_data_dir().resolve()
    directory = root / "briefs"
    if directory.is_symlink():
        raise ValueError("The briefs folder cannot be a linked folder.")
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.resolve().is_relative_to(root):
        raise ValueError("Briefs must stay inside the user's data folder.")
    path = directory / f"{day.isoformat()}.md"
    if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Brief files must stay inside the user's briefs folder.")
    return path


def save_brief(content: str, day: date | None = None) -> Path:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("A morning brief cannot be empty.")
    words = content.split()
    if len(words) >= 200:
        raise ValueError("A morning brief must be under 200 words.")
    path = _brief_path(day or date.today())
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(content.strip() + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


def read_brief(day: date | None = None) -> str | None:
    path = _brief_path(day or date.today())
    return path.read_text(encoding="utf-8") if path.exists() else None
