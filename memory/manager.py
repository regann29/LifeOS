import os
import re
import tempfile
from contextvars import ContextVar, Token
from datetime import date
from pathlib import Path
from typing import Final

MEMORY_FILES: Final[tuple[str, ...]] = ("profile.md", "preferences.md", "events.md")
SUMMARY_LIMIT: Final[int] = 1500
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
_USER_DATA_DIR: ContextVar[Path | None] = ContextVar("lifeos_user_data_dir", default=None)
_USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_EVENT_DATE_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2}):\s+\S")


def _user_id() -> str:
    user_id = os.getenv("LIFEOS_USER", "demo")
    if not _USER_ID_PATTERN.fullmatch(user_id):
        raise ValueError("LIFEOS_USER must contain only letters, numbers, underscores, or hyphens.")
    return user_id


def user_data_dir() -> Path:
    override = _USER_DATA_DIR.get()
    if override is not None:
        return override
    data_root = DATA_ROOT.resolve()
    user_dir = data_root / _user_id()
    if user_dir.resolve() != user_dir:
        raise ValueError("LIFEOS_USER cannot point to a linked folder.")
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def set_user_data_dir(path: Path) -> Token[Path | None]:
    return _USER_DATA_DIR.set(path)


def reset_user_data_dir(token: Token[Path | None]) -> None:
    _USER_DATA_DIR.reset(token)


def _memory_dir() -> Path:
    path = user_data_dir() / "memory"
    path.mkdir(parents=True, exist_ok=True)
    if path.resolve() != path:
        raise ValueError("The memory folder cannot be a linked folder.")
    for name in MEMORY_FILES:
        target = path / name
        if target.is_symlink() or not target.resolve().is_relative_to(path):
            raise ValueError("Memory files must stay inside the user's memory folder.")
        if not target.exists():
            _write(target, "")
    return path


def _memory_path(file: str) -> Path:
    if not isinstance(file, str) or file not in MEMORY_FILES:
        raise ValueError("Choose one of profile.md, preferences.md, or events.md.")
    path = _memory_dir() / file
    if path.is_symlink() or not path.resolve().is_relative_to(path.parent):
        raise ValueError("Memory files must stay inside the user's memory folder.")
    return path


def _write(path: Path, content: str) -> None:
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
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def normalize_memory_text(file: str, text: str) -> str:
    if file not in MEMORY_FILES:
        raise ValueError("Choose one of profile.md, preferences.md, or events.md.")
    if not isinstance(text, str) or "\n" in text or "\r" in text:
        raise ValueError("Each memory must fit on one line.")
    text = text.strip()
    if text.startswith("- "):
        text = text[2:].strip()
    if not text or len(text) > 1000:
        raise ValueError("Memory text must be between 1 and 1000 characters.")
    if file == "events.md":
        match = _EVENT_DATE_PATTERN.match(text)
        if match is None:
            raise ValueError("Event memories must start with a date in YYYY-MM-DD: format.")
        try:
            date.fromisoformat(match.group(1))
        except ValueError as error:
            raise ValueError("Event memories must use a valid calendar date.") from error
    return f"- {text}"


def _read_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _duplicate_key(line: str) -> str:
    return " ".join(line.removeprefix("- ").split()).casefold()


def list_files() -> list[str]:
    _memory_dir()
    return list(MEMORY_FILES)


def read(file: str) -> list[str]:
    return _read_lines(_memory_path(file))


def add(file: str, text: str) -> bool:
    path = _memory_path(file)
    line = normalize_memory_text(file, text)
    lines = _read_lines(path)
    if any(_duplicate_key(existing) == _duplicate_key(line) for existing in lines):
        return False
    lines.append(line)
    _write(path, "\n".join(lines) + "\n")
    return True


def update(file: str, line_number: int, text: str) -> None:
    path = _memory_path(file)
    lines = _read_lines(path)
    if line_number < 1 or line_number > len(lines):
        raise IndexError("Memory line does not exist.")
    lines[line_number - 1] = normalize_memory_text(file, text)
    _write(path, "\n".join(lines) + "\n")


def delete(file: str, line_number: int) -> None:
    path = _memory_path(file)
    lines = _read_lines(path)
    if line_number < 1 or line_number > len(lines):
        raise IndexError("Memory line does not exist.")
    del lines[line_number - 1]
    _write(path, "\n".join(lines) + ("\n" if lines else ""))


def _summary_text(
    profile: list[str], preferences: list[str], events: list[str]
) -> str:
    sections: list[str] = []
    for title, lines in (
        ("Profile", profile),
        ("Preferences", preferences),
        ("Events", events),
    ):
        if lines:
            sections.append(f"{title}:\n" + "\n".join(lines))
    return "\n\n".join(sections)


def load_summary() -> str:
    files = {name: read(name) for name in list_files()}
    profile = files["profile.md"]
    preferences = files["preferences.md"]
    events = files["events.md"]

    def event_date(line: str) -> date:
        match = _EVENT_DATE_PATTERN.match(line.removeprefix("- "))
        if match is None:
            return date.min
        try:
            return date.fromisoformat(match.group(1))
        except ValueError:
            return date.min

    events.sort(key=event_date, reverse=True)
    summary = _summary_text(profile, preferences, events)
    while len(summary) > SUMMARY_LIMIT and events:
        events.pop()
        summary = _summary_text(profile, preferences, events)
    while len(summary) > SUMMARY_LIMIT and (preferences or profile):
        if preferences:
            preferences.pop()
        else:
            profile.pop()
        summary = _summary_text(profile, preferences, events)
    return summary
