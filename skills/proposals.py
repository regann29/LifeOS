import json
import os
import tempfile
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

from memory.manager import user_data_dir
from skills.manager import Skill, parse_skill, save_skill

_LOCK = Lock()


def _store_path() -> Path:
    root = user_data_dir().resolve()
    path = root / "skill-proposals.json"
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Skill proposals must stay inside the user's data folder.")
    return path


def _read(path: Path) -> dict[str, Skill]:
    if not path.exists():
        return {}
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("The saved skill proposals are invalid.")
    proposals: dict[str, Skill] = {}
    for proposal_id, item in value.items():
        if (
            not isinstance(proposal_id, str)
            or len(proposal_id) != 32
            or not isinstance(item, dict)
            or not isinstance(item.get("content"), str)
        ):
            raise ValueError("The saved skill proposals contain an invalid entry.")
        proposals[proposal_id] = parse_skill(item["content"])
    return proposals


def _write(path: Path, proposals: dict[str, Skill]) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=".skill-proposals.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(
                {proposal_id: {"content": skill["content"]} for proposal_id, skill in proposals.items()},
                temporary,
                ensure_ascii=False,
            )
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def store_proposal(skill_content: str) -> dict[str, str]:
    skill = parse_skill(skill_content)
    proposal_id = uuid.uuid4().hex
    path = _store_path()
    with _LOCK:
        _write(path, {proposal_id: skill})
    return {"proposal_id": proposal_id, "content": skill["content"]}


def confirm_proposal(proposal_id: str, content: str | None = None) -> Skill:
    if not isinstance(proposal_id, str) or len(proposal_id) != 32:
        raise ValueError("Choose a valid skill proposal.")
    path = _store_path()
    with _LOCK:
        proposals = _read(path)
        if proposal_id not in proposals:
            raise ValueError("This skill proposal is unknown or already used.")
        original = proposals[proposal_id]
        selected_content = content if content is not None else original["content"]
        saved = save_skill(selected_content, original["name"])
        del proposals[proposal_id]
        _write(path, proposals)
        return saved
