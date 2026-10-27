import json
import os
import tempfile
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, TypedDict

from memory import manager
from memory.confirmation import save_proposals
from memory.proposals import MemoryProposal


class ProposalSelection(TypedDict):
    proposal_id: str
    text: str | None


_STORE_LOCK = Lock()


def _store_path() -> Path:
    directory = manager.user_data_dir()
    path = directory / "proposals.json"
    if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Memory proposals must stay inside the user's data folder.")
    return path


def _read_store(path: Path) -> dict[str, MemoryProposal]:
    if not path.exists():
        return {}
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("The saved proposal list is invalid.")
    proposals: dict[str, MemoryProposal] = {}
    for proposal_id, item in value.items():
        if (
            not isinstance(proposal_id, str)
            or not isinstance(item, dict)
            or set(item) != {"file", "text", "reason"}
            or not all(isinstance(item[key], str) for key in ("file", "text", "reason"))
        ):
            raise ValueError("The saved proposal list contains an invalid entry.")
        if item["file"] not in manager.MEMORY_FILES:
            raise ValueError("The saved proposal list contains an invalid memory file.")
        proposals[proposal_id] = {
            "file": item["file"],
            "text": item["text"],
            "reason": item["reason"],
        }
    return proposals


def _write_store(path: Path, proposals: dict[str, MemoryProposal]) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=".proposals.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(proposals, temporary, ensure_ascii=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def store_proposals(proposals: list[MemoryProposal]) -> list[dict[str, str]]:
    path = _store_path()
    with _STORE_LOCK:
        stored = {
            uuid.uuid4().hex: proposal
            for proposal in proposals
        }
        _write_store(path, stored)
    return [
        {"proposal_id": proposal_id, **proposal}
        for proposal_id, proposal in stored.items()
    ]


def confirm_proposals(selections: list[ProposalSelection]) -> tuple[int, int]:
    path = _store_path()
    with _STORE_LOCK:
        stored = _read_store(path)
        requested_ids = [selection["proposal_id"] for selection in selections]
        if len(set(requested_ids)) != len(requested_ids):
            raise ValueError("Each memory proposal can be selected only once.")
        missing = [proposal_id for proposal_id in requested_ids if proposal_id not in stored]
        if missing:
            raise ValueError("One or more memory proposals are unknown or already used.")

        proposals: list[MemoryProposal] = []
        for selection in selections:
            proposal = stored[selection["proposal_id"]]
            text = selection.get("text") or proposal["text"]
            proposals.append(
                {
                    "file": proposal["file"],
                    "text": manager.normalize_memory_text(proposal["file"], text),
                    "reason": proposal["reason"],
                }
            )

        saved, skipped = save_proposals(proposals)
        for proposal_id in requested_ids:
            del stored[proposal_id]
        _write_store(path, stored)
        return saved, skipped
