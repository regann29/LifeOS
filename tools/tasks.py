import json
import uuid
from pathlib import Path
from typing import TypedDict

from memory.manager import user_data_dir


class Task(TypedDict):
    id: str
    title: str
    done: bool


def _tasks_path() -> Path:
    path = user_data_dir() / "tasks.json"
    if path.is_symlink() or not path.resolve().is_relative_to(path.parent):
        raise ValueError("Tasks must stay inside the user's data folder.")
    return path


def _load_tasks(path: Path) -> list[Task]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("The tasks file must contain a JSON list.")
    tasks: list[Task] = []
    for item in raw:
        if (
            not isinstance(item, dict)
            or set(item) != {"id", "title", "done"}
            or not isinstance(item["id"], str)
            or not isinstance(item["title"], str)
            or not isinstance(item["done"], bool)
        ):
            raise ValueError("The tasks file contains an invalid task.")
        tasks.append({"id": item["id"], "title": item["title"], "done": item["done"]})
    return tasks


def _save_tasks(tasks: list[Task]) -> None:
    path = _tasks_path()
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def list_tasks() -> list[Task]:
    return _load_tasks(_tasks_path())


def add_task(title: str) -> Task:
    title = title.strip()
    if not title or len(title) > 200:
        raise ValueError("Task title must be between 1 and 200 characters.")
    task: Task = {"id": uuid.uuid4().hex[:8], "title": title, "done": False}
    tasks = list_tasks()
    tasks.append(task)
    _save_tasks(tasks)
    return task


def complete_task(task_id: str) -> Task:
    tasks = list_tasks()
    for task in tasks:
        if task["id"] == task_id:
            task["done"] = True
            _save_tasks(tasks)
            return task
    raise ValueError("No task found with that ID.")
