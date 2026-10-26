import json
from typing import Any, TypedDict

from tools.notes_search import search_notes
from tools.tasks import add_task, complete_task, list_tasks
from tools.web_search import search_web


class ToolPlan(TypedDict):
    risk: bool
    description: str


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "notes_search",
            "description": "Search the user's local notes for matching keywords.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Words to search for."}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tasks",
            "description": "List tasks, add a task, or mark a task complete. Adding or completing changes saved tasks and needs approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "add", "complete"]},
                    "title": {"type": "string", "description": "Required for add."},
                    "task_id": {"type": "string", "description": "Required for complete."},
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query."}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


def _argument_string(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip() or len(value) > 1000:
        raise ValueError(f"'{name}' must be a non-empty string of at most 1000 characters.")
    return value


def prepare_tool(name: str, arguments: dict[str, Any]) -> ToolPlan:
    if name == "notes_search":
        allowed = {"query"}
        if set(arguments) != allowed:
            raise ValueError("notes_search requires only a query.")
        query = _argument_string(arguments, "query")
        return {"risk": False, "description": f"Search local notes for: {query}"}

    if name == "web_search":
        if set(arguments) != {"query"}:
            raise ValueError("web_search requires only a query.")
        query = _argument_string(arguments, "query")
        return {"risk": False, "description": f"Search the web for: {query}"}

    if name == "tasks":
        action = _argument_string(arguments, "action")
        if action == "list":
            if set(arguments) != {"action"}:
                raise ValueError("Listing tasks accepts only the action.")
            return {"risk": False, "description": "List saved tasks"}
        if action == "add":
            if set(arguments) != {"action", "title"}:
                raise ValueError("Adding a task requires action and title.")
            title = _argument_string(arguments, "title")
            return {"risk": True, "description": f"Add task: {title}"}
        if action == "complete":
            if set(arguments) != {"action", "task_id"}:
                raise ValueError("Completing a task requires action and task_id.")
            task_id = _argument_string(arguments, "task_id")
            return {"risk": True, "description": f"Mark task {task_id} complete"}
        raise ValueError("Task action must be list, add, or complete.")

    raise ValueError(f"Unknown tool: {name}")


def execute_tool(name: str, arguments: dict[str, Any], allow_risky: bool = False) -> Any:
    plan = prepare_tool(name, arguments)
    if plan["risk"] and not allow_risky:
        raise ValueError("This tool changes data and must be explicitly approved first.")
    if name == "notes_search":
        return search_notes(_argument_string(arguments, "query"))
    if name == "web_search":
        return search_web(_argument_string(arguments, "query"))
    if name == "tasks":
        action = arguments["action"]
        if action == "list":
            return list_tasks()
        if action == "add":
            return add_task(_argument_string(arguments, "title"))
        if action == "complete":
            return complete_task(_argument_string(arguments, "task_id"))
    raise ValueError(f"Unknown tool: {name}")


def parse_arguments(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return value
