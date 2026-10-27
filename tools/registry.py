import json
import logging
from typing import Any, Literal, TypedDict

from openai.types.chat import ChatCompletionToolParam
from requests.exceptions import RequestException, Timeout

from tools.notes_search import search_notes
from tools.tasks import add_task, complete_task, list_tasks
from tools.web_search import search_web

logger = logging.getLogger(__name__)


class ToolPlan(TypedDict):
    risk: Literal["read", "write"]
    description: str


class ToolMetadata(TypedDict):
    description: str
    risk: Literal["read", "write"] | dict[str, Literal["read", "write"]]


TOOL_METADATA: dict[str, ToolMetadata] = {
    "notes_search": {
        "description": "Search local text and Markdown notes by keyword.",
        "risk": "read",
    },
    "tasks": {
        "description": "List tasks, add a task, or mark a task complete.",
        "risk": {"list": "read", "add": "write", "complete": "write"},
    },
    "web_search": {
        "description": "Search the web using Tavily.",
        "risk": "read",
    },
}


TOOL_DEFINITIONS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "notes_search",
            "description": f"{TOOL_METADATA['notes_search']['description']} Risk: read-only.",
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
            "description": (
                f"{TOOL_METADATA['tasks']['description']} "
                "Listing is read-only. Adding or completing writes data and requires approval."
            ),
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
            "description": (
                f"{TOOL_METADATA['web_search']['description']} "
                "Read-only. The search query is sent to Tavily."
            ),
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
        return {"risk": "read", "description": f"Search local notes for: {query}"}

    if name == "web_search":
        if set(arguments) != {"query"}:
            raise ValueError("web_search requires only a query.")
        query = _argument_string(arguments, "query")
        return {"risk": "read", "description": f"Search the web for: {query}"}

    if name == "tasks":
        action = _argument_string(arguments, "action")
        if action == "list":
            if set(arguments) != {"action"}:
                raise ValueError("Listing tasks accepts only the action.")
            return {"risk": "read", "description": "List saved tasks"}
        if action == "add":
            if set(arguments) != {"action", "title"}:
                raise ValueError("Adding a task requires action and title.")
            title = _argument_string(arguments, "title")
            return {"risk": "write", "description": f"Add task: {title}"}
        if action == "complete":
            if set(arguments) != {"action", "task_id"}:
                raise ValueError("Completing a task requires action and task_id.")
            task_id = _argument_string(arguments, "task_id")
            return {"risk": "write", "description": f"Mark task {task_id} complete"}
        raise ValueError("Task action must be list, add, or complete.")

    raise ValueError(f"Unknown tool: {name}")


def execute_tool(name: str, arguments: dict[str, Any], allow_risky: bool = False) -> Any:
    plan = prepare_tool(name, arguments)
    if plan["risk"] == "write" and not allow_risky:
        raise ValueError("This tool changes data and must be explicitly approved first.")
    try:
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
    except Timeout:
        return {"error": "Web search timed out after 10 seconds. Please try again."}
    except RequestException:
        return {"error": "Web search could not connect. Check the connection and try again."}
    except (OSError, UnicodeError):
        return {"error": f"The {name} tool could not access its local data."}
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        if name == "web_search":
            return {"error": "Web search returned an unexpected response."}
        return {"error": f"The {name} tool could not use its saved data."}
    except Exception as error:
        logger.warning("Tool %s failed with %s", name, type(error).__name__)
        return {"error": f"The {name} tool failed unexpectedly. Please try again."}


def parse_arguments(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return value
