import json
from collections.abc import Callable, Iterable, Sequence
from typing import TypedDict

from openai.types.chat import ChatCompletionMessageParam

from memory.manager import MEMORY_FILES, normalize_memory_text


class MemoryProposal(TypedDict):
    file: str
    text: str
    reason: str


_INSTRUCTION = """Review this conversation and suggest only lasting facts or preferences the user clearly stated.
Never infer or guess. Never propose passwords, API keys, card numbers, government IDs, or health, financial, or other highly sensitive details unless the user explicitly asked to save them.
Return ONLY valid JSON: an array of objects, each with exactly "file", "text", and "reason" string fields.
"file" must be profile.md (who the user is), preferences.md (how they like things done), or events.md (dated events).
"text" must be one bullet memory line. Event memories must begin "- YYYY-MM-DD: ".
If there is nothing suitable, return [] without any other text."""

GenerateText = Callable[[Sequence[ChatCompletionMessageParam]], Iterable[str]]


def extract_proposals(
    messages: Sequence[ChatCompletionMessageParam], generate_text: GenerateText
) -> list[MemoryProposal]:
    prompt: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": _INSTRUCTION},
        *[message for message in messages if message.get("role") != "system"],
    ]
    for _ in range(2):
        try:
            raw = "".join(generate_text(prompt))
            decoded = json.loads(raw)
            if not isinstance(decoded, list):
                raise ValueError("Proposal response must be a JSON list.")
            proposals: list[MemoryProposal] = []
            for item in decoded:
                if not isinstance(item, dict) or set(item) != {"file", "text", "reason"}:
                    raise ValueError("Every proposal must contain file, text, and reason.")
                file = item["file"]
                text = item["text"]
                reason = item["reason"]
                if not isinstance(file, str) or file not in MEMORY_FILES:
                    raise ValueError("Proposal contains an unsupported memory file.")
                if not isinstance(text, str) or not isinstance(reason, str) or not reason.strip():
                    raise ValueError("Proposal text and reason must be non-empty strings.")
                proposals.append(
                    {
                        "file": file,
                        "text": normalize_memory_text(file, text),
                        "reason": reason.strip(),
                    }
                )
            return proposals
        except (json.JSONDecodeError, ValueError):
            continue
    return []
