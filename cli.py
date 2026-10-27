import re

from openai import OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from core.llm import friendly_provider_error, stream_strict_answer
from core.agent import agent_events, resume_approval
from core.prompts import chat_system_prompt
from memory.manager import add
from memory.proposals import extract_proposals
from skills.manager import load_skill, skill_instructions
from skills.proposals import confirm_proposal, draft_skill, store_proposal


def _new_session() -> list[ChatCompletionMessageParam]:
    return [{"role": "system", "content": chat_system_prompt()}]


def _end_session(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    conversation = [message for message in messages if message.get("role") != "system"]
    if not conversation:
        print("Nothing to save.")
        return _new_session()
    proposals = extract_proposals(conversation, stream_strict_answer)
    if not proposals:
        print("Nothing to save.")
        return _new_session()

    for proposal in proposals:
        print(
            f"\nSuggested {proposal['file']}: {proposal['text']}\n"
            f"Reason: {proposal['reason']}"
        )
        try:
            answer = input("Save this memory? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return _new_session()
        if answer in {"y", "yes"}:
            try:
                if add(proposal["file"], proposal["text"]):
                    print("Saved.")
                else:
                    print("Already saved; skipped duplicate.")
            except ValueError as error:
                print(f"Could not save memory: {error}")
    return _new_session()


def _run_agent(
    messages: list[ChatCompletionMessageParam],
    allowed_tools: set[str] | None = None,
) -> str:
    response = ""
    events = agent_events(messages, allowed_tools=allowed_tools)
    while True:
        approval_id: str | None = None
        approved = False
        for event in events:
            if event["type"] == "answer":
                text = str(event["text"])
                response += text
                print(text, end="", flush=True)
            elif event["type"] == "tool_step":
                print(f"\n{event['label']}", flush=True)
            elif event["type"] == "approval_required":
                print(
                    f"\n\nApproval needed: {event['description']}\n"
                    f"Input: {event['input']}"
                )
                answer = input("Approve? [y/N] ").strip().lower()
                approval_id = str(event["approval_id"])
                approved = answer in {"y", "yes"}
                break
            elif event["type"] == "error":
                print(f"\nLifeOS could not continue: {event['message']}")
                return response
            elif event["type"] == "done":
                print()
                return response

        if approval_id is None:
            print()
            return response
        events = resume_approval(approval_id, approved)


def _save_skill(messages: list[ChatCompletionMessageParam], name: str) -> None:
    content = draft_skill(name, messages, stream_strict_answer)
    if content is None:
        print("Nothing to save.")
        return
    proposal = store_proposal(content)
    print("\nReview this skill draft:\n")
    print(content)
    try:
        answer = input("Save, edit, or reject? [y/e/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nDraft rejected.")
        return
    if answer == "e":
        print("Enter edited Markdown, then type .save on its own line. Type .cancel to reject.")
        lines: list[str] = []
        try:
            while True:
                line = input()
                if line == ".save":
                    break
                if line == ".cancel":
                    print("Draft rejected.")
                    return
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print("\nDraft rejected.")
            return
        content = "\n".join(lines)
        try:
            answer = input("Save this edited skill? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nDraft rejected.")
            return
    if answer not in {"y", "yes"}:
        print("Draft rejected.")
        return
    try:
        saved = confirm_proposal(proposal["proposal_id"], content)
        print(f"Saved skill '{saved['name']}'.")
    except (ValueError, FileExistsError) as error:
        print(f"Could not save this skill: {error}")


def main() -> None:
    messages = _new_session()
    print("LifeOS chat (type /end, /skill <name>, /brief, or /exit)")
    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text == "/exit":
            return
        if text == "/end":
            try:
                messages = _end_session(messages)
            except RuntimeError as error:
                print(f"\nLifeOS could not review this session: {error}")
            except OpenAIError as error:
                print(f"\nLifeOS could not review this session: {friendly_provider_error(error)}")
            continue
        if text == "/brief":
            try:
                from briefs.service import run_morning_brief

                print(run_morning_brief(stream_strict_answer))
            except (RuntimeError, ValueError, OSError, OpenAIError) as error:
                message = (
                    friendly_provider_error(error)
                    if isinstance(error, OpenAIError)
                    else str(error)
                )
                print(f"\nLifeOS could not prepare the brief: {message}")
            continue
        skill_match = re.fullmatch(r"/skill\s+(.+)", text)
        if skill_match:
            try:
                skill = load_skill(skill_match.group(1).strip())
                run_messages = [
                    *messages,
                    {"role": "user", "content": text},
                    {"role": "user", "content": skill_instructions(skill)},
                ]
                answer = _run_agent(
                    run_messages,
                    {tool for step in skill["steps"] for tool in step["tools"]},
                )
                messages.extend(
                    [
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": answer},
                    ]
                )
            except (FileNotFoundError, ValueError) as error:
                print(f"\nLifeOS could not run that skill: {error}")
            except (RuntimeError, OpenAIError) as error:
                message = (
                    friendly_provider_error(error)
                    if isinstance(error, OpenAIError)
                    else str(error)
                )
                print(f"\nLifeOS could not run that skill: {message}")
            continue
        skill_save_match = re.fullmatch(
            r"save this as a skill\s+([A-Za-z0-9-]+)", text, re.IGNORECASE
        )
        if skill_save_match:
            messages.append({"role": "user", "content": text})
            try:
                _save_skill(messages, skill_save_match.group(1))
            except (RuntimeError, OpenAIError) as error:
                message = (
                    friendly_provider_error(error)
                    if isinstance(error, OpenAIError)
                    else str(error)
                )
                print(f"\nLifeOS could not draft the skill: {message}")
            continue
        if not text:
            continue

        messages.append({"role": "user", "content": text})
        print("LifeOS: ", end="", flush=True)
        try:
            answer = _run_agent(messages)
            messages.append({"role": "assistant", "content": answer})
        except RuntimeError as error:
            messages.pop()
            print(f"\nLifeOS could not complete the request: {error}")
        except OpenAIError as error:
            messages.pop()
            print(f"\nLifeOS could not complete the request: {friendly_provider_error(error)}")


if __name__ == "__main__":
    main()
