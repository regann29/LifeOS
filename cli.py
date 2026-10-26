from openai import OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from core.llm import stream_chat
from core.agent import agent_events, resume_approval
from core.prompts import chat_system_prompt
from memory.manager import add
from memory.proposals import extract_proposals


def _new_session() -> list[ChatCompletionMessageParam]:
    return [{"role": "system", "content": chat_system_prompt()}]


def _end_session(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    conversation = [message for message in messages if message.get("role") != "system"]
    if not conversation:
        print("Nothing to save.")
        return _new_session()
    proposals = extract_proposals(conversation, stream_chat)
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


def _run_agent(messages: list[ChatCompletionMessageParam]) -> str:
    response = ""
    events = agent_events(messages)
    while True:
        approval_id: str | None = None
        approved = False
        for event in events:
            if event["type"] == "delta":
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


def main() -> None:
    messages = _new_session()
    print("LifeOS chat (type /end to review memories, /exit to quit)")
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
            except (RuntimeError, OpenAIError) as error:
                print(f"\nLifeOS could not review this session: {error}")
            continue
        if not text:
            continue

        messages.append({"role": "user", "content": text})
        print("LifeOS: ", end="", flush=True)
        try:
            answer = _run_agent(messages)
            messages.append({"role": "assistant", "content": answer})
        except (RuntimeError, OpenAIError) as error:
            messages.pop()
            print(f"\nLifeOS could not complete the request: {error}")


if __name__ == "__main__":
    main()
