from memory.manager import load_summary


def chat_system_prompt() -> str:
    summary = load_summary()
    if not summary:
        summary = "(No saved memories yet.)"
    return (
        "You are LifeOS, a helpful personal assistant. Use memory only when it helps "
        "answer the user. Never recite memories unprompted. If a memory seems outdated, ask.\n\n"
        f"What you know about the user:\n{summary}"
    )
