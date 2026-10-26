import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAIError

from core.llm import friendly_provider_error, stream_chat


def main() -> int:
    try:
        events = stream_chat(
            [{"role": "user", "content": "Reply with a short greeting."}]
        )
        answer = "".join(
            event["text"] for event in events if event["type"] == "answer"
        )
    except RuntimeError as error:
        print(str(error))
        return 1
    except OpenAIError as error:
        print(friendly_provider_error(error))
        return 1
    except Exception:
        print("The model check failed. Check your provider settings and try again.")
        return 1

    if not answer.strip():
        print("The model connection opened, but no answer text was returned.")
        return 1
    print(f"Model connection succeeded: {answer.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
