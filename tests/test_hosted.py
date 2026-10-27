import os
import time
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from core import hosted, llm
from main import app
from memory import manager


@pytest.fixture
def hosted_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    sessions = tmp_path / "hosted-sessions"
    demo = tmp_path / "demo"
    (demo / "memory").mkdir(parents=True)
    (demo / "notes").mkdir()
    (demo / "memory" / "profile.md").write_text("- Sample visitor.\n", encoding="utf-8")
    (demo / "memory" / "preferences.md").write_text("", encoding="utf-8")
    (demo / "memory" / "events.md").write_text("", encoding="utf-8")
    (demo / "notes" / "sample.md").write_text("Sample note.\n", encoding="utf-8")
    (demo / "tasks.json").write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(hosted, "SESSION_ROOT", sessions)
    monkeypatch.setattr(hosted, "DEMO_DATA_DIR", demo)
    monkeypatch.setattr(manager, "DATA_ROOT", tmp_path / "ordinary-data")
    monkeypatch.setattr(hosted, "_RATE_EVENTS", defaultdict(deque))
    monkeypatch.setattr(hosted, "_CALL_DATE", None)
    monkeypatch.setattr(hosted, "_CALL_COUNT", 0)
    monkeypatch.setattr(hosted, "RATE_LIMIT_REQUESTS", 3)
    monkeypatch.setenv("LIFEOS_HOSTED", "true")
    monkeypatch.setenv("LIFEOS_USER", "../must-be-ignored")
    return sessions, demo


def test_hosted_visitors_get_private_demo_copies_and_cookie(
    hosted_environment: tuple[Path, Path],
) -> None:
    sessions, demo = hosted_environment
    with TestClient(app) as client:
        first = client.get("/api/memory")
        session_id = client.cookies[hosted.SESSION_COOKIE]

        assert first.status_code == 200
        assert len(session_id) == 64
        assert "httponly" in first.headers["set-cookie"].lower()
        assert first.json()["files"][0]["lines"][0]["text"] == "- Sample visitor."
        assert client.post(
            "/api/memory/add",
            json={"file": "profile.md", "text": "Private first visitor."},
        ).status_code == 200

        second_client = TestClient(app)
        second = second_client.get("/api/memory")
        second_id = second_client.cookies[hosted.SESSION_COOKIE]

        assert second.status_code == 200
        assert "httponly" in second.headers["set-cookie"].lower()
        assert session_id != second_id
        assert second.json()["files"][0]["lines"] == [
            {"line_number": 1, "text": "- Sample visitor."}
        ]
        assert (sessions / session_id / "memory" / "profile.md").read_text(
            encoding="utf-8"
        ).endswith("- Private first visitor.\n")
        assert (demo / "memory" / "profile.md").read_text(encoding="utf-8") == (
            "- Sample visitor.\n"
        )
        second_client.close()


def test_hosted_sessions_expire_after_three_hours(
    hosted_environment: tuple[Path, Path],
) -> None:
    sessions, _ = hosted_environment
    expired_id = "a" * 64
    expired = sessions / expired_id
    expired.mkdir(parents=True)
    old_time = time.time() - hosted.SESSION_MAX_AGE_SECONDS - 1
    os.utime(expired, (old_time, old_time))

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert not expired.exists()


def test_hosted_rate_limit_is_per_session_and_ip(
    hosted_environment: tuple[Path, Path],
) -> None:
    with TestClient(app) as client:
        statuses = [
            client.post(
                "/api/chat",
                json={"messages": [{"role": "user", "content": "Hi"}]},
            ).status_code
            for _ in range(hosted.RATE_LIMIT_REQUESTS + 1)
        ]

    assert statuses[: hosted.RATE_LIMIT_REQUESTS] == [200] * hosted.RATE_LIMIT_REQUESTS
    assert statuses[-1] == 429


def test_hosted_rate_limit_is_shared_by_ip_across_sessions(hosted_environment) -> None:
    first_client = TestClient(app)
    second_client = TestClient(app)
    first_statuses = [
        first_client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        ).status_code
        for _ in range(2)
    ]
    second_statuses = [
        second_client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        ).status_code
        for _ in range(2)
    ]
    first_client.close()
    second_client.close()

    assert first_statuses == [200] * 2
    assert second_statuses[:1] == [200]
    assert second_statuses[-1] == 429


def test_hosted_message_size_is_limited(hosted_environment) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={
                "messages": [
                    {"role": "user", "content": "x" * (hosted.HOSTED_MAX_MESSAGE_CHARS + 1)}
                ]
            },
        )

    assert response.status_code == 413
    assert "limited" in response.json()["detail"]


def test_hosted_model_tokens_are_capped(
    hosted_environment: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "local-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_MAX_TOKENS", "4096")
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs: object) -> list[object]:
            captured.update(kwargs)
            return [SimpleNamespace(choices=[])]

    class FakeClient:
        def __init__(self, **_kwargs: str) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(llm, "OpenAI", FakeClient)
    list(llm.stream_chat([{"role": "user", "content": "Hi"}]))

    assert captured["max_tokens"] == hosted.HOSTED_MAX_TOKENS


def test_hosted_banner_is_only_rendered_when_enabled(
    hosted_environment: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TestClient(app) as client:
        hosted_response = client.get("/")
    monkeypatch.setenv("LIFEOS_HOSTED", "false")
    with TestClient(app) as client:
        local_response = client.get("/")

    assert "Demo with sample data. Do not enter real personal information." in hosted_response.text
    assert "Demo with sample data. Do not enter real personal information." not in local_response.text


def test_hosted_daily_model_call_cap_returns_resting_message(
    hosted_environment: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIFEOS_DAILY_MODEL_CALL_CAP", "1")
    monkeypatch.setenv("LLM_API_KEY", "local-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)

    class FakeCompletions:
        def create(self, **_kwargs: object) -> list[object]:
            return [SimpleNamespace(choices=[])]

    class FakeClient:
        def __init__(self, **_kwargs: str) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(llm, "OpenAI", FakeClient)
    assert list(llm.stream_chat([{"role": "user", "content": "One"}])) == []

    with pytest.raises(RuntimeError, match="demo is resting"):
        list(llm.stream_chat([{"role": "user", "content": "Two"}]))
