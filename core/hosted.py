import os
import re
import secrets
import shutil
import tempfile
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from memory import manager

DEMO_RESTING = "The demo is resting; please try again later."
HOSTED_MAX_MESSAGE_CHARS = 4000
HOSTED_MAX_TOTAL_MESSAGE_CHARS = 16000
HOSTED_MAX_TOKENS = 1024
RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_SECONDS = 600
SESSION_MAX_AGE_SECONDS = 3 * 60 * 60
SESSION_COOKIE = "lifeos_session"
SESSION_ROOT = Path(tempfile.gettempdir()) / "lifeos-hosted-sessions"
DEMO_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "demo"
_SESSION_ID_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_RATE_EVENTS: dict[tuple[str, str], deque[float]] = defaultdict(deque)
_RATE_LOCK = Lock()
_CALL_DATE: str | None = None
_CALL_COUNT = 0
_CALL_LOCK = Lock()
_SESSION_LOCK = Lock()


def is_hosted() -> bool:
    return os.getenv("LIFEOS_HOSTED", "").strip().casefold() in {"1", "true", "yes", "on"}


def _session_root() -> Path:
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    if SESSION_ROOT.is_symlink():
        raise RuntimeError("The hosted session folder cannot be a linked folder.")
    return SESSION_ROOT.resolve()


def _copy_demo_data(destination: Path) -> None:
    if DEMO_DATA_DIR.is_symlink() or not DEMO_DATA_DIR.is_dir():
        raise RuntimeError("The hosted demo sample data folder is unavailable.")
    for source in DEMO_DATA_DIR.rglob("*"):
        if source.is_symlink():
            raise RuntimeError("Demo sample data cannot contain linked files.")
    shutil.copytree(DEMO_DATA_DIR, destination)


def _cleanup_expired_sessions(root: Path, now: float) -> None:
    for path in root.iterdir():
        if not path.is_dir() or path.is_symlink() or not _SESSION_ID_PATTERN.fullmatch(path.name):
            continue
        if now - path.stat().st_mtime > SESSION_MAX_AGE_SECONDS:
            shutil.rmtree(path)


def _new_session(root: Path) -> tuple[str, Path]:
    while True:
        session_id = secrets.token_hex(32)
        destination = root / session_id
        if destination.exists():
            continue
        _copy_demo_data(destination)
        return session_id, destination


def _get_or_create_session(cookie_header: bytes) -> tuple[str, Path]:
    root = _session_root()
    now = time.time()
    with _SESSION_LOCK:
        _cleanup_expired_sessions(root, now)
        cookie_value = cookie_header.decode("latin-1") if cookie_header else ""
        presented_id = ""
        for item in cookie_value.split(";"):
            name, separator, value = item.strip().partition("=")
            if separator and name == SESSION_COOKIE:
                presented_id = value
                break

        if _SESSION_ID_PATTERN.fullmatch(presented_id):
            existing = root / presented_id
            if existing.is_dir() and not existing.is_symlink():
                os.utime(existing, (now, now))
                return presented_id, existing

        session_id, session_dir = _new_session(root)
        os.utime(session_dir, (now, now))
        return session_id, session_dir


def _client_ip(scope: dict[str, Any]) -> str:
    client = scope.get("client")
    if isinstance(client, (tuple, list)) and client:
        return str(client[0])
    return "unknown"


def _rate_limit_allows(session_id: str, ip_address: str, now: float | None = None) -> bool:
    current_time = time.monotonic() if now is None else now
    keys = (("session", session_id), ("ip", ip_address))
    with _RATE_LOCK:
        for key in keys:
            events = _RATE_EVENTS[key]
            while events and current_time - events[0] >= RATE_LIMIT_SECONDS:
                events.popleft()
            if len(events) >= RATE_LIMIT_REQUESTS:
                return False
        for key in keys:
            _RATE_EVENTS[key].append(current_time)
        return True


def consume_model_call() -> None:
    global _CALL_DATE, _CALL_COUNT
    if not is_hosted():
        return
    raw_cap = os.getenv("LIFEOS_DAILY_MODEL_CALL_CAP", "500")
    try:
        cap = int(raw_cap)
    except ValueError as error:
        raise RuntimeError("LIFEOS_DAILY_MODEL_CALL_CAP must be a positive integer.") from error
    if cap < 1:
        raise RuntimeError("LIFEOS_DAILY_MODEL_CALL_CAP must be a positive integer.")

    today = datetime.now(timezone.utc).date().isoformat()
    with _CALL_LOCK:
        if _CALL_DATE != today:
            _CALL_DATE = today
            _CALL_COUNT = 0
        if _CALL_COUNT >= cap:
            raise RuntimeError(DEMO_RESTING)
        _CALL_COUNT += 1


class HostedSessionMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if not is_hosted() or scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        session_id, session_dir = _get_or_create_session(
            dict(scope.get("headers", [])).get(b"cookie", b"")
        )
        token = manager.set_user_data_dir(session_dir)
        cookie_header = (
            f"{SESSION_COOKIE}={session_id}; Path=/; HttpOnly; SameSite=Strict; Max-Age=10800"
        )
        if scope.get("scheme") == "https":
            cookie_header += "; Secure"

        async def send_with_session(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"set-cookie", cookie_header.encode("ascii")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            path = scope.get("path", "")
            method = scope.get("method", "")
            if method == "POST" and path in {
                "/api/chat",
                "/api/session/end",
                "/api/skills/propose",
                "/api/skills/run",
                "/api/brief/run",
            }:
                if not _rate_limit_allows(session_id, _client_ip(scope)):
                    body = b'{"detail":"Demo request limit reached. Please try again later."}'
                    await send_with_session(
                        {
                            "type": "http.response.start",
                            "status": 429,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode("ascii")),
                                (b"retry-after", b"600"),
                            ],
                        }
                    )
                    await send_with_session({"type": "http.response.body", "body": body})
                    return
            await self.app(scope, receive, send_with_session)
        finally:
            manager.reset_user_data_dir(token)
