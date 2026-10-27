import json
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from openai import OpenAIError
from pydantic import BaseModel, ConfigDict, Field

from core.llm import friendly_provider_error, stream_strict_answer, validate_llm_configuration
from core.hosted import HOSTED_MAX_MESSAGE_CHARS, HOSTED_MAX_TOTAL_MESSAGE_CHARS, HostedSessionMiddleware, is_hosted
from core.agent import agent_events, resume_approval
from core.prompts import chat_system_prompt
from briefs.manager import read_brief
from briefs.scheduler import create_scheduler
from briefs.service import run_morning_brief
from skills.manager import Skill, list_skills, load_skill, skill_instructions, update_skill, delete_skill, validate_name
from skills.proposals import confirm_proposal, draft_skill, store_proposal
from memory import manager
from memory.proposals import MemoryProposal, extract_proposals
from memory.proposal_store import ProposalSelection, confirm_proposals, store_proposals
from tools.activity import list_activity

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    configuration_error = validate_llm_configuration()
    if configuration_error:
        logger.warning("%s", configuration_error)
    scheduler = create_scheduler()
    if scheduler is not None:
        scheduler.start()
    try:
        yield
    finally:
        if scheduler is not None:
            scheduler.shutdown(wait=False)


app = FastAPI(title="LifeOS", lifespan=lifespan)
app.add_middleware(HostedSessionMiddleware)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)


class MemoryChange(BaseModel):
    file: str
    text: str = Field(min_length=1, max_length=1000)


class MemoryLineChange(MemoryChange):
    line_number: int = Field(ge=1)


class MemoryLineReference(BaseModel):
    file: str
    line_number: int = Field(ge=1)


class MemoryProposalSelectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(min_length=32, max_length=32, pattern=r"^[a-f0-9]{32}$")
    text: str | None = Field(default=None, min_length=1, max_length=1000)

class MemoryProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposals: list[MemoryProposalSelectionInput] = Field(max_length=40)


class SkillDraftRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)


class SkillRunRequest(SkillDraftRequest):
    pass


class SkillProposalConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(min_length=32, max_length=32, pattern=r"^[a-f0-9]{32}$")
    content: str | None = Field(default=None, min_length=1, max_length=12000)


class SkillContentRequest(BaseModel):
    content: str = Field(min_length=1, max_length=12000)


def _validate_hosted_messages(messages: list[ChatMessage]) -> None:
    if not is_hosted():
        return
    content_lengths = [len(message.content) for message in messages]
    if any(length > HOSTED_MAX_MESSAGE_CHARS for length in content_lengths):
        raise HTTPException(
            status_code=413,
            detail=f"Demo messages are limited to {HOSTED_MAX_MESSAGE_CHARS} characters each.",
        )
    if sum(content_lengths) > HOSTED_MAX_TOTAL_MESSAGE_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Demo conversations are limited to {HOSTED_MAX_TOTAL_MESSAGE_CHARS} characters.",
        )


def _agent_stream_response(
    messages: list[dict[str, object]],
    allowed_tools: set[str] | None = None,
) -> StreamingResponse:
    def response_lines() -> Iterator[str]:
        try:
            for event in agent_events(messages, allowed_tools=allowed_tools):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except RuntimeError as error:
            yield json.dumps({"type": "error", "message": str(error)}) + "\n"
        except (OSError, ValueError) as error:
            logger.exception("A local tool or activity log operation failed")
            yield json.dumps({"type": "error", "message": str(error)}) + "\n"
        except OpenAIError as error:
            yield json.dumps({"type": "error", "message": friendly_provider_error(error)}) + "\n"

    return StreamingResponse(
        response_lines(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ApprovalDecision(BaseModel):
    approval_id: str = Field(min_length=32, max_length=64, pattern=r"^[a-f0-9]+$")
    approved: bool


class MemoryLineResponse(BaseModel):
    line_number: int
    text: str


class MemoryFileResponse(BaseModel):
    name: str
    lines: list[MemoryLineResponse]


@app.get("/")
def index() -> HTMLResponse:
    html_path = Path(__file__).resolve().parent / "web" / "index.html"
    html = html_path.read_text(encoding="utf-8")
    hosted_banner = (
        '<aside class="demo-banner" role="alert">'
        "Demo with sample data. Do not enter real personal information."
        "</aside>"
    )
    if is_hosted():
        html = html.replace("<!-- HOSTED_DEMO_BANNER -->", hosted_banner)
    else:
        html = html.replace("<!-- HOSTED_DEMO_BANNER -->", "")
    return HTMLResponse(html)


def _validate_hosted_message_size(request: ChatRequest) -> None:
    _validate_hosted_messages(request.messages)


@app.get("/app.js")
def app_script() -> FileResponse:
    return FileResponse("web/app.js", media_type="text/javascript")


@app.get("/styles.css")
def styles() -> FileResponse:
    return FileResponse("web/styles.css", media_type="text/css")


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    _validate_hosted_message_size(request)
    messages = [
        {"role": "system", "content": chat_system_prompt()},
        *[message.model_dump() for message in request.messages],
    ]

    return _agent_stream_response(messages)


@app.post("/api/approval")
def decide_approval(decision: ApprovalDecision) -> StreamingResponse:
    def response_lines() -> Iterator[str]:
        try:
            for event in resume_approval(decision.approval_id, decision.approved):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except RuntimeError as error:
            yield json.dumps({"type": "error", "message": str(error)}) + "\n"
        except (OSError, ValueError) as error:
            logger.exception("A local tool or activity log operation failed after approval")
            yield json.dumps({"type": "error", "message": str(error)}) + "\n"
        except OpenAIError as error:
            yield json.dumps({"type": "error", "message": friendly_provider_error(error)}) + "\n"

    return StreamingResponse(
        response_lines(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/activity")
def get_activity() -> dict[str, list[dict[str, object]]]:
    try:
        return {"entries": list_activity()}
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/api/memory")
def get_memory() -> dict[str, list[MemoryFileResponse]]:
    try:
        files = [
            {
                "name": file,
                "lines": [
                    {"line_number": index, "text": text}
                    for index, text in enumerate(manager.read(file), start=1)
                ],
            }
            for file in manager.list_files()
        ]
        return {"files": files}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/memory/add")
def add_memory(change: MemoryChange) -> dict[str, bool]:
    try:
        return {"added": manager.add(change.file, change.text)}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.put("/api/memory/update")
def update_memory(change: MemoryLineChange) -> dict[str, bool]:
    try:
        manager.update(change.file, change.line_number, change.text)
        return {"updated": True}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except IndexError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.delete("/api/memory/delete")
def delete_memory(change: MemoryLineReference) -> dict[str, bool]:
    try:
        manager.delete(change.file, change.line_number)
        return {"deleted": True}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except IndexError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/api/session/end")
def end_session(request: ChatRequest) -> dict[str, object]:
    _validate_hosted_message_size(request)
    messages = [message.model_dump() for message in request.messages]
    try:
        proposals = extract_proposals(messages, stream_strict_answer)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except OpenAIError as error:
        raise HTTPException(
            status_code=502,
            detail=friendly_provider_error(error),
        ) from error
    stored_proposals = store_proposals(proposals)
    return {
        "proposals": stored_proposals,
        "message": "Review each suggestion and select the memories to save."
        if proposals
        else "Nothing to save.",
    }


@app.post("/api/memory/confirm")
def confirm_memories(request: MemoryProposalRequest) -> dict[str, int]:
    try:
        selections: list[ProposalSelection] = [
            {"proposal_id": item.proposal_id, "text": item.text}
            for item in request.proposals
        ]
        saved, skipped = confirm_proposals(selections)
        return {"saved": saved, "skipped_duplicates": skipped}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
