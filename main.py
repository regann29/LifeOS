import json
import logging
from collections.abc import Iterator
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from openai import OpenAIError
from pydantic import BaseModel, Field

from core.llm import stream_chat
from core.prompts import chat_system_prompt
from memory import manager
from memory.confirmation import save_proposals
from memory.proposals import MemoryProposal, extract_proposals

logger = logging.getLogger(__name__)
app = FastAPI(title="LifeOS")


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


class MemoryProposalInput(BaseModel):
    file: str
    text: str = Field(min_length=1, max_length=1000)
    reason: str = Field(min_length=1, max_length=1000)


class MemoryProposalRequest(BaseModel):
    proposals: list[MemoryProposalInput] = Field(max_length=40)


class MemoryLineResponse(BaseModel):
    line_number: int
    text: str


class MemoryFileResponse(BaseModel):
    name: str
    lines: list[MemoryLineResponse]


@app.get("/")
def index() -> FileResponse:
    return FileResponse("web/index.html")


@app.get("/app.js")
def app_script() -> FileResponse:
    return FileResponse("web/app.js", media_type="text/javascript")


@app.get("/styles.css")
def styles() -> FileResponse:
    return FileResponse("web/styles.css", media_type="text/css")


@app.post("/api/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    messages = [
        {"role": "system", "content": chat_system_prompt()},
        *[message.model_dump() for message in request.messages],
    ]

    def response_lines() -> Iterator[str]:
        try:
            for text in stream_chat(messages):
                yield json.dumps({"type": "delta", "text": text}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        except RuntimeError as error:
            yield json.dumps({"type": "error", "message": str(error)}) + "\n"
        except OpenAIError:
            logger.exception("The model provider request failed")
            yield json.dumps(
                {
                    "type": "error",
                    "message": "The model request failed. Check the server logs and your provider settings.",
                }
            ) + "\n"

    return StreamingResponse(
        response_lines(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
    messages = [message.model_dump() for message in request.messages]
    try:
        proposals = extract_proposals(messages, stream_chat)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except OpenAIError as error:
        logger.exception("The model provider failed to suggest memories")
        raise HTTPException(
            status_code=502,
            detail="The model request failed. Check the server logs and provider settings.",
        ) from error
    return {
        "proposals": proposals,
        "message": "Review each suggestion and select the memories to save."
        if proposals
        else "Nothing to save.",
    }


@app.post("/api/memory/confirm")
def confirm_memories(request: MemoryProposalRequest) -> dict[str, int]:
    try:
        proposals: list[MemoryProposal] = [
            {"file": item.file, "text": item.text, "reason": item.reason}
            for item in request.proposals
        ]
        saved, skipped = save_proposals(proposals)
        return {"saved": saved, "skipped_duplicates": skipped}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
