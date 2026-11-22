"""FastAPI entry point: SSE chat endpoint, health checks and the static frontend.

Run locally with `uvicorn main:app --reload` or `python main.py`. On Cloud Run the
container CMD binds uvicorn to the injected $PORT.

Every payload yielded by `src.graph.stream_query` becomes one Server-Sent Events frame:

    event: <payload["type"]>
    data: <json.dumps(payload["data"])>

followed by a blank line. While the graph is busy (LLM calls, retrieval, web search) a
`: ping` comment line is written every SSE_HEARTBEAT_SECONDS so proxies and the Cloud Run
frontend don't close an idle connection. The stream always ends with exactly one `done`
frame, even when the graph raises or finishes without emitting one.
"""

import asyncio
import importlib
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src import graph as graph_module

logger = logging.getLogger("server")

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
INDEX_HTML = STATIC_DIR / "index.html"

MAX_QUERY_CHARS = 4000
MAX_SESSION_ID_CHARS = 128
# Read at request time (module attribute) so tests can shorten it with monkeypatch.
SSE_HEARTBEAT_SECONDS = float(os.environ.get("SSE_HEARTBEAT_SECONDS", "10"))

# Names a retriever module might expose for preloading Chroma + the ONNX embedder.
RETRIEVER_WARM_FUNCTION_NAMES = ("warm_up", "warm", "warm_up_retriever", "warmup", "preload")

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def warm_up_resources() -> bool:
    """Preload heavy resources (Chroma client, ONNX embedding model) before the first request.

    Returns True when a warm function ran. Never raises: a missing or broken retriever
    module must not stop the server from starting; the first query then pays the cost.
    """
    try:
        retriever_module = importlib.import_module("src.rag.retriever")
    except Exception as import_error:
        logger.warning("warm-up skipped, src.rag.retriever not importable: %s", import_error)
        return False

    for function_name in RETRIEVER_WARM_FUNCTION_NAMES:
        warm_function = getattr(retriever_module, function_name, None)
        if not callable(warm_function):
            continue
        try:
            started_at = time.perf_counter()
            warm_function()
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.info("warm-up via retriever.%s finished in %.0f ms", function_name, elapsed_ms)
            return True
        except Exception as warm_error:
            logger.warning("warm-up via retriever.%s failed: %s", function_name, warm_error)
            return False

    logger.info("warm-up skipped, src.rag.retriever exposes no warm function")
    return False


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Warm in a worker thread: model loading is sync and CPU-bound.
    await asyncio.to_thread(warm_up_resources)
    yield


app = FastAPI(title="Multi-agent RAG", lifespan=lifespan)


def describe_validation_error(error: dict) -> dict:
    field_path = ".".join(str(part) for part in error.get("loc", []))
    return {"field": field_path, "message": error.get("msg", "invalid value")}


@app.exception_handler(RequestValidationError)
async def reject_invalid_request(request: Request, validation_error: RequestValidationError) -> JSONResponse:
    # FastAPI's default 422 body echoes the submitted value back; for a 4,000+ char query that
    # is the whole input. Report only which field failed and why.
    problems = [describe_validation_error(error) for error in validation_error.errors()]
    return JSONResponse(status_code=422, content={"detail": problems})
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=1, max_length=MAX_SESSION_ID_CHARS)

    @field_validator("query")
    @classmethod
    def reject_blank_query(cls, query: str) -> str:
        stripped_query = query.strip()
        if not stripped_query:
            raise ValueError("query must not be blank")
        return stripped_query

    @field_validator("session_id")
    @classmethod
    def reject_blank_session_id(cls, session_id: str) -> str:
        stripped_session_id = session_id.strip()
        if not stripped_session_id:
            raise ValueError("session_id must not be blank")
        return stripped_session_id


def format_sse_frame(event_name: str, data: Any) -> str:
    # json.dumps escapes newlines, so the payload always fits on a single `data:` line.
    encoded_data = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event_name}\ndata: {encoded_data}\n\n"


SSE_HEARTBEAT_FRAME = ": ping\n\n"


async def stream_chat_frames(query: str, session_id: str) -> AsyncIterator[str]:
    """Drive the graph and translate its payloads into SSE frames, with heartbeats."""
    started_at = time.perf_counter()
    answer_parts: list[str] = []
    first_token_ms: float | None = None
    done_sent = False

    payload_iterator = graph_module.stream_query(query, session_id).__aiter__()
    pending_next: asyncio.Future | None = None

    def build_fallback_done() -> dict:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        return {"answer": "".join(answer_parts), "latency_ms": elapsed_ms, "ttft_ms": first_token_ms}

    try:
        while True:
            # Keep one __anext__ in flight and wait on it with a timeout. Using
            # asyncio.wait (not wait_for) means a timeout never cancels the graph step;
            # we just emit a heartbeat and keep waiting on the same future.
            if pending_next is None:
                pending_next = asyncio.ensure_future(payload_iterator.__anext__())
            finished, _ = await asyncio.wait({pending_next}, timeout=SSE_HEARTBEAT_SECONDS)
            if not finished:
                yield SSE_HEARTBEAT_FRAME
                continue

            completed_step = pending_next
            pending_next = None
            try:
                payload = completed_step.result()
            except StopAsyncIteration:
                break

            event_name = str(payload.get("type", "trace"))
            event_data = payload.get("data", {})
            if event_name == "token":
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started_at) * 1000
                answer_parts.append(str(event_data.get("text", "")))
            if event_name == "done":
                # The graph may report ttft_ms as None; fill it from what we observed.
                if isinstance(event_data, dict) and event_data.get("ttft_ms") is None:
                    event_data = {**event_data, "ttft_ms": first_token_ms}
                done_sent = True
            yield format_sse_frame(event_name, event_data)
            if done_sent:
                break
    except asyncio.CancelledError:
        # Client disconnected; let Starlette finish the cancellation.
        raise
    except Exception as graph_error:
        logger.exception("graph stream failed")
        error_data = {
            "message": f"{type(graph_error).__name__}: {graph_error}",
            "node": "server",
            "fatal": True,
        }
        yield format_sse_frame("error", error_data)
    finally:
        if pending_next is not None and not pending_next.done():
            pending_next.cancel()
        close_iterator = getattr(payload_iterator, "aclose", None)
        if close_iterator is not None:
            try:
                await close_iterator()
            except Exception:
                logger.debug("graph iterator close failed", exc_info=True)

    if not done_sent:
        yield format_sse_frame("done", build_fallback_done())


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    frame_stream = stream_chat_frames(request.query, request.session_id)
    return StreamingResponse(frame_stream, media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


# Cloud Run's Google frontend intercepts /healthz before it reaches the container
# (found in Assignment 1), so the deployed smoke test uses /health instead.
@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
