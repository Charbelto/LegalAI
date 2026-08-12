"""FastAPI application entrypoint for the Legal AI backend."""

from __future__ import annotations

from pathlib import Path
from queue import Queue
from typing import Any, Dict
import json
import os
import threading

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from backend.models import (
    ChatRequest,
    ChatResponse,
    ClearDataResponse,
    HealthResponse,
    ReadinessResponse,
    RuntimeConfigResponse,
    SessionDetail,
    SessionsResponse,
)
from backend.service import service


def _sse_payload(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _resolve_allowed_origins() -> list[str]:
    configured = os.getenv("LEGALAI_ALLOWED_ORIGINS", "").strip()
    if not configured:
        return [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    if configured == "*":
        return ["*"]
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


app = FastAPI(
    title="Legal AI API",
    description="Production backend for the Legal AI multi-agent assistant",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_resolve_allowed_origins(),
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
SERVE_STATIC = os.getenv("LEGALAI_SERVE_STATIC", "0").lower() in {"1", "true", "yes"}


@app.get("/")
def root():
    if SERVE_STATIC and FRONTEND_DIST.exists():
        return FileResponse(FRONTEND_DIST / "index.html")
    return {"name": "Legal AI API", "status": "ok"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(**service.get_health())


@app.get("/runtime", response_model=RuntimeConfigResponse)
def runtime_config() -> RuntimeConfigResponse:
    return RuntimeConfigResponse(**service.get_runtime_config())


@app.get("/readiness", response_model=ReadinessResponse)
def readiness() -> ReadinessResponse:
    return ReadinessResponse(**service.get_readiness())


@app.get("/sources")
def sources(limit: int = Query(default=10, ge=1, le=100)):
    return service.get_sources(limit=limit)


@app.get("/sessions", response_model=SessionsResponse)
def list_sessions(limit: int = Query(default=20, ge=1, le=100)) -> SessionsResponse:
    payload = service.list_sessions(limit=limit)
    return SessionsResponse(**payload)


@app.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session(session_id: str) -> SessionDetail:
    payload = service.get_session(session_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionDetail(**payload)


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    ok = service.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True, "message": f"Session '{session_id}' deleted."}


@app.delete("/sessions")
def clear_sessions():
    result = service.clear_sessions()
    return {
        "ok": result.get("failed", 0) == 0,
        "message": (
            f"Deleted {result.get('deleted', 0)} sessions"
            if result.get("failed", 0) == 0
            else f"Deleted {result.get('deleted', 0)} sessions, failed to delete {result.get('failed', 0)}"
        ),
        "deleted": result.get("deleted", 0),
        "failed": result.get("failed", 0),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        payload = service.process_query(
            user_query=request.message,
            session_id=request.session_id,
            fetch_news=request.fetch_news,
            num_articles=request.num_articles,
            expert_execution_mode=request.expert_execution_mode,
            seed=request.seed,
        )
        return ChatResponse(**payload)
    except Exception as exc:
        # str(exc) alone previously left every 500 in the benchmark's JSONL and
        # this console indistinguishable ("Internal Server Error") - print the
        # real traceback server-side so a failure is actually diagnosable.
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    sentinel = object()

    def event_generator():
        event_queue: Queue[Any] = Queue()

        def emit(event_type: str, payload: Dict[str, Any]):
            event = {"type": event_type}
            event.update(payload)
            event_queue.put(event)

        def worker():
            try:
                result = service.process_query(
                    user_query=request.message,
                    session_id=request.session_id,
                    fetch_news=request.fetch_news,
                    num_articles=request.num_articles,
                    expert_execution_mode=request.expert_execution_mode,
                    emit=emit,
                    seed=request.seed,
                )
                emit("final", result)
            except Exception as exc:
                emit("error", {"message": str(exc)})
            finally:
                event_queue.put(sentinel)

        threading.Thread(target=worker, daemon=True).start()

        yield _sse_payload({"type": "status", "message": "Request received"})
        while True:
            item = event_queue.get()
            if item is sentinel:
                break
            yield _sse_payload(item)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@app.post("/admin/clear", response_model=ClearDataResponse)
def clear_data() -> ClearDataResponse:
    ok, message = service.clear_all_data()
    if ok:
        return ClearDataResponse(ok=True, message=message)
    return ClearDataResponse(ok=False, message=message)


if SERVE_STATIC and FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        if full_path.startswith(("api", "docs", "redoc", "openapi")):
            raise HTTPException(status_code=404, detail="Not found")

        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.exists() and candidate.is_file():
            return FileResponse(candidate)

        return FileResponse(FRONTEND_DIST / "index.html")
