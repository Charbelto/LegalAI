"""Pydantic request/response models for the FastAPI backend."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Incoming chat request payload."""

    message: str = Field(..., min_length=1, max_length=8000, description="User message to process")
    session_id: Optional[str] = Field(default=None, description="Conversation session identifier")
    fetch_news: bool = Field(default=True, description="Whether fresh news fetching is enabled")
    num_articles: int = Field(default=5, ge=1, le=10, description="Number of articles to fetch when needed")
    expert_execution_mode: Optional[
        Literal["all", "single", "parallel", "legal_news_parallel", "legal_first", "verify_only", "planner_based", "dag"]
    ] = Field(
        default=None,
        description="Override expert execution mode for this request only",
    )

    seed: Optional[int] = Field(
        default=None,
        description="Sampling seed for this request only; used by the benchmark harness to vary repeats",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message must not be empty")
        return cleaned


class ChatResponse(BaseModel):
    """Final chat response payload returned by the backend."""

    session_id: str
    route: str
    response: str
    abstained: bool = False
    abstained_experts: List[str] = Field(default_factory=list)
    experts_run: int = 0
    expert_abstention_rate: float = 0.0
    truncation_warnings: List[Dict[str, Any]] = Field(default_factory=list)
    seed: Optional[int] = None
    thinking_log: List[Dict[str, Any]]
    fetched: bool = False
    articles_count: int = 0
    fetch_error: Optional[str] = None
    expert_execution_mode: str
    agent_timings_ms: Dict[str, float] = Field(default_factory=dict)
    workflow_elapsed_ms: float = 0.0
    retrieved_ids: List[str] = Field(default_factory=list)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class HealthResponse(BaseModel):
    """Service health and data status payload."""

    status: str
    documents: int
    articles: int
    has_local_db: bool
    last_fetch_time: Optional[str]
    uptime_seconds: float
    ollama_reachable: bool
    generation_provider: str = "ollama"
    chat_model: str
    embedding_model: str


class ClearDataResponse(BaseModel):
    """Response model for data clearing endpoint."""

    ok: bool
    message: str


class RuntimeConfigResponse(BaseModel):
    """Runtime configuration useful for diagnostics and deployment checks."""

    app_version: str
    ollama_base_url: str
    generation_provider: str = "ollama"
    chat_model: str
    embedding_model: str
    max_iterations: int
    expert_execution_mode: str
    serve_static: bool
    compl_ai_enabled: bool = False
    deterministic: bool = True
    llm_seed: int = 42
    num_ctx: int = 8192
    num_predict: int = 1024
    # PEFT pivot. generation_arm is None unless GENERATION_PROVIDER=local_peft;
    # benchmark.py treats a missing value as "this server predates the pivot"
    # and refuses to run an arm against it.
    generation_arm: Optional[str] = None
    local_models: List[Dict[str, Any]] = Field(default_factory=list)
    local_max_input_tokens: int = 7168
    eurlex_live_search_enabled: bool = False


class ReadinessCheck(BaseModel):
    """Single readiness check result."""

    name: str
    ok: bool
    detail: str


class ReadinessResponse(BaseModel):
    """Deployment/readiness status payload."""

    status: str
    checks: List[ReadinessCheck]
    warnings: List[str]


class SessionMessage(BaseModel):
    """Stored chat message for a persisted session."""

    role: str
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


class SessionSummary(BaseModel):
    """Compact session metadata used in session lists."""

    session_id: str
    created_at: str
    updated_at: str
    message_count: int
    last_user_message: Optional[str] = None


class SessionDetail(BaseModel):
    """Full persisted session with message history."""

    session_id: str
    created_at: str
    updated_at: str
    message_count: int
    messages: List[SessionMessage]


class SessionsResponse(BaseModel):
    """List wrapper for persisted sessions."""

    total: int
    sessions: List[SessionSummary]
