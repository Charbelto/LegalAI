"""Core service layer for running the Legal AI workflow via API."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Optional
import gc
import os
import shutil
import time
import uuid

import requests

import config
import embed
import utils as Utils
from auto_fetcher import AutoNewsFetcher, FetchProgress
from backend import session_store
from graph.workflow import create_legal_ai_graph
from query_analyzer import QueryAnalyzer

EventEmitter = Optional[Callable[[str, Dict[str, Any]], None]]


class LegalAIService:
    """Reusable backend service that wraps the existing multi-agent graph."""

    def __init__(self):
        self._graph = None
        self._init_lock = Lock()
        self._storage_lock = Lock()
        self._eu_act_loaded = False
        self._last_fetch_time: Optional[datetime] = None
        self._started_at = datetime.utcnow()

    def initialize(self):
        """Initialize long-lived workflow resources only once."""
        with self._init_lock:
            if self._graph is None:
                self._graph = create_legal_ai_graph()

    def _emit(self, emit: EventEmitter, event_type: str, payload: Dict[str, Any]):
        if emit:
            emit(event_type, payload)

    def _check_ollama(self) -> Dict[str, Any]:
        """Check whether Ollama is reachable and models are available."""
        api_base = config.OLLAMA_BASE_URL.rstrip("/")
        tags_url = f"{api_base}/api/tags"

        try:
            response = requests.get(tags_url, timeout=3)
            response.raise_for_status()
            payload = response.json() if response.content else {}
            models_payload = payload.get("models", []) if isinstance(payload, dict) else []
            available = {
                str(model.get("name", ""))
                for model in models_payload
                if isinstance(model, dict)
            }

            warnings = []
            # Chat model only needs to be pulled locally when generation actually
            # runs on Ollama. Embeddings always run locally regardless of
            # GENERATION_PROVIDER, so that check is unconditional.
            if config.GENERATION_PROVIDER == "ollama" and config.OLLAMA_MODEL not in available:
                warnings.append(f"Chat model '{config.OLLAMA_MODEL}' not found in Ollama tags")
            if config.OLLAMA_EMBEDDING_MODEL not in available:
                warnings.append(
                    f"Embedding model '{config.OLLAMA_EMBEDDING_MODEL}' not found in Ollama tags"
                )

            detail = "Ollama reachable"
            if warnings:
                detail = "Ollama reachable, but one or more configured models are missing"

            return {
                "ok": True,
                "detail": detail,
                "warnings": warnings,
            }
        except Exception as exc:
            return {
                "ok": False,
                "detail": f"Could not reach Ollama at {tags_url}: {exc}",
                "warnings": [
                    "Start Ollama and pull required models before querying the assistant"
                ],
            }

    def _resolved_chat_model(self) -> str:
        """The model actually doing generation right now - not always OLLAMA_MODEL.

        benchmark.py's preflight check and run_meta.json both read this via
        /config and /health; if it always reported OLLAMA_MODEL even while
        GENERATION_PROVIDER=deepseek, every provenance record of a
        DeepSeek-generated run would falsely claim qwen2.5 produced it.

        Under local_peft there is no single model, so this returns a compact
        role=model summary. The structured per-role detail lives in
        get_runtime_config()["local_models"].
        """
        if config.GENERATION_PROVIDER == "deepseek":
            return config.DEEPSEEK_MODEL
        if config.GENERATION_PROVIDER == "local_peft":
            return "; ".join(
                f"{role}={spec['base_model']}"
                for role, spec in config.LOCAL_PEFT_ROLES.items()
            )
        return config.OLLAMA_MODEL

    def _local_model_rows(self) -> list:
        """Per-role local-model provenance, or [] for the other providers."""
        if config.GENERATION_PROVIDER != "local_peft":
            return []
        try:
            import local_models

            return local_models.describe_roles()
        except Exception as exc:  # pragma: no cover - reported, not raised
            print(f"[service] could not describe local models: {exc}")
            return []

    def _generation_arm(self):
        """'peft', 'base', or None when generation is not local_peft.

        benchmark.py refuses to measure a server whose arm disagrees with the
        requested one, so this has to reflect what was actually loaded rather
        than what someone intended.
        """
        if config.GENERATION_PROVIDER != "local_peft":
            return None
        try:
            import local_models

            return local_models.arm_name()
        except Exception:  # pragma: no cover
            return "peft" if config.LOCAL_PEFT_USE_ADAPTERS else "base"

    def _classify_route(self, query: str, analyzer: QueryAnalyzer) -> str:
        route_prompt = (
            "Classify this query as 'legal', 'news', or 'general'. Respond with only one word.\n"
            f"Query: {query}\n"
            "Classification:"
        )
        route_response = analyzer.llm.invoke(route_prompt)
        route = route_response.content.strip().lower()

        if "legal" in route:
            return "legal"
        if "news" in route:
            return "news"
        return "general"

    def _normalize_expert_mode(self, mode: Optional[str]) -> str:
        """Normalize per-request expert execution mode with safe fallback."""
        normalized = str(mode or "").strip().lower()
        if normalized in {"graph", "dag"}:
            return "graph_engineering"
        if normalized in {"all", "single", "parallel", "legal_news_parallel", "legal_first", "verify_only", "planner_based", "graph_engineering"}:
            return normalized
        return config.EXPERT_EXECUTION_MODE

    def ensure_eu_ai_act_loaded(self, emit: EventEmitter = None):
        """Load the EU AI Act into Chroma only when needed."""
        if self._eu_act_loaded:
            return

        with self._storage_lock:
            if self._eu_act_loaded:
                return

            has_db = os.path.exists(Utils.DB_FOLDER)
            doc_count = Utils.get_db_document_count() if has_db else 0

            if has_db and doc_count > 0:
                self._eu_act_loaded = True
                return

            self._emit(emit, "status", {"message": "Loading EU AI Act into vector store..."})
            text = embed.pdf_to_text(Utils.EUROPEAN_ACT_URL)
            if not text.strip():
                raise RuntimeError("Failed to download EU AI Act document")

            embed.embed_text_in_chromadb(
                text,
                document_name="Artificial Intelligence Act",
                document_description="Artificial Intelligence Act",
            )
            self._eu_act_loaded = True
            self._emit(emit, "status", {"message": "EU AI Act loaded."})

    def get_health(self) -> Dict[str, Any]:
        """Return current API/service status."""
        documents = Utils.get_db_document_count()
        articles = Utils.load_articles(Utils.ARTICLES_FILE)
        article_count = len(articles) if isinstance(articles, list) else 0
        ollama = self._check_ollama()
        uptime_seconds = (datetime.utcnow() - self._started_at).total_seconds()

        return {
            "status": "ok",
            "documents": documents,
            "articles": article_count,
            "has_local_db": os.path.exists(Utils.DB_FOLDER),
            "last_fetch_time": self._last_fetch_time.isoformat() if self._last_fetch_time else None,
            "uptime_seconds": round(uptime_seconds, 2),
            "ollama_reachable": bool(ollama.get("ok", False)),
            "generation_provider": config.GENERATION_PROVIDER,
            "chat_model": self._resolved_chat_model(),
            "embedding_model": config.OLLAMA_EMBEDDING_MODEL,
        }

    def get_runtime_config(self) -> Dict[str, Any]:
        """Return runtime configuration details for diagnostics."""
        serve_static = os.getenv("LEGALAI_SERVE_STATIC", "0").lower() in {"1", "true", "yes"}
        return {
            "app_version": "1.1.0",
            "ollama_base_url": config.OLLAMA_BASE_URL,
            "generation_provider": config.GENERATION_PROVIDER,
            "chat_model": self._resolved_chat_model(),
            "embedding_model": config.OLLAMA_EMBEDDING_MODEL,
            "max_iterations": config.MAX_ITERATIONS,
            "expert_execution_mode": config.EXPERT_EXECUTION_MODE,
            "serve_static": serve_static,
            # Experiment-critical flags: the benchmark harness refuses to run
            # against a server with canned COMPL-AI answers enabled.
            "compl_ai_enabled": config.COMPL_AI_ENABLED,
            "deterministic": config.DETERMINISTIC,
            "llm_seed": config.LLM_SEED,
            "num_ctx": config.LLM_NUM_CTX,
            "num_predict": config.LLM_NUM_PREDICT,
            # PEFT-pivot provenance. benchmark.py's preflight aborts on an
            # arm mismatch, so these are load-bearing, not decorative.
            "generation_arm": self._generation_arm(),
            "local_models": self._local_model_rows(),
            "local_max_input_tokens": config.LOCAL_MAX_INPUT_TOKENS,
            "eurlex_live_search_enabled": config.EURLEX_LIVE_SEARCH_ENABLED,
        }

    def get_readiness(self) -> Dict[str, Any]:
        """Return deployment readiness checks with caveat warnings."""
        checks = []
        warnings = []

        ollama = self._check_ollama()
        checks.append(
            {
                "name": "ollama",
                "ok": bool(ollama.get("ok", False)),
                "detail": str(ollama.get("detail", "")),
            }
        )
        warnings.extend(ollama.get("warnings", []))

        has_db = os.path.exists(Utils.DB_FOLDER)
        doc_count = Utils.get_db_document_count() if has_db else 0
        checks.append(
            {
                "name": "vector_store",
                "ok": has_db and doc_count > 0,
                "detail": (
                    f"Vector store ready with {doc_count} document chunks"
                    if has_db and doc_count > 0
                    else "Vector store missing or empty; first legal query will initialize it"
                ),
            }
        )

        frontend_dist = Path("frontend") / "dist"
        serve_static = os.getenv("LEGALAI_SERVE_STATIC", "0").lower() in {"1", "true", "yes"}
        checks.append(
            {
                "name": "frontend_build",
                "ok": (not serve_static) or frontend_dist.exists(),
                "detail": (
                    "Frontend build found"
                    if frontend_dist.exists()
                    else "Frontend build missing (run npm run build in frontend/)"
                ),
            }
        )

        if serve_static and not frontend_dist.exists():
            warnings.append("LEGALAI_SERVE_STATIC is enabled but frontend/dist is missing")

        ready = all(check["ok"] for check in checks)
        status = "ready" if ready else "degraded"

        return {
            "status": status,
            "checks": checks,
            "warnings": warnings,
        }

    def get_sources(self, limit: int = 10) -> Dict[str, Any]:
        """Return source metadata for frontend status panels."""
        safe_limit = max(1, min(limit, 100))
        articles = Utils.load_articles_metadata()
        sorted_articles = sorted(articles, key=lambda x: x.get("fetched_at", ""), reverse=True)

        return {
            "articles": sorted_articles[:safe_limit],
            "total_articles": len(sorted_articles),
            "documents": Utils.get_db_documents(limit=safe_limit),
        }

    def list_sessions(self, limit: int = 20) -> Dict[str, Any]:
        """List persisted chat sessions."""
        return session_store.list_sessions(limit=limit)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a persisted chat session by id."""
        return session_store.get_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete one persisted chat session."""
        return session_store.delete_session(session_id)

    def clear_sessions(self) -> Dict[str, Any]:
        """Clear all persisted chat sessions."""
        return session_store.clear_sessions()

    def _delete_with_retries(
        self,
        path: str,
        is_dir: bool,
        retries: int = 6,
        delay: float = 0.35,
    ) -> tuple[bool, str]:
        """Delete file/directory with retries for transient Windows locks."""
        last_error = ""

        for _ in range(retries):
            try:
                if is_dir:
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                return True, ""
            except FileNotFoundError:
                return True, ""
            except PermissionError as exc:
                last_error = str(exc)
                gc.collect()
                time.sleep(delay)
            except Exception as exc:  # pragma: no cover - defensive fallback
                return False, str(exc)

        return False, last_error or "Unknown delete error"

    def clear_all_data(self) -> tuple[bool, str]:
        """Clear local article and vector-store data."""
        errors = []

        with self._storage_lock:
            self._graph = None
            gc.collect()

            if os.path.exists(Utils.ARTICLES_FILE):
                ok, err = self._delete_with_retries(Utils.ARTICLES_FILE, is_dir=False, retries=3)
                if not ok:
                    errors.append(f"articles file: {err}")

            if os.path.exists(Utils.ARTICLES_FOLDER):
                ok, err = self._delete_with_retries(Utils.ARTICLES_FOLDER, is_dir=True, retries=4)
                if not ok:
                    errors.append(f"articles folder: {err}")

            if os.path.exists(Utils.DB_FOLDER):
                ok, err = self._delete_with_retries(Utils.DB_FOLDER, is_dir=True, retries=8, delay=0.5)
                if not ok:
                    errors.append(f"vector store: {err}")

            self._eu_act_loaded = False
            self._last_fetch_time = None

        if errors:
            return False, "Could not fully clear data. " + " | ".join(errors)

        return True, "All data cleared."

    def process_query(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        fetch_news: bool = True,
        num_articles: int = 5,
        expert_execution_mode: Optional[str] = None,
        emit: EventEmitter = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a complete query workflow and return the final response payload."""
        self.initialize()

        # Per-request seed: lets the benchmark harness vary the sampling seed per
        # repeat so that repeats carry generation variance (see agents/base.py).
        if seed is not None:
            import agents.base as agent_base

            agent_base.set_runtime_seed(int(seed))
            if config.DETERMINISTIC:
                print(
                    "[service] WARNING seed supplied while LEGALAI_DETERMINISTIC=1; "
                    "decoding is greedy so repeats will be identical. Set "
                    "LEGALAI_DETERMINISTIC=0 for variance-bearing repeats."
                )
        self.ensure_eu_ai_act_loaded(emit=emit)

        safe_num_articles = max(1, min(num_articles, 10))
        effective_expert_mode = self._normalize_expert_mode(expert_execution_mode)

        sid = session_id or f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        persisted_history = session_store.load_langchain_messages(sid, max_messages=30)

        # Check if it matches a COMPL-AI interactive flow.
        # DEMO ONLY: these canned answers bypass the whole multi-agent workflow and
        # report synthetic telemetry (1.0 ms), so they are disabled unless
        # LEGALAI_ENABLE_COMPL_AI=1. Benchmark runs must never enable them.
        compl_response = None
        if config.COMPL_AI_ENABLED:
            import compl_ai
            compl_response = compl_ai.get_compl_ai_response(user_query)
        if compl_response:
            self._emit(emit, "status", {"message": "COMPL-AI compliance flow triggered..."})
            session_store.save_exchange(
                session_id=sid,
                user_message=user_query,
                assistant_message=compl_response,
                route="legal",
                fetched=False,
                articles_count=0,
                fetch_error=None,
            )
            return {
                "session_id": sid,
                "route": "legal",
                "response": compl_response,
                "thinking_log": [{"step": "compl_ai", "details": "COMPL-AI compliance flow diagnostic output"}],
                "fetched": False,
                "articles_count": 0,
                "fetch_error": None,
                "expert_execution_mode": effective_expert_mode,
                "agent_timings_ms": {"compl_ai": 1.0},
                "workflow_elapsed_ms": 1.0,
            }

        analyzer = QueryAnalyzer()
        self._emit(emit, "status", {"message": "Analyzing query..."})
        route = self._classify_route(user_query, analyzer)
        self._emit(emit, "route", {"route": route})

        fetch_result: Dict[str, Any] = {
            "fetched": False,
            "articles_count": 0,
            "error": None,
        }

        should_fetch = fetch_news and analyzer.should_fetch_news(user_query, route)

        if should_fetch:
            self._emit(emit, "status", {"message": "Fetching latest sources..."})
            fetcher = AutoNewsFetcher(num_articles=safe_num_articles)

            def progress_callback(progress: FetchProgress):
                self._emit(
                    emit,
                    "fetch_progress",
                    {
                        "step": progress.step,
                        "total_steps": progress.total_steps,
                        "current_action": progress.current_action,
                        "current_article_url": progress.current_article_url,
                        "current_article_title": progress.current_article_title,
                        "fetched_articles": progress.fetched_articles[-3:],
                    },
                )

            with self._storage_lock:
                fetch_result = fetcher.fetch_with_progress(
                    user_query,
                    route,
                    progress_callback=progress_callback,
                    clear_existing=True,
                )

            if fetch_result.get("fetched"):
                self._last_fetch_time = datetime.now()
                self._emit(
                    emit,
                    "status",
                    {
                        "message": (
                            f"Fetched {fetch_result.get('articles_count', 0)} article(s) and updated the vector store."
                        )
                    },
                )
            elif fetch_result.get("error"):
                self._emit(emit, "status", {"message": f"Fetch issue: {fetch_result['error']}"})
        else:
            self._emit(emit, "status", {"message": "Using existing knowledge base."})

        self._emit(emit, "status", {"message": "Running multi-agent workflow..."})
        self._emit(emit, "status", {"message": f"Expert mode: {effective_expert_mode.upper()}"})

        initial_state = {
            "query": user_query,
            "session_id": sid,
            "thinking_log": [],
            "chat_history": persisted_history,
            "retrieved_docs": [],
            "agent_outputs": {},
            "agent_timings": {},
            "draft_response": "",
            "final_response": "",
            "validation_result": {},
            "iteration_count": 0,
            "error_message": "",
            "route": "",
            "expert_execution_mode": effective_expert_mode,
            "fetched_sources": [],
            "validation_issues": "",
        }

        if self._graph is None:  # pragma: no cover - defensive check
            raise RuntimeError("Workflow graph not initialized")

        result = None
        streamed_graph_steps = 0
        workflow_started_at = time.perf_counter()

        # No lock here: the compiled graph is stateless per invoke/stream call
        # (no checkpointer), so concurrent requests each get their own
        # initial_state and don't interact through the graph itself. This used
        # to be wrapped in a single shared Lock, which meant the server
        # serialized every request end-to-end regardless of client
        # concurrency - the actual reason concurrent benchmark runs never
        # overlapped. The one piece of real shared mutable state (each agent's
        # lazily-cached LLM client, keyed on a process-wide seed generation
        # counter) is harmless to race for GENERATION_PROVIDER=deepseek, since
        # DeepSeek's client never depends on the seed value (see
        # agents/base.py build_chat_llm) - it does NOT make concurrent Ollama
        # runs with per-request seeds safe.
        try:
            for streamed_state in self._graph.stream(initial_state, stream_mode="values"):
                result = streamed_state
                thinking_log = streamed_state.get("thinking_log", [])
                if len(thinking_log) > streamed_graph_steps:
                    for step in thinking_log[streamed_graph_steps:]:
                        self._emit(
                            emit,
                            "thinking",
                            {
                                "step": step.get("step", "unknown"),
                                "details": step.get("details", ""),
                                "elapsed_ms": step.get("elapsed_ms"),
                            },
                        )
                    streamed_graph_steps = len(thinking_log)
        except Exception:
            result = self._graph.invoke(initial_state)

        if result is None:
            result = self._graph.invoke(initial_state)

        workflow_elapsed_ms = round((time.perf_counter() - workflow_started_at) * 1000, 2)
        result_timings = result.get("agent_timings", {}) if isinstance(result, dict) else {}
        if not isinstance(result_timings, dict):
            result_timings = {}

        response_text = result.get("final_response", "No response generated.")

        session_store.save_exchange(
            session_id=sid,
            user_message=user_query,
            assistant_message=response_text,
            route=route,
            fetched=bool(fetch_result.get("fetched", False)),
            articles_count=int(fetch_result.get("articles_count", 0) or 0),
            fetch_error=fetch_result.get("error"),
        )

        # Extract retrieved document IDs
        retrieved_docs = result.get("retrieved_docs", []) or []
        retrieved_ids = []
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                meta = doc.get("metadata", {}) or {}
                doc_id = meta.get("id") or doc.get("id")
                if doc_id:
                    retrieved_ids.append(str(doc_id))
            elif hasattr(doc, "metadata") and doc.metadata and "id" in doc.metadata:
                retrieved_ids.append(str(doc.metadata["id"]))
            elif hasattr(doc, "id") and doc.id:
                retrieved_ids.append(str(doc.id))

        # Sum up token counts
        agent_tokens = result.get("agent_tokens", {}) or {}
        prompt_tokens = 0
        completion_tokens = 0
        if isinstance(agent_tokens, dict):
            for agent_name, token_info in agent_tokens.items():
                if isinstance(token_info, dict):
                    prompt_tokens += token_info.get("prompt", 0) or 0
                    completion_tokens += token_info.get("completion", 0) or 0

        return {
            "session_id": sid,
            "route": route,
            "response": response_text,
            "abstained": bool(result.get("abstained", False)),
            "abstained_experts": list(result.get("abstained_experts", []) or []),
            "experts_run": int(result.get("experts_run", 0) or 0),
            "expert_abstention_rate": float(result.get("expert_abstention_rate", 0.0) or 0.0),
            "truncation_warnings": list(result.get("truncation_warnings", []) or []),
            # The seed this specific request was called with, not the racy
            # process-wide "current" seed (backend/service.py used to read that
            # via agent_base.get_runtime_seed(), which under concurrent
            # requests can reflect a DIFFERENT in-flight request's seed).
            "seed": seed,
            "thinking_log": result.get("thinking_log", []),
            "fetched": bool(fetch_result.get("fetched", False)),
            "articles_count": int(fetch_result.get("articles_count", 0) or 0),
            "fetch_error": fetch_result.get("error"),
            "expert_execution_mode": effective_expert_mode,
            "agent_timings_ms": result_timings,
            "workflow_elapsed_ms": workflow_elapsed_ms,
            "retrieved_ids": retrieved_ids,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }


service = LegalAIService()
