"""Persistent session storage for Legal AI chat histories."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import re
from threading import Lock
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

SESSIONS_DIR = Path("sessions")

_STORE_LOCK = Lock()


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _safe_session_id(session_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id).strip("_")
    return cleaned[:120] or "default"


def _session_path(session_id: str) -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SESSIONS_DIR / f"{_safe_session_id(session_id)}.json"


def _base_payload(session_id: str) -> Dict[str, Any]:
    timestamp = _now_iso()
    return {
        "session_id": session_id,
        "created_at": timestamp,
        "updated_at": timestamp,
        "messages": [],
    }


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}

    return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def list_sessions(limit: int = 20) -> Dict[str, Any]:
    """List stored sessions sorted by newest update timestamp."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    for file in SESSIONS_DIR.glob("*.json"):
        payload = _read_json(file)
        if not payload:
            continue

        messages = payload.get("messages", [])
        last_user = next(
            (
                msg.get("content", "")
                for msg in reversed(messages)
                if isinstance(msg, dict) and msg.get("role") == "user"
            ),
            "",
        )

        items.append(
            {
                "session_id": payload.get("session_id", file.stem),
                "created_at": payload.get("created_at", ""),
                "updated_at": payload.get("updated_at", ""),
                "message_count": len(messages),
                "last_user_message": (last_user[:140] + "...") if len(last_user) > 140 else last_user,
            }
        )

    items.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    sliced = items[: max(1, min(limit, 100))]
    return {"total": len(items), "sessions": sliced}


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Load a session payload by session id."""
    path = _session_path(session_id)
    payload = _read_json(path)
    if not payload:
        return None

    messages = payload.get("messages", [])
    return {
        "session_id": payload.get("session_id", session_id),
        "created_at": payload.get("created_at", ""),
        "updated_at": payload.get("updated_at", ""),
        "message_count": len(messages),
        "messages": messages,
    }


def delete_session(session_id: str) -> bool:
    """Delete a single session file if present."""
    path = _session_path(session_id)
    if not path.exists():
        return False

    try:
        path.unlink()
    except Exception:
        return False

    return True


def clear_sessions() -> Dict[str, Any]:
    """Delete all persisted sessions."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    deleted = 0
    failed = 0

    for file in SESSIONS_DIR.glob("*.json"):
        try:
            file.unlink()
            deleted += 1
        except Exception:
            failed += 1

    return {"deleted": deleted, "failed": failed}


def load_langchain_messages(session_id: str, max_messages: int = 30) -> List[BaseMessage]:
    """Load persisted messages and convert them to LangChain message objects."""
    payload = get_session(session_id)
    if not payload:
        return []

    result: List[BaseMessage] = []
    for item in payload.get("messages", [])[-max_messages:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role", "")
        content = str(item.get("content", "")).strip()
        if not content:
            continue

        if role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))

    return result


def save_exchange(
    session_id: str,
    user_message: str,
    assistant_message: str,
    route: str,
    fetched: bool,
    articles_count: int,
    fetch_error: Optional[str],
) -> None:
    """Persist a single user/assistant exchange."""
    if not user_message.strip() or not assistant_message.strip():
        return

    with _STORE_LOCK:
        path = _session_path(session_id)
        payload = _read_json(path) or _base_payload(session_id)
        payload["session_id"] = session_id

        messages = payload.get("messages")
        if not isinstance(messages, list):
            messages = []

        # Avoid duplicate writes if the same exchange is saved twice.
        if len(messages) >= 2:
            prev_user = messages[-2] if isinstance(messages[-2], dict) else {}
            prev_assistant = messages[-1] if isinstance(messages[-1], dict) else {}
            if (
                prev_user.get("role") == "user"
                and prev_assistant.get("role") == "assistant"
                and prev_user.get("content", "") == user_message
                and prev_assistant.get("content", "") == assistant_message
            ):
                payload["updated_at"] = _now_iso()
                _write_json(path, payload)
                return

        timestamp = _now_iso()
        messages.append(
            {
                "role": "user",
                "content": user_message,
                "timestamp": timestamp,
                "metadata": {},
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": assistant_message,
                "timestamp": timestamp,
                "metadata": {
                    "route": route,
                    "fetched": bool(fetched),
                    "articles_count": int(articles_count or 0),
                    "fetch_error": fetch_error,
                },
            }
        )

        payload["messages"] = messages
        payload.setdefault("created_at", timestamp)
        payload["updated_at"] = timestamp

        _write_json(path, payload)
