"""Memory Agent - Manages conversation history and session state."""

from typing import Any, Dict
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from agents.base import BaseAgent
from backend import session_store


class MemoryAgent(BaseAgent):
    """Memory Agent retrieves and manages conversation history."""

    def __init__(self):
        """Initialize the Memory Agent."""
        super().__init__(temperature=0.0)
        # In-memory store for session histories
        self.session_store: Dict[str, ChatMessageHistory] = {}

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get fresh session history for a session, hydrated from persistence."""
        history = ChatMessageHistory()
        persisted = session_store.load_langchain_messages(session_id)
        if persisted:
            history.add_messages(persisted)
        self.session_store[session_id] = history
        return self.session_store[session_id]

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve and update conversation history.

        Args:
            state: The current state with 'session_id' and optionally 'final_response'.

        Returns:
            Updated state with 'chat_history' populated.
        """
        session_id = state.get("session_id", "default")
        self.log(f"Retrieving memory for session: {session_id}")

        # Get the session history
        history = self._get_session_history(session_id)

        # Convert to list format for state
        state["chat_history"] = history.messages

        self.log(f"Retrieved {len(state['chat_history'])} messages from history")
        return state

    def add_exchange(self, session_id: str, query: str, response: str):
        """Manually add a query-response exchange to history.

        Args:
            session_id: The session identifier.
            query: The user query.
            response: The assistant response.
        """
        history = self._get_session_history(session_id)
        history.add_user_message(query)
        history.add_ai_message(response)
        session_store.save_exchange(
            session_id=session_id,
            user_message=query,
            assistant_message=response,
            route="",
            fetched=False,
            articles_count=0,
            fetch_error=None,
        )
        self.log(f"Added exchange to session {session_id}")
