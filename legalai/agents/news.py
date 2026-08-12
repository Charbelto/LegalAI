"""News Agent - Expert on current AI news and developments."""

from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from agents.base import BaseAgent
import config
import utils as Utils


class NewsAgent(BaseAgent):
    """News Agent specializes in current AI-related news and developments."""

    def __init__(self):
        """Initialize the News Agent.

        role="news" binds this agent to its own fine-tuned model under
        GENERATION_PROVIDER=local_peft (Qwen2.5 3B + a NewsQA LoRA adapter).
        """
        super().__init__(temperature=0.3, role="news")
        self.prompt = ChatPromptTemplate.from_template(config.NEWS_PROMPT)

    def _format_history(self, history: list) -> str:
        """Format chat history for the prompt.

        Args:
            history: List of chat messages.

        Returns:
            Formatted history string.
        """
        if not history:
            return "No previous conversation."

        # Take only last 6 messages for context
        recent_history = history[-6:]
        formatted = []
        for msg in recent_history:
            role = "User" if msg.type == "human" else "Assistant"
            formatted.append(f"{role}: {msg.content}")

        return "\n".join(formatted)

    def _format_context(self, docs: list) -> str:
        """Format retrieved documents for prompt context."""
        if not docs:
            return "No relevant documents found."

        parts = []
        for index, doc in enumerate(docs[:5], 1):
            if hasattr(doc, "page_content"):
                content = str(getattr(doc, "page_content", ""))
                metadata = getattr(doc, "metadata", {}) or {}
                source = metadata.get("name") or metadata.get("source") or "Unknown Source"
            elif isinstance(doc, dict):
                content = str(doc.get("page_content") or doc.get("content") or "")
                metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                source = metadata.get("name") or metadata.get("source") or "Unknown Source"
            else:
                content = str(doc)
                source = "Unknown Source"

            parts.append(f"[Document {index} from {source}]:\n{content}\n")

        return "\n".join(parts)

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Answer the query using news expertise.

        Args:
            state: The current state with 'query', 'retrieved_docs', 'chat_history'.

        Returns:
            Updated state with agent output.
        """
        query = state.get("query", "")
        self.log(f"Processing news query: {query[:50]}...")

        # Get context from retrieved documents
        docs = state.get("retrieved_docs", [])
        context = self._format_context(docs)

        # Format chat history
        chat_history = self._format_history(state.get("chat_history", []))

        # Create the chain and invoke
        chain = self.prompt | self.llm

        response = chain.invoke({
            "query": query,
            "context": context,
            "chat_history": chat_history,
            "current_date": Utils.get_current_date(),
        })

        # Store output
        if "agent_outputs" not in state:
            state["agent_outputs"] = {}
        state["agent_outputs"]["news"] = response.content

        # Record tokens
        self._record_tokens(state, "news", response)

        self.log(f"Generated news response ({len(response.content)} chars)")
        return state
