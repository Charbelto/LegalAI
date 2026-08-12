"""Legal Agent - Expert on EU AI Act and AI regulations."""

from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from agents.base import BaseAgent
import config
import utils as Utils


class LegalAgent(BaseAgent):
    """Legal Agent specializes in EU AI Act and AI regulation questions."""

    def __init__(self):
        """Initialize the Legal Agent.

        role="legal" is what binds this agent to its own fine-tuned model under
        GENERATION_PROVIDER=local_peft (Llama 3.2 3B + a LegalBench LoRA
        adapter). Other providers ignore it and hand back the single shared
        model, so this stays backwards compatible.
        """
        super().__init__(temperature=0.2, role="legal")
        self.prompt = ChatPromptTemplate.from_template(config.LEGAL_PROMPT)

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
        """Answer the query using legal expertise.

        Args:
            state: The current state with 'query', 'retrieved_docs', 'chat_history'.

        Returns:
            Updated state with agent output.
        """
        query = state.get("query", "")
        self.log(f"Processing legal query: {query[:50]}...")

        # Get context from retrieved documents
        docs = list(state.get("retrieved_docs", []) or [])

        # Live official-source search. Disabled by default and forced off by
        # benchmark.py - see eurlex_search.py for why a time-varying context
        # cannot sit inside a controlled run.
        #
        # _format_context only renders the first 5 documents, so appending live
        # results to an already-full static list would silently discard them.
        # The static corpus is trimmed to make room instead, keeping the
        # highest-ranked Act chunks and the live sources both visible.
        if config.EURLEX_LIVE_SEARCH_ENABLED:
            import eurlex_search

            live_docs = eurlex_search.search_recent_legal_sources(query)
            if live_docs:
                keep_static = max(1, 5 - len(live_docs))
                self.log(
                    f"Added {len(live_docs)} live EU source(s); keeping top "
                    f"{keep_static} static chunk(s)"
                )
                docs = docs[:keep_static] + live_docs

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
        state["agent_outputs"]["legal"] = response.content

        # Record tokens
        self._record_tokens(state, "legal", response)

        self.log(f"Generated legal response ({len(response.content)} chars)")
        return state
