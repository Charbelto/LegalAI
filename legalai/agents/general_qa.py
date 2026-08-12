"""General QA Agent - Fallback for general knowledge questions."""

from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from agents.base import BaseAgent
import config
import utils as Utils


class GeneralQAAgent(BaseAgent):
    """General QA Agent handles general knowledge and fallback queries."""

    def __init__(self):
        """Initialize the General QA Agent.

        role="general_qa" binds this agent to its own fine-tuned model under
        GENERATION_PROVIDER=local_peft (Phi-3.5-mini + a Dolly-15k LoRA
        adapter). Note the coordination nodes share this agent's *base* weights
        with the adapter disabled - see config.LOCAL_COORDINATOR_ROLE - so this
        role's cached model and theirs are deliberately separate entries.
        """
        super().__init__(temperature=0.5, role="general_qa")
        self.prompt = ChatPromptTemplate.from_template(config.GENERAL_QA_PROMPT)

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

    def _is_small_talk(self, query: str) -> bool:
        """Detect greetings and brief social messages."""
        normalized = query.strip().lower()
        small_talk = {
            "hi",
            "hello",
            "hey",
            "yo",
            "thanks",
            "thank you",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
        }
        return normalized in small_talk

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
        """Answer the query using general knowledge.

        Args:
            state: The current state with 'query', 'retrieved_docs', 'chat_history'.

        Returns:
            Updated state with agent output.
        """
        query = state.get("query", "")
        self.log(f"Processing general QA query: {query[:50]}...")

        if self._is_small_talk(query):
            short_reply = (
                "Hi! I can help with EU AI Act rules, compliance questions, and recent AI governance news. "
                "What would you like to explore?"
            )

            if "agent_outputs" not in state:
                state["agent_outputs"] = {}
            state["agent_outputs"]["general_qa"] = short_reply
            self.log("Detected small talk; returned concise greeting")
            return state

        # Get context from retrieved documents (may or may not be useful)
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
        state["agent_outputs"]["general_qa"] = response.content

        # Record tokens
        self._record_tokens(state, "general_qa", response)

        self.log(f"Generated general QA response ({len(response.content)} chars)")
        return state
