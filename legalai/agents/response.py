"""Response Agent - Final formatting and polishing of responses."""

from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from agents.base import BaseAgent
import config
import utils as Utils


class ResponseAgent(BaseAgent):
    """Response Agent formats the final response for the user."""

    def __init__(self):
        """Initialize the Response Agent."""
        super().__init__(temperature=0.3)
        self.prompt = ChatPromptTemplate.from_template(config.RESPONSE_PROMPT)

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Format the final response for presentation.

        Args:
            state: The current state with 'draft_response' and 'validation_result'.

        Returns:
            Updated state with 'final_response' populated.
        """
        draft_response = state.get("draft_response", "")
        query = state.get("query", "")

        self.log("Formatting final response")

        # Propagate abstention only when the aggregator decided the system abstains
        # (or the draft is nothing but the abstention sentence). Matching the
        # sentence as a substring previously discarded otherwise-complete answers
        # that merely quoted or referenced it.
        if state.get("abstained") or draft_response.strip() == config.ABSTENTION_SENTENCE:
            state["final_response"] = config.ABSTENTION_SENTENCE
            self.log("Abstention propagated from aggregator; skipping formatting pass")
            return state

        if not draft_response:
            state["final_response"] = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            return state

        small_talk_queries = {
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

        if query.strip().lower() in small_talk_queries:
            state["final_response"] = draft_response.strip()
            self.log("Detected small talk; skipping formatting LLM pass")
            return state

        # Create the chain and invoke
        chain = self.prompt | self.llm

        response = chain.invoke({
            "response": draft_response,
            "query": query,
            "current_date": Utils.get_current_date(),
        })

        state["final_response"] = response.content

        # Record tokens
        self._record_tokens(state, "response", response)

        self.log(f"Final response formatted ({len(response.content)} chars)")

        # Store the exchange in memory for future context
        # This is handled by MemoryAgent in the next iteration

        return state

    def format_error_response(self, error_message: str) -> str:
        """Format a user-friendly error response.

        Args:
            error_message: The technical error message.

        Returns:
            User-friendly formatted error message.
        """
        return f"I apologize, but I encountered an issue while processing your request. Please try again or rephrase your question."
