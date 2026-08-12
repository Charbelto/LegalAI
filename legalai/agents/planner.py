"""Planner Agent - Entry point that orchestrates the workflow."""

from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from agents.base import BaseAgent
import config
import utils as Utils


class PlannerAgent(BaseAgent):
    """Planner Agent initializes the state and coordinates execution."""

    def __init__(self):
        """Initialize the Planner Agent."""
        super().__init__(temperature=0.1)
        self.prompt = ChatPromptTemplate.from_template(config.PLANNER_PROMPT)

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize state and prepare for execution.

        Args:
            state: The current state with at least 'query' and 'session_id'.

        Returns:
            Updated state with initialized fields.
        """
        self.log(f"Planning execution for query: {state.get('query', 'N/A')[:50]}...")

        # Initialize state fields if not present
        if "agent_outputs" not in state or state["agent_outputs"] is None:
            state["agent_outputs"] = {}

        if "agent_timings" not in state or state["agent_timings"] is None:
            state["agent_timings"] = {}

        if "agent_tokens" not in state or state["agent_tokens"] is None:
            state["agent_tokens"] = {}

        if "plan" not in state or state["plan"] is None:
            state["plan"] = []

        if "retrieved_docs" not in state or state["retrieved_docs"] is None:
            state["retrieved_docs"] = []

        if "chat_history" not in state or state["chat_history"] is None:
            state["chat_history"] = []

        if "route" not in state or state["route"] is None:
            state["route"] = ""

        if "expert_execution_mode" not in state or state["expert_execution_mode"] is None:
            state["expert_execution_mode"] = config.EXPERT_EXECUTION_MODE

        if "draft_response" not in state:
            state["draft_response"] = ""

        if "final_response" not in state:
            state["final_response"] = ""

        if "validation_result" not in state or state["validation_result"] is None:
            state["validation_result"] = {"pass": True, "issues": ""}

        if "iteration_count" not in state:
            state["iteration_count"] = 0

        if "error_message" not in state:
            state["error_message"] = ""

        if "thinking_log" not in state or state["thinking_log"] is None:
            state["thinking_log"] = []

        if "fetched_sources" not in state or state["fetched_sources"] is None:
            state["fetched_sources"] = []

        if "validation_issues" not in state:
            state["validation_issues"] = ""

        # Determine execution path if planner-based mode is active
        mode = state.get("expert_execution_mode", config.EXPERT_EXECUTION_MODE)
        if mode == "planner_based":
            self.log("Planner Agent running LLM planning...")
            chain = self.prompt | self.llm
            response = chain.invoke({
                "query": state.get("query", ""),
                "current_date": Utils.get_current_date(),
                "session_id": state.get("session_id", "N/A"),
            })

            # Parse JSON response
            import json
            import re
            content = response.content.strip()
            # Clean up potential markdown formatting
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n", "", content)
                content = re.sub(r"\n```$", "", content)

            try:
                parsed = json.loads(content)
                plan = parsed.get("plan", ["legal"])
                normalized_plan = []
                for p in plan:
                    p_clean = str(p).strip().lower()
                    if "legal" in p_clean:
                        normalized_plan.append("legal")
                    elif "news" in p_clean:
                        normalized_plan.append("news")
                    elif "general" in p_clean or "qa" in p_clean:
                        normalized_plan.append("general_qa")
                if not normalized_plan:
                    normalized_plan = ["legal"]
                state["plan"] = normalized_plan
                self.log(f"Planner determined execution path: {normalized_plan}")
            except Exception as e:
                self.log(f"Failed to parse planner plan JSON, falling back to ['legal']: {e}. Content: {content}")
                state["plan"] = ["legal"]

            # Record tokens
            meta = response.response_metadata or {}
            prompt_tokens = meta.get("prompt_eval_count", 0) or meta.get("prompt_tokens", 0) or 0
            completion_tokens = meta.get("eval_count", 0) or meta.get("completion_tokens", 0) or 0
            state["agent_tokens"]["planner"] = {
                "prompt": prompt_tokens,
                "completion": completion_tokens
            }

        self.log(f"State initialized. Iteration: {state['iteration_count']}")
        return state
