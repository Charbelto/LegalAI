"""Aggregator Agent - Combines outputs from multiple agents into a coherent response."""

from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from agents.base import BaseAgent
import config
import utils as Utils

# The three domain experts, and only these. `agent_outputs` is a shared state
# channel that non-expert nodes also write to (agents/router.py records its route
# under "router"), so any count or rate derived from it must filter to these keys
# or it silently includes non-experts.
EXPERT_KEYS = ("legal", "news", "general_qa")


class AggregatorAgent(BaseAgent):
    """Aggregator Agent synthesizes outputs from multiple agents."""

    def __init__(self):
        """Initialize the Aggregator Agent."""
        super().__init__(temperature=0.2)
        self.prompt = ChatPromptTemplate.from_template(config.AGGREGATOR_PROMPT)

    def _format_history(self, history: list) -> str:
        """Format chat history for the prompt.

        Args:
            history: List of chat messages.

        Returns:
            Formatted history string.
        """
        if not history:
            return "No previous conversation."

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

    def _partition_abstentions(self, agent_outputs: Dict[str, str]) -> tuple:
        """Split expert outputs into substantive answers and abstentions.

        An expert abstains by emitting exactly ``config.ABSTENTION_SENTENCE`` (see
        LEGAL_PROMPT). Abstention is a per-expert *signal*, not a veto over the
        ensemble: a topology that runs three experts must not be silenced because
        one of them lacked authoritative context. Silencing on any single
        abstention systematically penalised every topology containing the legal
        expert while leaving router-only and expert-free topologies untouched,
        which confounds the single-vs-multi comparison this experiment exists to
        make.

        Only the three DOMAIN EXPERT keys are considered. ``agent_outputs`` is a
        shared channel that non-expert nodes also write to - agents/router.py
        stores its route decision under "router" - and counting every key made
        experts_run 4 in a three-expert topology. That inflated denominator went
        straight into expert_abstention_rate (abstained / experts_run), which the
        paper reports, understating it by a quarter on every run. Restricting to
        the expert keys is the fix; the alternative, stopping the router writing
        there, would change a field the UI reads.

        Returns:
            (substantive_outputs, abstained_expert_keys)
        """
        substantive = {}
        abstained = []
        for key, output in (agent_outputs or {}).items():
            if key not in EXPERT_KEYS:
                continue
            if not isinstance(output, str) or not output.strip():
                continue
            if config.ABSTENTION_SENTENCE in output:
                abstained.append(key)
            else:
                substantive[key] = output
        return substantive, sorted(abstained)

    def _primary_expert_key(self, route: str) -> str:
        """Map router route values to expert output keys."""
        if route in {"legal", "news"}:
            return route
        return "general_qa"

    def _build_expert_output(self, agent_outputs: Dict[str, str], route: str) -> tuple[str, str]:
        """Build a merged expert context, prioritizing the routed domain first."""
        primary = self._primary_expert_key(route)
        ordered_keys = [primary, "legal", "news", "general_qa"]

        seen = set()
        unique_order = []
        for key in ordered_keys:
            if key not in seen:
                seen.add(key)
                unique_order.append(key)

        available = []
        for key in unique_order:
            output = agent_outputs.get(key, "")
            if isinstance(output, str) and output.strip():
                available.append((key, output.strip()))

        if not available:
            return "No expert analysis available.", "none"

        if len(available) == 1:
            key, value = available[0]
            return value, key

        labels = {
            "legal": "Legal",
            "news": "News",
            "general_qa": "General QA",
        }

        merged_sections = []
        for key, value in available:
            merged_sections.append(f"[{labels.get(key, key)}]\n{value}")

        merged_output = "\n\n".join(merged_sections)
        active_keys = ", ".join(key for key, _ in available)
        return merged_output, f"multi: {active_keys}"

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all agent outputs into a unified response.

        Args:
            state: The current state with agent outputs, retrieved docs, chat history.

        Returns:
            Updated state with 'draft_response' populated.
        """
        query = state.get("query", "")
        route = state.get("route", "general")
        agent_outputs = state.get("agent_outputs", {})

        self.log(f"Aggregating outputs for route: {route}")

        substantive_outputs, abstained_experts = self._partition_abstentions(agent_outputs)
        experts_run = len(substantive_outputs) + len(abstained_experts)

        # Abstention telemetry: reported as its own metric rather than silently
        # rewriting the answer, so abstention rate can be compared across topologies.
        state["abstained_experts"] = abstained_experts
        state["experts_run"] = experts_run
        state["expert_abstention_rate"] = (
            round(len(abstained_experts) / experts_run, 4) if experts_run else 0.0
        )

        if experts_run and not substantive_outputs:
            # Every expert that ran abstained, so the system abstains as a whole.
            state["abstained"] = True
            state["draft_response"] = config.ABSTENTION_SENTENCE
            self.log(f"All {experts_run} expert(s) abstained; propagating abstention")
            return state

        state["abstained"] = False
        expert_output, agent_label = self._build_expert_output(substantive_outputs, route)

        if abstained_experts:
            self.log(
                f"Partial abstention: {abstained_experts} abstained; aggregating "
                f"{sorted(substantive_outputs)}"
            )
            expert_output += (
                "\n\n[Note: the following experts abstained for lack of authoritative "
                f"support and contributed nothing: {', '.join(abstained_experts)}. "
                "Answer only from the analysis above and do not speculate about their "
                "areas.]"
            )

        if self._is_small_talk(query):
            # For social talk, prefer concise general QA output if available.
            short_output = agent_outputs.get("general_qa", "")
            state["draft_response"] = short_output.strip() or expert_output
            self.log("Detected small talk; skipped long-form aggregation")
            return state

        # Get context
        docs = state.get("retrieved_docs", [])
        context = self._format_context(docs)

        # Format chat history
        chat_history = self._format_history(state.get("chat_history", []))

        # If there are validation issues from a previous run, incorporate them
        validation_issues = state.get("validation_result", {}).get("issues", "")
        if validation_issues and validation_issues != "None":
            expert_output += f"\n\n[Note: Please address these issues from previous validation: {validation_issues}]"

        # Create the chain and invoke
        chain = self.prompt | self.llm

        response = chain.invoke({
            "query": query,
            "context": context,
            "chat_history": chat_history,
            "agent_type": agent_label,
            "expert_output": expert_output,
            "current_date": Utils.get_current_date(),
        })

        state["draft_response"] = response.content

        # Record tokens
        self._record_tokens(state, "aggregator", response)

        self.log(f"Generated aggregated response ({len(response.content)} chars)")
        return state
