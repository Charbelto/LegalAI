"""LangGraph workflow definition for the Legal AI multi-agent system."""

from time import perf_counter
from typing import Literal, Any, Callable
from langgraph.graph import StateGraph, END

from state import AgentState
from graph.routing import select_single_expert
from agents import (
    PlannerAgent,
    RouterAgent,
    MemoryAgent,
    RetrievalAgent,
    LegalAgent,
    NewsAgent,
    GeneralQAAgent,
    AggregatorAgent,
    ValidationAgent,
    ResponseAgent,
)
import config


def add_thinking_step(
    state: AgentState,
    step_name: str,
    details: str = "",
    elapsed_ms: float | None = None,
):
    """Add a thinking step to the log for visualization."""
    if "thinking_log" not in state or state["thinking_log"] is None:
        state["thinking_log"] = []
    entry = {
        "step": step_name,
        "details": details,
        "agent_outputs": state.get("agent_outputs", {}).copy()
    }
    if elapsed_ms is not None:
        entry["elapsed_ms"] = round(elapsed_ms, 2)
    state["thinking_log"].append(entry)


def record_agent_timing(state: AgentState, node_name: str, elapsed_ms: float):
    """Accumulate timing metrics (in ms) for each graph node."""
    timings = state.get("agent_timings", {})
    if not isinstance(timings, dict):
        timings = {}

    try:
        previous = float(timings.get(node_name, 0.0) or 0.0)
    except (TypeError, ValueError):
        previous = 0.0

    timings[node_name] = round(previous + elapsed_ms, 2)
    state["agent_timings"] = timings


def invoke_with_timing(
    state: AgentState,
    node_name: str,
    invoke_fn: Callable[[AgentState], AgentState],
) -> tuple[AgentState, float]:
    """Invoke a node and return both result state and elapsed time in ms."""
    started_at = perf_counter()
    result = invoke_fn(state)
    elapsed_ms = (perf_counter() - started_at) * 1000
    record_agent_timing(result, node_name, elapsed_ms)
    return result, elapsed_ms


def create_legal_ai_graph() -> Any:
    """Create and compile the Legal AI multi-agent workflow graph.

    Uses sequential execution to avoid concurrent state update issues.
    Expert execution can run in single-expert or all-experts mode.

    Returns:
        Compiled LangGraph ready for execution.
    """
    # Initialize agents
    planner = PlannerAgent()
    router = RouterAgent()
    memory = MemoryAgent()
    retrieval = RetrievalAgent()
    legal = LegalAgent()
    news = NewsAgent()
    general_qa = GeneralQAAgent()
    aggregator = AggregatorAgent()
    validator = ValidationAgent()
    response = ResponseAgent()

    # Create the graph
    workflow = StateGraph(AgentState)

    # Define node functions with thinking logs
    def planner_node(state: AgentState) -> AgentState:
        """Planner node - initializes state."""
        add_thinking_step(state, "planner", "Initializing workflow and state")
        result, elapsed_ms = invoke_with_timing(state, "planner", planner.invoke)
        add_thinking_step(
            result,
            "planner_complete",
            f"Query: {result.get('query', '')[:50]}... ({elapsed_ms:.1f} ms)",
            elapsed_ms=elapsed_ms,
        )
        return result

    def router_node(state: AgentState) -> AgentState:
        """Router node - classifies the query."""
        add_thinking_step(state, "router", "Classifying query type")
        result, elapsed_ms = invoke_with_timing(state, "router", router.invoke)
        route = result.get("route", "general")
        mode = result.get("expert_execution_mode", config.EXPERT_EXECUTION_MODE)
        add_thinking_step(
            result,
            "router_complete",
            f"Classified as: {route} | Mode: {str(mode).upper()} ({elapsed_ms:.1f} ms)",
            elapsed_ms=elapsed_ms,
        )
        return result

    def memory_node(state: AgentState) -> AgentState:
        """Memory node - retrieves conversation history."""
        add_thinking_step(state, "memory", "Retrieving conversation history")
        result, elapsed_ms = invoke_with_timing(state, "memory", memory.invoke)
        history_len = len(result.get("chat_history", []))
        add_thinking_step(
            result,
            "memory_complete",
            f"Retrieved {history_len} messages from history ({elapsed_ms:.1f} ms)",
            elapsed_ms=elapsed_ms,
        )
        return result

    def retrieval_node(state: AgentState) -> AgentState:
        """Retrieval node - searches vector store."""
        add_thinking_step(state, "retrieval", "Searching vector database for relevant documents")
        result, elapsed_ms = invoke_with_timing(state, "retrieval", retrieval.invoke)
        docs_count = len(result.get("retrieved_docs", []))
        add_thinking_step(
            result,
            "retrieval_complete",
            f"Retrieved {docs_count} documents ({elapsed_ms:.1f} ms)",
            elapsed_ms=elapsed_ms,
        )
        return result

    def legal_node(state: AgentState) -> AgentState:
        """Legal agent node - processes legal queries."""
        add_thinking_step(state, "legal", "Legal Agent analyzing EU AI Act provisions")
        result, elapsed_ms = invoke_with_timing(state, "legal", legal.invoke)
        output = result.get("agent_outputs", {}).get("legal", "")[:100]
        add_thinking_step(
            result,
            "legal_complete",
            f"Legal analysis: {output}... ({elapsed_ms:.1f} ms)",
            elapsed_ms=elapsed_ms,
        )
        return {
            "agent_outputs": result.get("agent_outputs", {}),
            "agent_timings": result.get("agent_timings", {}),
            "thinking_log": result.get("thinking_log", []),
            "agent_tokens": result.get("agent_tokens", {}),
            "truncation_warnings": result.get("truncation_warnings", []),
        }

    def news_node(state: AgentState) -> AgentState:
        """News agent node - processes news queries."""
        add_thinking_step(state, "news", "News Agent analyzing current events")
        result, elapsed_ms = invoke_with_timing(state, "news", news.invoke)
        output = result.get("agent_outputs", {}).get("news", "")[:100]
        add_thinking_step(
            result,
            "news_complete",
            f"News analysis: {output}... ({elapsed_ms:.1f} ms)",
            elapsed_ms=elapsed_ms,
        )
        return {
            "agent_outputs": result.get("agent_outputs", {}),
            "agent_timings": result.get("agent_timings", {}),
            "thinking_log": result.get("thinking_log", []),
            "agent_tokens": result.get("agent_tokens", {}),
            "truncation_warnings": result.get("truncation_warnings", []),
        }

    def general_qa_node(state: AgentState) -> AgentState:
        """General QA agent node - processes general queries."""
        add_thinking_step(state, "general_qa", "General QA Agent processing query")
        result, elapsed_ms = invoke_with_timing(state, "general_qa", general_qa.invoke)
        output = result.get("agent_outputs", {}).get("general_qa", "")[:100]
        add_thinking_step(
            result,
            "general_qa_complete",
            f"QA response: {output}... ({elapsed_ms:.1f} ms)",
            elapsed_ms=elapsed_ms,
        )
        return {
            "agent_outputs": result.get("agent_outputs", {}),
            "agent_timings": result.get("agent_timings", {}),
            "thinking_log": result.get("thinking_log", []),
            "agent_tokens": result.get("agent_tokens", {}),
            "truncation_warnings": result.get("truncation_warnings", []),
        }

    def aggregator_node(state: AgentState) -> AgentState:
        """Aggregator node - combines all outputs."""
        add_thinking_step(state, "aggregator", "Combining all agent outputs into unified response")
        result, elapsed_ms = invoke_with_timing(state, "aggregator", aggregator.invoke)
        draft = result.get("draft_response", "")[:100]
        add_thinking_step(
            result,
            "aggregator_complete",
            f"Draft response: {draft}... ({elapsed_ms:.1f} ms)",
            elapsed_ms=elapsed_ms,
        )
        return result

    def validator_node(state: AgentState) -> AgentState:
        """Validator node - checks response quality."""
        add_thinking_step(state, "validator", "Validating response quality")
        result, elapsed_ms = invoke_with_timing(state, "validator", validator.invoke)
        validation = result.get("validation_result", {})
        status = "PASSED" if validation.get("pass", False) else "FAILED"
        
        # Increment iteration count if validation failed
        if not validation.get("pass", False):
            result["iteration_count"] = result.get("iteration_count", 0) + 1
            
        add_thinking_step(
            result,
            "validator_complete",
            f"Validation: {status} | Iteration: {result.get('iteration_count', 0)} ({elapsed_ms:.1f} ms)",
            elapsed_ms=elapsed_ms,
        )
        return result

    def response_node(state: AgentState) -> AgentState:
        """Response node - formats final output."""
        add_thinking_step(state, "response", "Formatting final response")
        result, elapsed_ms = invoke_with_timing(state, "response", response.invoke)
        final = result.get("final_response", "")[:100]
        add_thinking_step(
            result,
            "response_complete",
            f"Final response ready: {final}... ({elapsed_ms:.1f} ms)",
            elapsed_ms=elapsed_ms,
        )
        return result

    # Add all nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("router", router_node)
    workflow.add_node("memory", memory_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("legal", legal_node)
    workflow.add_node("news", news_node)
    workflow.add_node("general_qa", general_qa_node)
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("response", response_node)

    # Define sequential edges (avoiding parallel branches for state safety)
    # Start -> Planner
    workflow.set_entry_point("planner")

    # Planner -> Router
    workflow.add_edge("planner", "router")

    # Router -> Memory (sequential to avoid concurrent updates)
    workflow.add_edge("router", "memory")

    # Memory -> Retrieval
    workflow.add_edge("memory", "retrieval")

    # Retrieval -> Expert(s)
    def _effective_expert_mode(state: AgentState) -> str:
        mode = str(state.get("expert_execution_mode", config.EXPERT_EXECUTION_MODE)).strip().lower()
        if mode in {"all", "single", "parallel", "legal_news_parallel", "legal_first", "verify_only", "planner_based", "graph_engineering", "graph", "dag"}:
            return mode
        return "all"

    def route_from_retrieval(state: AgentState) -> Literal["legal", "news", "general_qa", "aggregator"] | list[str]:
        """Route from retrieval to either one expert, full expert chain, or parallel paths."""
        mode = _effective_expert_mode(state)
        if mode == "all":
            return "legal"
        elif mode == "parallel":
            return ["legal", "news", "general_qa"]
        elif mode == "legal_news_parallel":
            return ["legal", "news"]
        elif mode == "legal_first":
            return "legal"
        elif mode == "verify_only":
            return "aggregator"
        elif mode == "planner_based":
            plan = state.get("plan") or []
            if not plan:
                return "legal"
            experts = []
            if "legal" in plan:
                experts.append("legal")
            if "news" in plan:
                experts.append("news")
            if "general_qa" in plan:
                experts.append("general_qa")
            if not experts:
                return "legal"
            return experts if len(experts) > 1 else experts[0]
        elif mode in {"graph_engineering", "graph", "dag"}:
            return ["legal", "news"]

        # "single" mode: exactly ONE expert, always.
        return select_single_expert(state.get("route", "general"))

    workflow.add_conditional_edges(
        "retrieval",
        route_from_retrieval,
        {
            "legal": "legal",
            "news": "news",
            "general_qa": "general_qa",
            "aggregator": "aggregator",
        },
    )

    def route_after_legal(state: AgentState) -> Literal["news", "aggregator", "general_qa"]:
        mode = _effective_expert_mode(state)
        route = state.get("route", "")
        is_multi_route = len([r for r in ["legal", "news", "general"] if r in route]) > 1
        if mode in {"graph_engineering", "graph", "dag"}:
            return "general_qa"
        if mode == "planner_based":
            return "aggregator"
        if mode in {"parallel", "legal_news_parallel"} or (mode == "single" and is_multi_route):
            return "aggregator"
        if mode in {"all", "legal_first"}:
            return "news"
        return "aggregator"

    workflow.add_conditional_edges(
        "legal",
        route_after_legal,
        {
            "news": "news",
            "aggregator": "aggregator",
            "general_qa": "general_qa",
        },
    )

    def route_after_news(state: AgentState) -> Literal["general_qa", "aggregator"]:
        mode = _effective_expert_mode(state)
        route = state.get("route", "")
        is_multi_route = len([r for r in ["legal", "news", "general"] if r in route]) > 1
        if mode in {"graph_engineering", "graph", "dag"}:
            return "general_qa"
        if mode == "planner_based":
            return "aggregator"
        if mode in {"parallel", "legal_news_parallel"} or (mode == "single" and is_multi_route):
            return "aggregator"
        if mode == "all":
            return "general_qa"
        return "aggregator"

    workflow.add_conditional_edges(
        "news",
        route_after_news,
        {
            "general_qa": "general_qa",
            "aggregator": "aggregator",
        },
    )

    workflow.add_edge("general_qa", "aggregator")

    # Aggregator -> Validator, but ONLY for graph_engineering.
    #
    # Loop Engineering (the terminal Reflection/Evaluator-Optimizer pass) is the
    # architectural feature that distinguishes Graph Engineering from the other
    # topologies in this experiment. Earlier, the validator ran unconditionally
    # for every mode, which made "reflection" a constant present in every arm
    # rather than the thing being tested - ALL and PARALLEL would silently get
    # the same fix-up pass as graph_engineering, so any quality or latency
    # difference measured against them was NOT attributable to the terminal
    # verification loop. Restricting it to graph_engineering makes reflection the
    # actual manipulated variable: ALL and PARALLEL terminate immediately after
    # aggregation, graph_engineering alone gets the critique-and-revise loop.
    def route_after_aggregator(state: AgentState) -> Literal["validator", "response"]:
        mode = _effective_expert_mode(state)
        if mode in {"graph_engineering", "graph", "dag"}:
            return "validator"
        return "response"

    workflow.add_conditional_edges(
        "aggregator",
        route_after_aggregator,
        {
            "validator": "validator",
            "response": "response",
        },
    )

    # Validator -> Response (if pass) or back to Planner (if fail and retries left)
    # Or re-fetch if sources are bad
    def validate_and_route(state: AgentState) -> Literal["response", "planner", "refetch", "response_max_retry"]:
        """Route based on validation result and iteration count."""
        validation_result = state.get("validation_result", {})
        iteration_count = state.get("iteration_count", 0)

        # If validation passed, go to response
        if validation_result.get("pass", True):
            return "response"

        # Check if we need to re-fetch sources
        if validation_result.get("retry_fetch", False) or not validation_result.get("source_relevant", True):
            if iteration_count < config.MAX_ITERATIONS:
                return "refetch"

        # If validation failed but we haven't exceeded max iterations, retry with planner
        if iteration_count < config.MAX_ITERATIONS:
            return "planner"

        # Max retries exceeded, go to response anyway
        return "response_max_retry"

    workflow.add_conditional_edges(
        "validator",
        validate_and_route,
        {
            "response": "response",
            "planner": "planner",
            "refetch": "retrieval",  # Re-fetch goes back to retrieval to get new sources
            "response_max_retry": "response",
        }
    )

    # Response -> END
    workflow.add_edge("response", END)

    # Compile the graph
    return workflow.compile()


def create_agent_executor() -> Any:
    """Create a simplified agent executor for direct use.

    This is a convenience wrapper around create_legal_ai_graph().

    Returns:
        Compiled LangGraph ready for execution.
    """
    return create_legal_ai_graph()
