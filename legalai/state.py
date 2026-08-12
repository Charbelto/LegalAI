"""Shared state schema for the multi-agent Legal AI system."""

from typing import Any, Dict, List, Literal, TypedDict, Annotated
from typing_extensions import TypedDict as ExtTypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


def add_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """Reducer for combining messages."""
    if not isinstance(left, list):
        left = []
    if not isinstance(right, list):
        right = []
    return left + right


def replace_if_present(left: List[Any], right: List[Any]) -> List[Any]:
    """Reducer that replaces list values when a new one is provided.

    This graph returns full state dictionaries from nodes, so additive reducers
    would duplicate list entries on every step.
    """
    if isinstance(right, list):
        return right
    return left if isinstance(left, list) else []


def append_thinking_steps(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reducer that appends new thinking steps, avoiding duplicates based on 'step' and 'details'."""
    if not isinstance(left, list):
        left = []
    if not isinstance(right, list):
        right = []
    
    seen = set()
    result = []
    
    # Add left items first
    for item in left:
        if not isinstance(item, dict):
            continue
        key = (item.get("step"), item.get("details"))
        if key not in seen:
            seen.add(key)
            result.append(item)
            
    # Add right items
    for item in right:
        if not isinstance(item, dict):
            continue
        key = (item.get("step"), item.get("details"))
        if key not in seen:
            seen.add(key)
            result.append(item)
            
    return result


def dict_reducer(left: Dict, right: Dict) -> Dict:
    """Reducer for combining dictionaries."""
    if not isinstance(left, dict):
        left = {}
    if not isinstance(right, dict):
        right = {}
    result = left.copy()
    result.update(right)
    return result


def append_unique_dicts(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reducer that appends new dict entries, deduplicating exact repeats.

    Needed for fields more than one node in the same fan-out step can
    independently append to (e.g. truncation_warnings, written by any agent's
    _record_tokens) - a plain field only tolerates a single writer per step,
    and 2+ fanned-out branches each contributing a value raises LangGraph's
    InvalidUpdateError instead of silently dropping one branch's contribution.

    A plain additive concat (left + right) is NOT safe here: several nodes in
    this graph (aggregator_node, validator_node, response_node, etc.) return
    their entire received state rather than just their own delta, so once one
    node sets this field, every later node "contributes" the same
    already-accumulated list again. Naive concatenation treats each re-return
    as fresh and doubles the list at every subsequent step. dict_reducer above
    doesn't have this problem (merging an unchanged dict into itself is a
    no-op); a list does, unless repeats are filtered out - the same reasoning
    behind append_thinking_steps' dedup-by-key approach.
    """
    if not isinstance(left, list):
        left = []
    if not isinstance(right, list):
        right = []

    def _key(item):
        if isinstance(item, dict):
            return tuple(sorted(item.items()))
        return item

    seen = {_key(item) for item in left}
    result = list(left)
    for item in right:
        k = _key(item)
        if k not in seen:
            seen.add(k)
            result.append(item)
    return result


class AgentState(TypedDict):
    """State schema for the Legal AI multi-agent graph.

    This state is passed between all agents in the workflow.
    Using Annotated with reducers to handle concurrent updates.
    """

    # Input fields (set once, never modified)
    query: str
    session_id: str

    # Memory context (accumulates)
    chat_history: Annotated[List[BaseMessage], replace_if_present]

    # Routing decision
    route: Literal["legal", "news", "general", ""]

    # Expert execution mode (single routed expert or all experts)
    expert_execution_mode: str

    # Retrieved documents from vector store
    retrieved_docs: Annotated[List[Document], lambda x, y: y if y else x]

    # Outputs from individual agents (accumulates)
    agent_outputs: Annotated[Dict[str, str], dict_reducer]

    # Per-node timing metrics in milliseconds (accumulates)
    agent_timings: Annotated[Dict[str, float], dict_reducer]

    # Thinking process log for visualization
    thinking_log: Annotated[List[Dict[str, Any]], append_thinking_steps]

    # Intermediate and final responses
    draft_response: str
    final_response: str

    # Validation results
    validation_result: Dict[str, Any]

    # Loop control
    iteration_count: int

    # Error tracking
    error_message: str

    # Fetched sources tracking
    fetched_sources: Annotated[List[Dict[str, Any]], replace_if_present]

    # Validation issues tracking for retry context
    validation_issues: str

    # LLM token counts per agent node (accumulates)
    agent_tokens: Annotated[Dict[str, Dict[str, int]], dict_reducer]

    # Planned execution path/experts for planner_based mode
    plan: Annotated[List[str], replace_if_present]

    # Abstention telemetry, set once per run by the aggregator only (never
    # written concurrently, so no reducer is needed). LangGraph's StateGraph
    # derives its tracked state channels from this TypedDict's keys; a field
    # the aggregator sets but that isn't declared here is silently dropped
    # when state propagates to the next node, no matter what that node returns.
    abstained: bool
    abstained_experts: List[str]
    experts_run: int
    expert_abstention_rate: float

    # Silent-truncation warnings appended by any agent's _record_tokens (see
    # agents/base.py). Unlike abstained/experts_run above, this can be written
    # by MULTIPLE nodes fanned out in the same step (legal/news/general_qa all
    # run concurrently for parallel/legal_news_parallel/dag), so it needs a
    # reducer - a plain field here raises LangGraph's InvalidUpdateError the
    # moment 2+ of those branches each contribute a value in the same step.
    truncation_warnings: Annotated[List[Dict[str, Any]], append_unique_dicts]
