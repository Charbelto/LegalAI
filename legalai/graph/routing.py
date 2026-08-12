"""Pure routing helpers.

Kept free of langgraph/chromadb imports so the topology decisions can be unit
tested without standing up the whole stack.
"""

from typing import Optional

# Router labels in the order the expert nodes are named in the graph.
_LABEL_TO_NODE = (("legal", "legal"), ("news", "news"), ("general", "general_qa"))


def select_single_expert(route: Optional[str]) -> str:
    """Pick exactly one expert node for SINGLE mode.

    SINGLE previously fanned out to every label present in a multi-label route,
    so the "single agent" baseline was really a 2-3 agent ensemble whenever the
    router hedged. That makes the single-vs-multi comparison unfalsifiable, so
    SINGLE now runs the router's *primary* label: the one mentioned first.

    Args:
        route: Raw router output, e.g. "legal", "legal,news", "general".

    Returns:
        Exactly one of "legal", "news", "general_qa".
    """
    normalized = str(route or "general").lower()

    candidates = []
    for label, node in _LABEL_TO_NODE:
        position = normalized.find(label)
        if position >= 0:
            candidates.append((position, node))

    if not candidates:
        return "general_qa"

    return min(candidates)[1]
