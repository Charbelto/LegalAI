"""Router Agent - Classifies queries into legal/news/general primary focus labels."""

from typing import Any, Dict, Literal
from langchain_core.prompts import ChatPromptTemplate
from agents.base import BaseAgent
import config
import utils as Utils


class RouterAgent(BaseAgent):
    """Router Agent classifies user queries to set primary synthesis focus."""

    def __init__(self):
        """Initialize the Router Agent."""
        super().__init__(temperature=0.0)  # Low temperature for consistent classification
        self.prompt = ChatPromptTemplate.from_template(config.ROUTER_PROMPT)

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the query and set the route.

        Args:
            state: The current state with 'query'.

        Returns:
            Updated state with 'route' field set.
        """
        query = state.get("query", "")
        self.log(f"Routing query: {query[:50]}...")

        # Create the classification chain
        chain = self.prompt | self.llm

        # Get classification with current date
        response = chain.invoke({
            "query": query,
            "current_date": Utils.get_current_date()
        })
        classification = response.content.strip().lower()

        # Normalize to valid routes (allowing multiple comma-separated)
        valid_routes = ["legal", "news", "general"]
        routes_found = []

        for valid_route in valid_routes:
            if valid_route in classification:
                routes_found.append(valid_route)

        if not routes_found:
            routes_found = ["general"]

        route = ", ".join(routes_found)
        state["route"] = route
        self.log(f"Route determined: {route}")

        # Store classification in agent outputs
        if "agent_outputs" not in state:
            state["agent_outputs"] = {}
        state["agent_outputs"]["router"] = route

        # Record tokens
        self._record_tokens(state, "router", response)

        return state
