"""Query Analyzer - Generates optimized search queries from user prompts."""

from langchain_core.prompts import ChatPromptTemplate
from agents.base import build_chat_llm
import config
import utils as Utils


class QueryAnalyzer:
    """Analyzes user queries and generates optimized search queries."""

    def __init__(self, model: str = None):
        """Initialize the Query Analyzer.

        Args:
            model: The Ollama model to use (ignored when
                config.GENERATION_PROVIDER=deepseek). Defaults to config.OLLAMA_MODEL.
        """
        # Routed through build_chat_llm so this respects GENERATION_PROVIDER
        # too - it is invoked on every /chat request (route classification),
        # not just when fetch_news is on, so it must not silently stay on
        # Ollama while every other agent switches to DeepSeek.
        self.llm = build_chat_llm(model=model, temperature=0.2)
        self.prompt = ChatPromptTemplate.from_template(config.QUERY_ANALYSIS_PROMPT)

    def generate_search_query(self, user_prompt: str) -> str:
        """Convert a user question into an optimized web search query.

        Args:
            user_prompt: The user's natural language question.

        Returns:
            Optimized search query string.
        """
        chain = self.prompt | self.llm
        response = chain.invoke({
            "user_prompt": user_prompt,
            "current_date": Utils.get_current_date()
        })

        # Clean up the response
        search_query = response.content.strip()

        # Remove any quotes that might have been added
        search_query = search_query.strip('"').strip("'")

        print(f"[QueryAnalyzer] Generated search query: '{search_query}'")
        return search_query

    def should_fetch_news(self, query: str, route: str) -> bool:
        """Determine if the query requires fetching fresh news.

        Args:
            query: The user's query.
            route: The classified route (legal/news/general).

        Returns:
            True if news fetching is recommended.
        """
        # Always fetch for news route
        if route == "news":
            print(f"[QueryAnalyzer] Route is 'news', fetching recommended")
            return True

        # Check for time-sensitive keywords
        time_keywords = [
            "recent", "latest", "new", "update", "current",
            "2024", "2025", "2026", "this year", "last year",
            "today", "yesterday", "last week", "last month",
            "breaking", "just announced", "news"
        ]

        query_lower = query.lower()
        for keyword in time_keywords:
            if keyword in query_lower:
                print(f"[QueryAnalyzer] Found time keyword '{keyword}', fetching recommended")
                return True

        print(f"[QueryAnalyzer] No time-sensitive keywords found, fetching not needed")
        return False


def quick_generate_search_query(user_prompt: str) -> str:
    """Quick helper function to generate a search query without instantiating the class.

    Args:
        user_prompt: The user's natural language question.

    Returns:
        Optimized search query string.
    """
    analyzer = QueryAnalyzer()
    return analyzer.generate_search_query(user_prompt)


def should_fetch_news_for_query(query: str, route: str) -> bool:
    """Quick helper to check if news fetching is needed.

    Args:
        query: The user's query.
        route: The classified route.

    Returns:
        True if news fetching is recommended.
    """
    analyzer = QueryAnalyzer()
    return analyzer.should_fetch_news(query, route)
