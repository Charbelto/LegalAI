"""Auto Fetcher - Orchestrates automatic news fetching with progress tracking."""

from typing import Callable, Optional, List, Dict
from datetime import datetime
import utils as Utils
import embed
from query_analyzer import QueryAnalyzer, should_fetch_news_for_query


class FetchProgress:
    """Tracks progress of the fetch operation."""

    def __init__(self):
        self.step = 0
        self.total_steps = 5
        self.current_action = ""
        self.details = {}
        self.current_article_url = None
        self.current_article_title = None
        self.fetched_articles = []  # List of {url, title, status, error}
        self.current_article_index = 0
        self.total_articles = 0

    def update(self, step: int, action: str, details: dict = None):
        """Update progress.

        Args:
            step: Current step number (1-based).
            action: Description of current action.
            details: Additional details about progress.
        """
        self.step = step
        self.current_action = action
        if details:
            self.details.update(details)

    def set_article(self, index: int, url: str, title: str = ""):
        """Set the current article being fetched.

        Args:
            index: Article index (0-based).
            url: The URL being fetched.
            title: Article title if known.
        """
        self.current_article_index = index
        self.current_article_url = url
        self.current_article_title = title

    def add_fetched_article(self, url: str, title: str, status: str, error: str = ""):
        """Add a fetched article to the list.

        Args:
            url: The article URL.
            title: Article title.
            status: "success" or "failed".
            error: Error message if failed.
        """
        self.fetched_articles.append({
            "url": url,
            "title": title or "Unknown",
            "status": status,
            "error": error
        })

    def get_progress_percent(self) -> float:
        """Get progress as percentage (0.0 to 1.0)."""
        return self.step / self.total_steps

    def get_detailed_status(self) -> str:
        """Get a detailed status message for display.

        Returns:
            Formatted status string showing current activity.
        """
        base_status = f"Step {self.step}/{self.total_steps}: {self.current_action}"

        if self.current_article_url and self.step == 3:
            # During fetching step
            article_info = self.current_article_title or self.current_article_url[:60] + "..."
            base_status += f"\n  Fetching {self.current_article_index + 1}/{self.total_articles}: {article_info}"

        return base_status

    def __str__(self) -> str:
        return self.get_detailed_status()


class AutoNewsFetcher:
    """Automatically fetches and embeds news based on query analysis."""

    def __init__(self, num_articles: int = 5):
        """Initialize the auto fetcher.

        Args:
            num_articles: Number of articles to fetch per query.
        """
        self.num_articles = num_articles
        self.query_analyzer = QueryAnalyzer()
        self.last_fetch_time = None
        self.last_fetch_count = 0

    def analyze_and_decide(self, user_query: str, route: str) -> tuple[bool, str]:
        """Analyze query and decide if fetching is needed.

        Args:
            user_query: The user's original query.
            route: The classified route (legal/news/general).

        Returns:
            Tuple of (should_fetch, search_query).
        """
        # Check if fetching is needed
        should_fetch = should_fetch_news_for_query(user_query, route)

        if not should_fetch:
            return False, ""

        # Generate optimized search query
        search_query = self.query_analyzer.generate_search_query(user_query)

        return True, search_query

    def _fetch_articles_with_details(
        self,
        search_query: str,
        progress: FetchProgress,
        progress_callback: Optional[Callable[[FetchProgress], None]] = None
    ) -> List[Dict]:
        """Fetch articles with detailed per-article progress tracking.

        Args:
            search_query: The query to search for.
            progress: The FetchProgress object to update.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of fetched articles.
        """
        from scraper import fetch_and_scrape_articles

        articles = []

        # First, get the search results without scraping
        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                search_results = list(ddgs.text(search_query, max_results=self.num_articles))

                if not search_results:
                    return articles

                progress.total_articles = len(search_results)

                # Now fetch each article individually with progress
                for i, result in enumerate(search_results):
                    url = result.get('href', '')
                    title = result.get('title', 'Unknown')

                    # Update progress for this article
                    progress.set_article(i, url, title)
                    progress.current_action = f"Fetching article {i + 1}/{len(search_results)}"
                    if progress_callback:
                        progress_callback(progress)

                    try:
                        # Fetch and scrape this individual article
                        article = self._fetch_single_article(result)
                        if article:
                            articles.append(article)
                            progress.add_fetched_article(url, article.get('title', title), "success")
                        else:
                            progress.add_fetched_article(url, title, "failed", "Could not extract content")
                    except Exception as e:
                        progress.add_fetched_article(url, title, "failed", str(e))
                        print(f"Error fetching article {url}: {e}")

        except Exception as e:
            print(f"Error in search: {e}")
            # Fall back to standard batch fetching
            articles = Utils.fetch_news_articles(search_query, self.num_articles)

            # Populate progress entries from fallback results so UI counts remain accurate.
            progress.fetched_articles = [
                {
                    "url": article.get("url", ""),
                    "title": article.get("title", "Unknown"),
                    "status": "success",
                    "error": "",
                }
                for article in articles
            ]

        return articles

    def _fetch_single_article(self, result: Dict) -> Optional[Dict]:
        """Fetch and scrape a single article.

        Args:
            result: Search result dict with 'href', 'title', etc.

        Returns:
            Article dict or None if failed.
        """
        import time

        url = result.get('href', '')
        title = result.get('title', '')

        if not url:
            return None

        try:
            # Use the scraper's internal logic for a single article
            # We'll use a simple approach: get the URL and scrape it
            from bs4 import BeautifulSoup
            import requests

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Try to find main content
            content = ""
            selectors = ['article', '[role="main"]', '.article-content', '.post-content',
                        '.entry-content', 'main', '.content']

            for selector in selectors:
                element = soup.select_one(selector)
                if element:
                    content = element.get_text(separator='\n', strip=True)
                    break

            if not content:
                # Fallback to paragraphs
                paragraphs = soup.find_all('p')
                content = '\n'.join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50)

            if not content or len(content) < 200:
                return None

            return {
                'title': title or soup.title.string if soup.title else 'Unknown',
                'url': url,
                'source': result.get('source', 'Unknown'),
                'content': content[:10000]  # Limit content size
            }

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
        finally:
            # Be polite - add a small delay between requests
            time.sleep(0.5)

    def fetch_with_progress(
        self,
        user_query: str,
        route: str,
        progress_callback: Optional[Callable[[FetchProgress], None]] = None,
        clear_existing: bool = True
    ) -> Dict:
        """Fetch news with progress tracking.

        Args:
            user_query: The user's original query.
            route: The classified route.
            progress_callback: Optional callback to receive progress updates.
            clear_existing: Whether to clear existing articles before embedding.

        Returns:
            Dictionary with fetch results and metadata.
        """
        progress = FetchProgress()
        result = {
            "fetched": False,
            "articles_count": 0,
            "search_query": "",
            "error": None,
            "timestamp": None
        }

        try:
            # Step 1: Analyze query
            progress.update(1, "Analyzing query...", {"query": user_query})
            if progress_callback:
                progress_callback(progress)

            should_fetch, search_query = self.analyze_and_decide(user_query, route)

            if not should_fetch:
                progress.update(5, "Skipping fetch - query doesn't require fresh news")
                if progress_callback:
                    progress_callback(progress)
                result["fetched"] = False
                return result

            result["search_query"] = search_query

            # Step 2: Generate search query
            progress.update(2, f"Generating search query: '{search_query}'")
            if progress_callback:
                progress_callback(progress)

            # Step 3: Fetch articles with detailed progress
            progress.total_articles = self.num_articles
            progress.update(3, f"Fetching {self.num_articles} articles...")
            if progress_callback:
                progress_callback(progress)

            # Fetch with detailed progress tracking
            articles = self._fetch_articles_with_details(search_query, progress, progress_callback)

            if not articles:
                progress.update(5, "No articles found")
                if progress_callback:
                    progress_callback(progress)
                result["error"] = "No articles found for the generated search query"
                return result

            result["articles_count"] = len(articles)
            result["fetched_articles"] = progress.fetched_articles or [
                {
                    "url": article.get("url", ""),
                    "title": article.get("title", "Unknown"),
                    "status": "success",
                    "error": "",
                }
                for article in articles
            ]
            progress.update(3, f"Fetched {len(articles)} articles", {"articles": len(articles)})
            if progress_callback:
                progress_callback(progress)

            # Step 4: Save articles
            progress.update(4, "Saving articles...")
            if progress_callback:
                progress_callback(progress)

            Utils.save_online_articles(search_query, articles)

            # Step 5: Embed articles
            progress.update(5, "Embedding articles into vector store...")
            if progress_callback:
                progress_callback(progress)

            articles_metadata = Utils.load_articles(Utils.ARTICLES_FILE)
            embed.embed_articles_from_files(
                articles_metadata,
                clear_existing=clear_existing
            )

            self.last_fetch_time = datetime.now()
            self.last_fetch_count = len(articles)
            result["fetched"] = True
            result["timestamp"] = self.last_fetch_time.isoformat()

            progress.update(5, f"Successfully embedded {len(articles)} articles!")
            if progress_callback:
                progress_callback(progress)

        except Exception as e:
            error_msg = str(e)
            result["error"] = error_msg
            progress.update(5, f"Error: {error_msg}")
            if progress_callback:
                progress_callback(progress)

        return result

    def get_status(self) -> Dict:
        """Get current fetcher status.

        Returns:
            Dictionary with status information.
        """
        articles = Utils.load_articles(Utils.ARTICLES_FILE)
        article_count = len(articles) if isinstance(articles, list) else 0

        return {
            "last_fetch_time": self.last_fetch_time.isoformat() if self.last_fetch_time else None,
            "last_fetch_count": self.last_fetch_count,
            "total_articles": article_count,
            "num_articles_setting": self.num_articles
        }


def auto_fetch_if_needed(
    user_query: str,
    route: str,
    progress_callback: Optional[Callable[[FetchProgress], None]] = None,
    num_articles: int = 5
) -> Dict:
    """Convenience function to auto-fetch news if needed.

    Args:
        user_query: The user's query.
        route: The classified route.
        progress_callback: Optional callback for progress updates.
        num_articles: Number of articles to fetch.

    Returns:
        Dictionary with fetch results.
    """
    fetcher = AutoNewsFetcher(num_articles=num_articles)
    return fetcher.fetch_with_progress(user_query, route, progress_callback)


def should_auto_fetch(user_query: str, route: str) -> bool:
    """Quick check if auto-fetching is recommended.

    Args:
        user_query: The user's query.
        route: The classified route.

    Returns:
        True if fetching is recommended.
    """
    return should_fetch_news_for_query(user_query, route)
