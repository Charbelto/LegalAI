"""
Web scraping module for fetching news articles from online sources.
Uses DuckDuckGo search to find articles and BeautifulSoup to scrape content.
"""

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
import time
from typing import List, Dict, Optional


def search_news(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Search for news articles using DuckDuckGo.

    Args:
        query: Search query string
        num_results: Number of results to return (default: 5)

    Returns:
        List of dictionaries with 'title', 'url', and 'source' keys
    """
    results = []

    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=num_results)
            search_results_list = list(search_results)

            for result in search_results_list:
                results.append({
                    'title': result.get('title', 'Untitled'),
                    'url': result.get('href', ''),
                    'source': result.get('source', 'Unknown')
                })

        print(f"Found {len(results)} articles for query: '{query}'")

    except Exception as e:
        print(f"Error searching for news: {e}")

    return results


def scrape_article(url: str, timeout: int = 10) -> Optional[str]:
    """
    Scrape article content from a URL.

    Args:
        url: The URL to scrape
        timeout: Request timeout in seconds

    Returns:
        Extracted text content or None if failed
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(['script', 'style', 'nav', 'footer', 'header']):
            script.decompose()

        # Try common article content selectors
        content_selectors = [
            'article',
            '[role="main"]',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.content',
            'main',
            '.story-body',
            '.article-body',
            '#article-content',
            '.news-content'
        ]

        text_content = ""

        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    text = element.get_text(separator='\n', strip=True)
                    if len(text) > len(text_content):
                        text_content = text

                if len(text_content) > 500:  # Found substantial content
                    break

        # Fallback: extract all paragraphs if no specific container found
        if not text_content or len(text_content) < 500:
            paragraphs = soup.find_all('p')
            text_content = '\n\n'.join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50)

        # Clean up the text
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text_content = '\n'.join(chunk for chunk in chunks if chunk)

        if len(text_content) < 100:
            print(f"Warning: Limited content extracted from {url} ({len(text_content)} chars)")
            return None

        print(f"Successfully scraped {len(text_content)} characters from {url}")
        return text_content

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
    except Exception as e:
        print(f"Error scraping {url}: {e}")

    return None


def fetch_and_scrape_articles(query: str, num_results: int = 5, delay: float = 1.0) -> List[Dict]:
    """
    Search for articles and scrape their content.

    Args:
        query: Search query string
        num_results: Number of articles to fetch
        delay: Delay between requests in seconds (be polite to servers)

    Returns:
        List of dictionaries containing article data with 'title', 'url', 'source', and 'content'
    """
    search_results = search_news(query, num_results)
    articles = []

    for result in search_results:
        print(f"Scraping: {result['title'][:80]}...")

        content = scrape_article(result['url'])

        if content:
            articles.append({
                'title': result['title'],
                'url': result['url'],
                'source': result['source'],
                'content': content
            })

        # Be polite to servers
        if delay > 0:
            time.sleep(delay)

    print(f"Successfully scraped {len(articles)} out of {len(search_results)} articles")
    return articles


if __name__ == "__main__":
    # Test the scraper
    test_query = "artificial intelligence regulation 2024"
    print(f"\nTesting scraper with query: '{test_query}'\n")

    articles = fetch_and_scrape_articles(test_query, num_results=3)

    print("\n--- Scraped Articles ---")
    for i, article in enumerate(articles, 1):
        print(f"\n{i}. {article['title']}")
        print(f"   Source: {article['source']}")
        print(f"   URL: {article['url']}")
        print(f"   Content length: {len(article['content'])} characters")
        print(f"   Preview: {article['content'][:200]}...")
