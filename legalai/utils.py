import json as JS
import os as OS
import hashlib
from datetime import datetime
from typing import List, Dict, Optional
import scraper

ARTICLES_FILE ='articles.json'
ARTICLES_FOLDER = 'articles'
DB_FOLDER = 'chroma_storage'
DATA_FOLDER = 'data'
# EU AI Act source, in preference order (embed.py tries each until one yields a PDF).
#
# The original single URL was the Parliament's adopted-text PDF
# (TA-9-2024-0138_EN.pdf). It now returns HTTP 202 with an empty body, and
# eur-lex.europa.eu answers non-browser clients with an AWS WAF challenge
# (x-amzn-waf-action: challenge), so neither can be fetched programmatically any
# more - which is how this project's corpus came to be empty.
#
# The primary source below is Cellar, the EU Publications Office's
# machine-access repository, which is built for programmatic retrieval and
# requires an explicit Accept-Language. It serves CELEX 32024R1689, i.e.
# Regulation (EU) 2024/1689 as published in the Official Journal - the
# definitive text, and better provenance than the pre-final adopted-text stage
# the old URL pointed at.
EUROPEAN_ACT_SOURCES = [
    {
        "url": "http://publications.europa.eu/resource/celex/32024R1689",
        "headers": {"Accept": "application/pdf", "Accept-Language": "eng"},
        "label": "Cellar CELEX:32024R1689 (OJ text, Regulation (EU) 2024/1689)",
    },
    {
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689",
        "headers": {"Accept": "application/pdf"},
        "label": "EUR-Lex CELEX:32024R1689 (WAF-protected; usually fails headless)",
    },
    {
        "url": "https://www.europarl.europa.eu/doceo/document/TA-9-2024-0138_EN.pdf",
        "headers": {},
        "label": "European Parliament TA-9-2024-0138 (original source, now returns 202)",
    },
]

# Kept for backwards compatibility: anything still reading a single URL gets the
# primary source.
EUROPEAN_ACT_URL = EUROPEAN_ACT_SOURCES[0]["url"]


def fetch_news_articles(query: str, num_results: int = 5) -> List[Dict]:
    """
    Fetch news articles from online sources based on a search query.

    Args:
        query: Search query string
        num_results: Number of articles to fetch (default: 5)

    Returns:
        List of article dictionaries with 'title', 'url', 'source', and 'content'
    """
    print(f"Fetching news for query: '{query}'")
    articles = scraper.fetch_and_scrape_articles(query, num_results)
    return articles


def save_online_articles(query: str, articles: List[Dict]) -> None:
    """
    Save scraped online articles to local storage and update articles.json.

    Args:
        query: The search query used to fetch articles
        articles: List of article dictionaries from fetch_news_articles()
    """
    # Ensure directories exist
    if not OS.path.exists(ARTICLES_FOLDER):
        OS.makedirs(ARTICLES_FOLDER, exist_ok=True)
        print(f"Created directory: {ARTICLES_FOLDER}")

    # Load existing articles data
    articles_data = load_articles(ARTICLES_FILE)

    # Ensure articles_data is a list
    if not isinstance(articles_data, list):
        articles_data = []

    saved_count = 0
    for article in articles:
        # Create safe filename from title
        safe_title = "".join(c for c in article['title'][:50] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = safe_title.replace(' ', '_')
        url_hash = hashlib.sha1(article['url'].encode('utf-8')).hexdigest()[:8]
        filename = f"{safe_title}_{url_hash}.txt"
        filepath = OS.path.join(ARTICLES_FOLDER, filename)

        # Save content to file
        save_article_content(filepath, article['content'])

        # Add to articles metadata
        article_entry = {
            'title': article['title'],
            'url': article['url'],
            'source': article['source'],
            'query': query,
            'file': filepath,
            'content_length': len(article['content']),
            'fetched_at': datetime.now().isoformat(),
        }

        # Avoid duplicates based on URL
        if not any(a.get('url') == article['url'] for a in articles_data):
            articles_data.append(article_entry)
            saved_count += 1

    # Save updated articles metadata
    save_articles(ARTICLES_FILE, articles_data)
    print(f"Saved {saved_count} new articles to local storage")

def get_current_date() -> str:
    """Get the current date in a formatted string.

    Returns:
        Current date as string in YYYY-MM-DD format.
    """
    return datetime.now().strftime("%Y-%m-%d")


def get_db_documents(limit: int = 50) -> List[Dict]:
    """Get documents from ChromaDB with their metadata.

    Args:
        limit: Maximum number of documents to return.

    Returns:
        List of document dictionaries with source and preview.
    """
    import config

    # Avoid noisy connection errors before the vector store exists.
    if not OS.path.exists(config.CHROMA_PERSIST_DIRECTORY):
        return []

    try:
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

        db = Chroma(
            persist_directory=config.CHROMA_PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name=config.CHROMA_COLLECTION_NAME,
        )

        # Get all documents
        results = db._collection.get(limit=limit)

        documents = []
        for i, doc in enumerate(results.get('documents', [])):
            metadata = results.get('metadatas', [{}])[i] if results.get('metadatas') else {}
            source = metadata.get('source', 'Unknown')
            documents.append({
                'source': source,
                'preview': doc[:100] + "..." if len(doc) > 100 else doc,
                'metadata': metadata
            })

        return documents
    except Exception as e:
        print(f"Error getting documents: {e}")
        return []


def load_articles_metadata() -> List[Dict]:
    """Load articles metadata with fetched_at timestamps.

    Returns:
        List of article metadata dictionaries.
    """
    articles = load_articles(ARTICLES_FILE)
    if not isinstance(articles, list):
        return []

    # Add fetched_at if missing (use current time as fallback)
    for article in articles:
        if 'fetched_at' not in article:
            article['fetched_at'] = datetime.now().isoformat()

    return articles


# load articles data from file_name (.json)
def load_articles(file_name) -> list:
    result = []
    if OS.path.exists(file_name):
        with open(file_name, 'r', encoding='utf-8') as file:
            try:
                loaded = JS.load(file)
                # Ensure result is a list
                if isinstance(loaded, list):
                    result = loaded
                else:
                    print(f"Warning: Loaded data is not a list. Resetting file.")
                    # Reset the file to an empty list
                    with open(file_name, 'w', encoding='utf-8') as f:
                        JS.dump([], f)
            except JS.JSONDecodeError:
                print("File exists but is not valid JSON. Resetting to empty list.")
                # Reset the file to an empty list
                with open(file_name, 'w', encoding='utf-8') as f:
                    JS.dump([], f)
    else:
        with open(file_name, 'w', encoding='utf-8') as file:
            JS.dump([], file)
        print(f"File '{file_name}' did not exist and was created.")
        OS.makedirs('articles', exist_ok=True)
        print("'articles' directory was created (or already existed)")

    return result 

# save articles data to file_name (.json)
def save_articles(file_name, data):
    try:
        with open(file_name, 'w', encoding='utf-8') as file:
            JS.dump(data, file, indent=4, ensure_ascii=False)
            print(f"Data successfully saved to '{file_name}'.")    
    except Exception as e:
        print(f"Error: trying to save articles data [{e}]")

# save articles content to individual files
def save_article_content(file_name, content):
    try:
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(content)
    except IOError as e:
        print(f"An IOError occurred: {e.strerror}")
    except Exception as e:
        print(f"Error: {e}")
    else:
        print(f"Content successfully written to '{file_name}'.")

# load article body text from file
def load_article_content(file_name):
    result = ''
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            result = file.read()
    except Exception as e:
        print(f"An unexpected error occurred while reading content file '{file_name}': {e}")

    return result


# Multi-agent helper utilities
def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to maximum length with ellipsis.

    Args:
        text: The text to truncate.
        max_length: Maximum length allowed.

    Returns:
        Truncated text with ellipsis if needed.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_agent_output(agent_name: str, output: str, max_length: int = 200) -> str:
    """Format agent output for display/logging.

    Args:
        agent_name: Name of the agent.
        output: The agent's output.
        max_length: Maximum length to display.

    Returns:
        Formatted string.
    """
    truncated = truncate_text(output, max_length)
    return f"[{agent_name}] {truncated}"


def safe_get(dictionary: dict, key: str, default=None):
    """Safely get a value from a dictionary.

    Args:
        dictionary: The dictionary to query.
        key: The key to look up.
        default: Default value if key not found.

    Returns:
        The value or default.
    """
    return dictionary.get(key, default) if isinstance(dictionary, dict) else default


# Query analysis helpers
def generate_search_query(user_prompt: str) -> str:
    """Generate an optimized search query from user prompt.

    This is a convenience function that uses QueryAnalyzer.

    Args:
        user_prompt: The user's natural language question.

    Returns:
        Optimized search query string.
    """
    from query_analyzer import QueryAnalyzer
    analyzer = QueryAnalyzer()
    return analyzer.generate_search_query(user_prompt)


def should_fetch_news(user_query: str, route: str) -> bool:
    """Check if news fetching is recommended for this query.

    This is a convenience function that uses QueryAnalyzer.

    Args:
        user_query: The user's query.
        route: The classified route (legal/news/general).

    Returns:
        True if news fetching is recommended.
    """
    from query_analyzer import should_fetch_news_for_query
    return should_fetch_news_for_query(user_query, route)


def get_db_document_count() -> int:
    """Get the number of documents currently in ChromaDB.

    Returns:
        Number of documents, or 0 if database doesn't exist.
    """
    import config

    # Avoid noisy connection errors before the vector store exists.
    if not OS.path.exists(config.CHROMA_PERSIST_DIRECTORY):
        return 0

    try:
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

        db = Chroma(
            persist_directory=config.CHROMA_PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name=config.CHROMA_COLLECTION_NAME,
        )

        return db._collection.count()
    except Exception as e:
        print(f"Error getting document count: {e}")
        return 0