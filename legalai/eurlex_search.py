"""Live search over EUR-Lex and the Commission's digital-strategy pages.

Why this exists
---------------
The legal expert's authoritative corpus is the static EU AI Act text embedded in
ChromaDB (351 chunks). That text is fixed at the version that was ingested, so
the agent cannot see anything published since: delegated acts, corrigenda,
guidelines, harmonised-standards references, or Commission implementing
decisions. This module gives the legal expert the same kind of recency channel
the news expert already has, but restricted to official EU sources rather than
general web news.

Why it is OFF during the benchmark
----------------------------------
`config.EURLEX_LIVE_SEARCH_ENABLED` defaults to 0 and `benchmark.py` forces it
to 0, exactly as it forces the news agent's live fetch off. A network-dependent,
time-varying context would make two runs of the same query incomparable, and any
measured difference between topologies would be partly an artefact of what the
web happened to return in that minute. Recency is valuable interactively and
poison in a controlled experiment.

Sources searched (both official):
  * eur-lex.europa.eu           - consolidated legislation, OJ publications
  * digital-strategy.ec.europa.eu - Commission AI-policy pages and guidance

Retrieved text is returned as plain dicts shaped like the retrieval agent's
documents (page_content + metadata), so the legal agent can format them through
the same `_format_context` path with no special casing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import config

# Official hosts only. A general web search would reintroduce exactly the
# source-quality problem the legal agent's abstention rule exists to prevent.
ALLOWED_DOMAINS = (
    "eur-lex.europa.eu",
    "digital-strategy.ec.europa.eu",
)

_WHITESPACE = re.compile(r"\s+")
_MAX_CHARS_PER_DOC = 2500


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", str(text or "")).strip()


def _search_results(query: str, max_results: int) -> List[Dict[str, str]]:
    """Run a site-restricted search. Returns [] on any failure.

    Uses the `ddgs` package already in requirements.txt (the news scraper's
    dependency), so this adds no new third-party service.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:  # older package name
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError:
            print("[eurlex] ddgs not installed; live legal search unavailable")
            return []

    site_filter = " OR ".join(f"site:{d}" for d in ALLOWED_DOMAINS)
    full_query = f"{query} ({site_filter})"

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(full_query, max_results=max_results * 3))
    except Exception as exc:
        print(f"[eurlex] search failed: {exc}")
        return []

    results = []
    for item in raw:
        url = str(item.get("href") or item.get("url") or "")
        if not any(domain in url for domain in ALLOWED_DOMAINS):
            # Belt and braces: the site: filter is a hint, not a guarantee.
            continue
        results.append(
            {
                "url": url,
                "title": _clean(item.get("title", "")),
                "snippet": _clean(item.get("body") or item.get("snippet") or ""),
            }
        )
        if len(results) >= max_results:
            break
    return results


def _fetch_page_text(url: str, timeout: float) -> str:
    """Fetch and strip one page. Returns "" on any failure."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return ""

    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "LegalAI-research/1.0 (academic use)"},
        )
        response.raise_for_status()
    except Exception as exc:
        print(f"[eurlex] fetch failed for {url}: {exc}")
        return ""

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        return _clean(soup.get_text(" "))[:_MAX_CHARS_PER_DOC]
    except Exception as exc:
        print(f"[eurlex] parse failed for {url}: {exc}")
        return ""


def search_recent_legal_sources(query: str, max_results: int = None) -> List[Dict[str, Any]]:
    """Search official EU legal sources for material relevant to `query`.

    Returns a list of retrieval-agent-shaped documents. Returns [] when the
    feature is disabled, when the network is unavailable, or when nothing
    on-domain matches - the legal agent treats an empty result as "no additional
    live support", which its abstention rule already handles correctly.
    """
    if not config.EURLEX_LIVE_SEARCH_ENABLED:
        return []
    if not str(query or "").strip():
        return []

    limit = int(max_results or config.EURLEX_MAX_RESULTS)
    hits = _search_results(query, limit)
    if not hits:
        return []

    documents: List[Dict[str, Any]] = []
    for hit in hits:
        body = _fetch_page_text(hit["url"], config.EURLEX_TIMEOUT_S)
        # Fall back to the search snippet rather than dropping the source: a
        # title plus snippet from EUR-Lex is still an authoritative pointer.
        content = body or hit["snippet"]
        if not content:
            continue
        documents.append(
            {
                "page_content": f"{hit['title']}\n\n{content}",
                "metadata": {
                    "name": hit["title"] or hit["url"],
                    "source": hit["url"],
                    "origin": "eurlex_live",
                    "live": True,
                },
            }
        )

    print(f"[eurlex] {len(documents)} live legal source(s) for: {query[:60]}")
    return documents
