"""Tests for the legal expert's live EU-source search.

Network-free: the search and fetch layers are stubbed, because the properties
worth guarding are policy decisions rather than connectivity - is it off by
default, does it refuse off-domain results, and does it degrade to [] instead of
raising when the network misbehaves.

The off-by-default test is the load-bearing one. A live lookup inside a benchmark
run makes the legal expert's context depend on what the web returned that minute,
so two runs of the same query stop being comparable and any between-topology
difference is partly a fetch difference.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def eurlex():
    if "config" not in sys.modules:
        spec = importlib.util.spec_from_file_location("config", ROOT / "config.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["config"] = module
        spec.loader.exec_module(module)
    if "eurlex_search" in sys.modules:
        return sys.modules["eurlex_search"]
    spec = importlib.util.spec_from_file_location("eurlex_search", ROOT / "eurlex_search.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["eurlex_search"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def cfg(eurlex):
    return sys.modules["config"]


def test_live_search_is_off_by_default(cfg):
    """A time-varying context would make benchmark runs incomparable."""
    assert cfg.EURLEX_LIVE_SEARCH_ENABLED is False


def test_returns_nothing_while_disabled(eurlex, cfg, monkeypatch):
    """Disabled must short-circuit before any network call is attempted."""
    monkeypatch.setattr(cfg, "EURLEX_LIVE_SEARCH_ENABLED", False)

    def _explode(*_args, **_kwargs):  # pragma: no cover - must never run
        raise AssertionError("search was attempted while disabled")

    monkeypatch.setattr(eurlex, "_search_results", _explode)
    assert eurlex.search_recent_legal_sources("high-risk AI system") == []


def test_blank_query_returns_nothing(eurlex, cfg, monkeypatch):
    monkeypatch.setattr(cfg, "EURLEX_LIVE_SEARCH_ENABLED", True)
    assert eurlex.search_recent_legal_sources("   ") == []


def test_off_domain_results_are_rejected(eurlex, cfg, monkeypatch):
    """Only official EU hosts may reach the legal expert's context.

    A general web result would reintroduce the source-quality problem the legal
    agent's abstention rule exists to prevent, and the site: filter passed to the
    search backend is a hint rather than a guarantee.
    """
    monkeypatch.setattr(cfg, "EURLEX_LIVE_SEARCH_ENABLED", True)

    class _FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def text(self, _query, max_results=None):
            return [
                {"href": "https://blog.example.com/ai-act", "title": "Blog take", "body": "..."},
                {"href": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=x",
                 "title": "Regulation", "body": "Official text"},
                {"href": "https://digital-strategy.ec.europa.eu/en/policies/x",
                 "title": "Guidelines", "body": "Commission guidance"},
            ]

    fake_module = type(sys)("ddgs")
    fake_module.DDGS = _FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_module)
    # Skip page fetching; the snippet fallback is enough to exercise filtering.
    monkeypatch.setattr(eurlex, "_fetch_page_text", lambda *_a, **_k: "")

    docs = eurlex.search_recent_legal_sources("obligations", max_results=5)
    sources = [d["metadata"]["source"] for d in docs]

    assert len(docs) == 2, sources
    assert not any("example.com" in s for s in sources)
    assert all(
        any(domain in s for domain in eurlex.ALLOWED_DOMAINS) for s in sources
    ), sources


def test_search_failure_degrades_to_empty(eurlex, cfg, monkeypatch):
    """A network failure must yield no documents, never an exception.

    The legal agent treats an empty result as "no additional live support", which
    its abstention rule already handles. Raising here would abort a whole
    benchmark run over a transient lookup.
    """
    monkeypatch.setattr(cfg, "EURLEX_LIVE_SEARCH_ENABLED", True)

    class _BrokenDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def text(self, *_a, **_k):
            raise RuntimeError("network down")

    fake_module = type(sys)("ddgs")
    fake_module.DDGS = _BrokenDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_module)

    assert eurlex.search_recent_legal_sources("anything") == []


def test_documents_are_retrieval_shaped(eurlex, cfg, monkeypatch):
    """Output must match the retrieval agent's document shape.

    The legal agent formats live and static sources through one code path, so a
    different shape here would surface as missing context rather than an error.
    """
    monkeypatch.setattr(cfg, "EURLEX_LIVE_SEARCH_ENABLED", True)

    class _OneResult:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def text(self, *_a, **_k):
            return [{"href": "https://eur-lex.europa.eu/x", "title": "T", "body": "snippet"}]

    fake_module = type(sys)("ddgs")
    fake_module.DDGS = _OneResult
    monkeypatch.setitem(sys.modules, "ddgs", fake_module)
    monkeypatch.setattr(eurlex, "_fetch_page_text", lambda *_a, **_k: "Full page body.")

    docs = eurlex.search_recent_legal_sources("q", max_results=1)
    assert len(docs) == 1
    doc = docs[0]
    assert set(doc) == {"page_content", "metadata"}
    assert "Full page body." in doc["page_content"]
    assert doc["metadata"]["origin"] == "eurlex_live"
    assert doc["metadata"]["live"] is True
    assert doc["metadata"]["source"] == "https://eur-lex.europa.eu/x"
