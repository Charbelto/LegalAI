"""Test bootstrap.

These tests must run without Ollama, ChromaDB, or a network connection, so the
agent modules are loaded directly by path (bypassing agents/__init__.py, which
pulls in the retrieval stack) and heavy third-party imports are stubbed only when
they are genuinely unavailable. On a normal dev machine the real packages are
imported instead.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub_if_missing(name: str, builder):
    try:
        __import__(name)
        return False
    except Exception:
        for mod_name, mod in builder().items():
            sys.modules.setdefault(mod_name, mod)
        return True


def _langchain_core_stub():
    package = types.ModuleType("langchain_core")
    package.__path__ = []
    prompts = types.ModuleType("langchain_core.prompts")

    class ChatPromptTemplate:
        def __init__(self, template=""):
            self.template = template

        @classmethod
        def from_template(cls, template):
            return cls(template)

        def __or__(self, other):  # pragma: no cover - tests inject FakePrompt
            raise RuntimeError("Tests must inject a fake prompt/chain, not pipe a stub.")

    prompts.ChatPromptTemplate = ChatPromptTemplate
    package.prompts = prompts
    return {"langchain_core": package, "langchain_core.prompts": prompts}


def _langchain_ollama_stub():
    module = types.ModuleType("langchain_ollama")

    class ChatOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, *args, **kwargs):
            raise RuntimeError("No LLM is available in tests.")

    module.ChatOllama = ChatOllama
    return {"langchain_ollama": module}


def _utils_stub():
    module = types.ModuleType("utils")
    module.get_current_date = lambda: "2026-07-25"
    return {"utils": module}


_stub_if_missing("langchain_core.prompts", _langchain_core_stub)
_stub_if_missing("langchain_ollama", _langchain_ollama_stub)
_stub_if_missing("utils", _utils_stub)

# Register a lightweight 'agents' package pointing at the real directory so that
# submodules can be imported without executing agents/__init__.py.
if "agents" not in sys.modules:
    agents_pkg = types.ModuleType("agents")
    agents_pkg.__path__ = [str(ROOT / "agents")]
    sys.modules["agents"] = agents_pkg

# Same for 'graph': graph/__init__.py imports workflow.py, which needs langgraph.
if "graph" not in sys.modules:
    graph_pkg = types.ModuleType("graph")
    graph_pkg.__path__ = [str(ROOT / "graph")]
    sys.modules["graph"] = graph_pkg


def _load(module_name: str, relative_path: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class MockResponse:
    """Stands in for an LLM response object."""

    def __init__(self, content):
        self.content = content
        self.response_metadata = {"prompt_eval_count": 100, "eval_count": 50}


class FakeChain:
    """Records the variables an agent passes to its prompt chain."""

    def __init__(self, content="MOCK AGGREGATED ANSWER"):
        self.content = content
        self.calls = []

    def invoke(self, variables):
        self.calls.append(variables)
        return MockResponse(self.content)


class FakePrompt:
    """Replaces ChatPromptTemplate so `prompt | llm` never needs real langchain."""

    def __init__(self, chain):
        self.chain = chain

    def __or__(self, _other):
        return self.chain


@pytest.fixture(scope="session")
def config_module():
    return _load("config", "config.py")


@pytest.fixture(scope="session")
def base_module():
    _load("config", "config.py")
    return _load("agents.base", "agents/base.py")


@pytest.fixture()
def aggregator_agent(base_module):
    module = _load("agents.aggregator", "agents/aggregator.py")
    agent = module.AggregatorAgent()
    chain = FakeChain()
    agent.prompt = FakePrompt(chain)
    agent.llm = object()  # never used: FakePrompt short-circuits the pipe
    return agent, chain


@pytest.fixture()
def response_agent(base_module):
    module = _load("agents.response", "agents/response.py")
    agent = module.ResponseAgent()
    chain = FakeChain("MOCK POLISHED ANSWER")
    agent.prompt = FakePrompt(chain)
    agent.llm = object()
    return agent, chain


@pytest.fixture(scope="session")
def analyze_module():
    return _load("analyze_results", "analyze_results.py")
