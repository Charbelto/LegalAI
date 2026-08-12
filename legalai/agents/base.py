"""Base agent class with common functionality for all agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict
from langchain_ollama import ChatOllama
import config

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - only required when GENERATION_PROVIDER=deepseek
    ChatOpenAI = None


# ---------------------------------------------------------------------------
# Runtime seed control.
#
# Repeats of the same (query, topology) pair are only informative if the model is
# actually sampling: with LEGALAI_DETERMINISTIC=1 decoding is greedy and every
# repeat returns byte-identical text, which is what collapsed the earlier
# confidence intervals to +/-0.00. For variance-bearing runs set
# LEGALAI_DETERMINISTIC=0 and pass a distinct seed per repeat; agents rebuild
# their chat model whenever the runtime seed changes.
# ---------------------------------------------------------------------------
_RUNTIME_SEED = None
_SEED_GENERATION = 0


def set_runtime_seed(seed):
    """Set the seed used by every agent's chat model. None restores the config default."""
    global _RUNTIME_SEED, _SEED_GENERATION
    if seed != _RUNTIME_SEED:
        _RUNTIME_SEED = seed
        _SEED_GENERATION += 1
    return _RUNTIME_SEED


def get_runtime_seed():
    """Return the seed currently applied to agent chat models."""
    return _RUNTIME_SEED


def build_chat_llm(model: str = None, temperature: float = 0.0, seed=None, role: str = None):
    """Construct a chat model for the currently configured GENERATION_PROVIDER.

    Single chokepoint for generation, shared by every agent (via
    BaseAgent._build_llm) and by QueryAnalyzer. Switching
    config.GENERATION_PROVIDER therefore changes every generation call in the
    system at once - nothing quietly keeps using the old backend because it
    built its own ChatOllama independently.

    `role` only matters for GENERATION_PROVIDER=local_peft, where it is the
    whole point: since the PEFT pivot each domain expert runs its OWN
    fine-tuned model rather than a shared one, so "which model" is a property
    of the caller's role. The other two providers ignore it, because there is
    only one model to hand out.

    DeepSeek speaks the OpenAI chat-completions schema, so it is called via
    langchain_openai.ChatOpenAI with a custom base_url - no bespoke HTTP code
    needed here (mirrors the judge's approach in llm_judge.py).
    """
    if config.GENERATION_PROVIDER == "local_peft":
        # Imported lazily so the ollama/deepseek paths still work on a machine
        # with no torch installed (local_models pulls in the whole HF stack).
        import local_models

        return local_models.LocalChatModel(
            role=role,
            temperature=temperature,
            seed=seed,
            max_new_tokens=config.LLM_NUM_PREDICT,
        )

    if config.GENERATION_PROVIDER == "deepseek":
        if ChatOpenAI is None:
            raise RuntimeError(
                "GENERATION_PROVIDER=deepseek requires the 'langchain-openai' package. "
                "Install it: pip install langchain-openai"
            )
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError(
                "GENERATION_PROVIDER=deepseek but DEEPSEEK_API_KEY is not set. "
                "Add it to legalai/.env."
            )
        # No seed: DeepSeek's API does not document seed support the way
        # OpenAI's does, and temperature>0 alone already gives repeats real
        # sampling variance, which is all the experiment design requires.
        return ChatOpenAI(
            model=config.DEEPSEEK_MODEL,
            base_url=config.DEEPSEEK_BASE_URL,
            api_key=config.DEEPSEEK_API_KEY,
            temperature=temperature,
            max_tokens=config.LLM_NUM_PREDICT,
        )

    return ChatOllama(
        model=model or config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
        seed=seed,
        num_predict=config.LLM_NUM_PREDICT,
        num_ctx=config.LLM_NUM_CTX,
    )


class BaseAgent(ABC):
    """Base class for all agents in the Legal AI system."""

    def __init__(self, model: str = None, temperature: float = 0.0, role: str = None):
        """Initialize the base agent with LLM configuration.

        Args:
            model: The Ollama model to use. Defaults to config.OLLAMA_MODEL.
                Ignored by the deepseek and local_peft providers.
            temperature: Temperature for LLM generation.
            role: Which model this agent should get under
                GENERATION_PROVIDER=local_peft. The three domain experts pass
                "legal" / "news" / "general_qa" and each receives its own
                separately fine-tuned model; everything else leaves this as None
                and shares the coordination model (see
                config.LOCAL_COORDINATOR_ROLE). Ignored by the other providers,
                which only ever have one model to give out.
        """
        self._model = model or config.OLLAMA_MODEL
        self._temperature = temperature
        self._role = role
        self._llm = None
        self._llm_injected = False
        self._llm_seed_generation = -1
        self.name = self.__class__.__name__

    def _build_llm(self):
        """Construct this agent's chat model, honouring the current runtime seed
        and config.GENERATION_PROVIDER (ollama, deepseek or local_peft)."""
        runtime_seed = get_runtime_seed()
        if runtime_seed is not None:
            seed = runtime_seed
        elif config.DETERMINISTIC:
            seed = config.LLM_SEED
        else:
            seed = None

        return build_chat_llm(
            model=self._model,
            temperature=0.0 if config.DETERMINISTIC else self._temperature,
            seed=seed,
            role=self._role,
        )

    @property
    def llm(self):
        """Chat model, rebuilt lazily whenever the runtime seed changes."""
        if self._llm_injected:
            return self._llm
        if self._llm is None or self._llm_seed_generation != _SEED_GENERATION:
            self._llm = self._build_llm()
            self._llm_seed_generation = _SEED_GENERATION
        return self._llm

    @llm.setter
    def llm(self, value):
        """Allow test harnesses to inject a mock model (never rebuilt afterwards)."""
        self._llm = value
        self._llm_injected = True

    @abstractmethod
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's logic.

        Args:
            state: The current state dictionary.

        Returns:
            Updated state dictionary with agent outputs.
        """
        pass

    def get_llm(self):
        """Get the LLM instance (ChatOllama or ChatOpenAI, depending on
        config.GENERATION_PROVIDER)."""
        return self.llm

    def log(self, message: str):
        """Log a message from this agent."""
        print(f"[{self.name}] {message}")

    def _record_tokens(self, state: Dict[str, Any], agent_key: str, response: Any):
        """Extract and record token counts from response metadata into state.

        Ollama (ChatOllama) reports flat keys (prompt_eval_count/eval_count).
        DeepSeek/OpenAI (ChatOpenAI) reports a nested "token_usage" dict
        instead. The local PEFT models report both flat spellings (see
        local_models.LocalChatModel). All three are handled here so switching
        GENERATION_PROVIDER doesn't silently zero out the token telemetry the
        paper relies on.
        """
        meta = getattr(response, "response_metadata", {}) or {}
        usage = meta.get("token_usage") or meta.get("usage") or {}
        prompt = (
            meta.get("prompt_eval_count", 0)
            or meta.get("prompt_tokens", 0)
            or usage.get("prompt_tokens", 0)
            or 0
        )
        completion = (
            meta.get("eval_count", 0)
            or meta.get("completion_tokens", 0)
            or usage.get("completion_tokens", 0)
            or 0
        )
        if "agent_tokens" not in state or state["agent_tokens"] is None:
            state["agent_tokens"] = {}
        state["agent_tokens"][agent_key] = {
            "prompt": prompt,
            "completion": completion
        }

        # Silent-truncation guard.
        #
        # Ollama drops tokens without error once a prompt exceeds num_ctx, which
        # would quietly remove retrieved context and make a topology look worse
        # than it is. num_ctx is an Ollama-only concept (a hosted model like
        # DeepSeek V4 Flash has its own, much larger, fixed context window), so
        # the proximity heuristic only applies to that provider.
        if (
            config.GENERATION_PROVIDER == "ollama"
            and prompt
            and prompt > 0.9 * config.LLM_NUM_CTX
        ):
            self.log(
                f"WARNING prompt_tokens={prompt} is within 10% of num_ctx="
                f"{config.LLM_NUM_CTX}; raise LEGALAI_NUM_CTX to avoid truncation"
            )
            state.setdefault("truncation_warnings", []).append(
                {"agent": agent_key, "prompt_tokens": prompt}
            )

        # The local PEFT models don't need a heuristic: local_models truncates
        # explicitly and says so, so record the actual event rather than a
        # guess at whether one happened.
        truncated_from = meta.get("input_truncated_from")
        if truncated_from:
            self.log(
                f"WARNING prompt truncated from {truncated_from} to {prompt} tokens "
                f"(LEGALAI_LOCAL_MAX_INPUT_TOKENS={config.LOCAL_MAX_INPUT_TOKENS}); "
                "oldest retrieved documents were dropped"
            )
            state.setdefault("truncation_warnings", []).append(
                {
                    "agent": agent_key,
                    "prompt_tokens": prompt,
                    "truncated_from": int(truncated_from),
                }
            )

