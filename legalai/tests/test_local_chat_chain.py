"""Integration test: does `prompt | LocalChatModel` actually work?

Every agent in this project builds its chain as `self.prompt | self.llm`, where
prompt is a real langchain ChatPromptTemplate. LocalChatModel is NOT a langchain
Runnable, so what that `|` produces depends on langchain's coercion rules rather
than on anything in our code: `Runnable.__or__` is tried first and only falls
back to our `__ror__` if it declines. If langchain coerces LocalChatModel into a
RunnableLambda instead, the chain still has to pass the rendered prompt value
through correctly and return an object exposing `.content` and
`.response_metadata`, because that is what every agent reads.

Getting this wrong does not fail loudly - it fails as a TypeError deep inside the
first benchmark run, or worse, as a silently mis-rendered prompt. So it is worth
an explicit test, with the heavy model layer stubbed so no GPU or weights are
needed.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch", reason="local PEFT stack not installed")
pytest.importorskip("langchain_core.prompts")


@pytest.fixture(scope="module")
def local_models_module():
    if "config" not in sys.modules:
        spec = importlib.util.spec_from_file_location("config", ROOT / "config.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["config"] = module
        spec.loader.exec_module(module)
    if "local_models" in sys.modules:
        return sys.modules["local_models"]
    spec = importlib.util.spec_from_file_location("local_models", ROOT / "local_models.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["local_models"] = module
    spec.loader.exec_module(module)
    return module


class _StubTokenizer:
    """Tokenises by whitespace; enough to exercise encode/decode round trips."""

    chat_template = "stub"
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.last_rendered = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.last_rendered = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return self.last_rendered

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        ids = [2 + (i % 50) for i, _ in enumerate(text.split())] or [2]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=True):
        return "STUB GENERATED ANSWER"


class _StubModel:
    """Returns the prompt plus a fixed number of new tokens."""

    def __init__(self, new_tokens=7):
        self.new_tokens = new_tokens
        self.calls = []
        self._param = torch.zeros(1)

    def parameters(self):
        yield self._param

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.calls.append({"input_len": int(input_ids.shape[-1]), "kwargs": kwargs})
        extra = torch.arange(100, 100 + self.new_tokens).unsqueeze(0)
        return torch.cat([input_ids, extra], dim=-1)


@pytest.fixture()
def stubbed_chat(local_models_module, monkeypatch):
    """A LocalChatModel whose weights are stubs, so no GPU is touched."""
    tokenizer = _StubTokenizer()
    model = _StubModel()
    loaded = local_models_module._LoadedModel(
        key="stub::peft",
        model=model,
        tokenizer=tokenizer,
        base_model_id="stub/model",
        adapter_dir=None,
    )
    monkeypatch.setattr(local_models_module, "get_loaded_model", lambda role: loaded)
    chat = local_models_module.LocalChatModel(role="legal", temperature=0.0, max_new_tokens=32)
    return chat, tokenizer, model


def test_prompt_pipe_model_produces_a_working_chain(stubbed_chat):
    """`ChatPromptTemplate | LocalChatModel` must render and generate correctly."""
    from langchain_core.prompts import ChatPromptTemplate

    chat, tokenizer, model = stubbed_chat
    prompt = ChatPromptTemplate.from_template(
        "You are a legal expert.\n\nContext:\n{context}\n\nQuestion: {query}"
    )
    chain = prompt | chat

    response = chain.invoke({"context": "Article 6 applies.", "query": "Is it high-risk?"})

    # The agents only ever read these two attributes.
    assert response.content == "STUB GENERATED ANSWER"
    assert response.response_metadata["completion_tokens"] == 7

    # The template variables must actually have reached the model, not been
    # passed through as an unrendered template or a repr of the dict.
    assert model.calls, "the model was never invoked"
    assert "Article 6 applies." in tokenizer.last_rendered
    assert "Is it high-risk?" in tokenizer.last_rendered
    assert "{context}" not in tokenizer.last_rendered


def test_token_counts_are_the_keys_record_tokens_reads(stubbed_chat):
    """Telemetry must land under the keys BaseAgent._record_tokens looks for.

    _record_tokens reads prompt_eval_count / prompt_tokens and eval_count /
    completion_tokens. If the local provider reported a different spelling, every
    token, cost and quality-per-1k-token number in the paper would be zero while
    the run looked healthy.
    """
    chat, _tokenizer, _model = stubbed_chat
    meta = chat.invoke("A short question").response_metadata

    assert meta["prompt_eval_count"] == meta["prompt_tokens"] > 0
    assert meta["eval_count"] == meta["completion_tokens"] == 7
    assert meta["arm"] in {"peft", "base"}
    assert meta["role"] == "legal"


def test_overlong_prompts_truncate_from_the_left_and_say_so(stubbed_chat, monkeypatch):
    """Truncation must keep the query end of the prompt and be reported.

    Left truncation is deliberate: the query, the chat-history tail and the
    generation prompt all sit at the END of the rendered template, so dropping the
    head sheds the oldest retrieved documents first. Reporting it is what lets
    truncation_warnings surface a context that was silently shortened.
    """
    import config

    chat, _tokenizer, model = stubbed_chat
    monkeypatch.setattr(config, "LOCAL_MAX_INPUT_TOKENS", 16)

    long_prompt = " ".join(f"word{i}" for i in range(200))
    meta = chat.invoke(long_prompt).response_metadata

    assert meta["prompt_tokens"] == 16
    assert meta["input_truncated_from"] > 16
    assert model.calls[-1]["input_len"] == 16


def test_greedy_when_temperature_is_zero(stubbed_chat):
    """temperature=0 must disable sampling, not pass 0.0 to a sampler."""
    chat, _tokenizer, model = stubbed_chat
    chat.invoke("question")
    kwargs = model.calls[-1]["kwargs"]

    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs


@pytest.mark.parametrize(
    "role", ["aggregator", "validator", "response", "router", "planner", "memory", None]
)
def test_coordination_node_via_chat_model_has_no_adapter(
    local_models_module, monkeypatch, role
):
    """Constructing LocalChatModel for a coordination role must NOT load an adapter.

    Regression test for a bug that survived an earlier fix. get_loaded_model was
    corrected to derive the adapter decision from the REQUESTED role, so that
    get_loaded_model("aggregator") correctly declined the adapter - and a test
    asserting exactly that passed. But LocalChatModel.__init__ called
    get_loaded_model(self.role), where self.role had already been resolved from
    "aggregator" to "general_qa", so in practice every coordination node ran with
    the general expert's LoRA attached.

    The consequence was not subtle: the aggregator emitted word-salad at every
    prompt length, which was misattributed in turn to adapter damage, a
    train/serve context mismatch, and missing repetition control. The test that
    should have caught it bypassed the constructor, which is the only place the
    defect lived - hence this one goes through the constructor for every
    coordination role.
    """
    import config

    monkeypatch.setattr(config, "LOCAL_PEFT_USE_ADAPTERS", True)
    monkeypatch.setattr(config, "LOCAL_COORDINATOR_USE_ADAPTER", False)

    requested = {}

    def _capture(asked_for):
        requested["role"] = asked_for
        return local_models_module._LoadedModel(
            key="stub", model=_StubModel(), tokenizer=_StubTokenizer(),
            base_model_id="stub/model", adapter_dir=None,
        )

    monkeypatch.setattr(local_models_module, "get_loaded_model", _capture)
    local_models_module.LocalChatModel(role=role, temperature=0.0)

    # The unresolved role must reach get_loaded_model, or it cannot tell a
    # coordination node apart from the expert that shares its weights.
    assert requested["role"] == role
    assert local_models_module.uses_adapter(requested["role"]) is False


def test_expert_via_chat_model_does_get_its_adapter(local_models_module, monkeypatch):
    """The mirror case: an expert must still receive its adapter."""
    import config

    monkeypatch.setattr(config, "LOCAL_PEFT_USE_ADAPTERS", True)
    requested = {}

    def _capture(asked_for):
        requested["role"] = asked_for
        return local_models_module._LoadedModel(
            key="stub", model=_StubModel(), tokenizer=_StubTokenizer(),
            base_model_id="stub/model", adapter_dir="adapters/legal",
        )

    monkeypatch.setattr(local_models_module, "get_loaded_model", _capture)
    local_models_module.LocalChatModel(role="legal", temperature=0.0)

    assert requested["role"] == "legal"
    assert local_models_module.uses_adapter("legal") is True


def test_unload_all_releases_weights_even_with_a_live_reference(
    local_models_module, monkeypatch
):
    """unload_all() must not depend on the caller having dropped its reference.

    Regression test. validate_adapters.py loads six models in sequence (two arms
    x three roles) and called unload_all() while its own LocalChatModel was still
    in scope. Clearing the registry alone left the weights referenced, so
    empty_cache() freed nothing and the six loads accumulated far past 8GB.
    Clearing the entry's model/tokenizer makes the release independent of caller
    discipline.
    """
    tokenizer, model = _StubTokenizer(), _StubModel()
    entry = local_models_module._LoadedModel(
        key="stub::peft", model=model, tokenizer=tokenizer,
        base_model_id="stub/model", adapter_dir=None,
    )
    # _MODELS holds a POOL (list) of replicas per key, not a bare instance -
    # see config.LOCAL_MODEL_POOL_SIZE / local_models.get_loaded_model.
    monkeypatch.setattr(local_models_module, "_MODELS", {"stub::peft": [entry]})

    # A caller still holding the chat model, exactly as validate_adapters did.
    monkeypatch.setattr(local_models_module, "get_loaded_model", lambda role: entry)
    chat = local_models_module.LocalChatModel(role="legal", temperature=0.0)

    local_models_module.unload_all()

    assert local_models_module._MODELS == {}
    # The weights must be released through the still-live reference too.
    assert chat._loaded.model is None
    assert entry.tokenizer is None


def test_sampling_when_temperature_is_positive(local_models_module, monkeypatch):
    """A positive temperature must actually sample, or repeats carry no variance."""
    tokenizer, model = _StubTokenizer(), _StubModel()
    loaded = local_models_module._LoadedModel(
        key="stub::peft", model=model, tokenizer=tokenizer,
        base_model_id="stub/model", adapter_dir=None,
    )
    monkeypatch.setattr(local_models_module, "get_loaded_model", lambda role: loaded)

    chat = local_models_module.LocalChatModel(role="news", temperature=0.3, seed=1234)
    chat.invoke("question")
    kwargs = model.calls[-1]["kwargs"]

    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == pytest.approx(0.3)
