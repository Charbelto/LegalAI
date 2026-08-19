"""Per-role local generation models for GENERATION_PROVIDER=local_peft.

What this module exists to do
----------------------------
Before the PEFT pivot every agent in the graph shared one chat model (local
qwen2.5 via Ollama, or hosted DeepSeek). The pivot replaces that with three
*different* small open-weight models, one per domain expert, each 4-bit
quantised and each carrying its own LoRA adapter trained on its own domain
dataset. So "which model am I" becomes a property of the agent's role rather
than a global setting, and this module is the registry that resolves it.

Design points that matter for the experiment
--------------------------------------------
* **One load per (base model, arm), process-wide.** Three 3-4B models at 4 bits
  only just fit in 8GB; reloading per agent instance would blow VRAM instantly.
  ``get_chat_model`` caches on the resolved model key and hands the same
  underlying weights to every agent that asks for that role.

* **Token counts are real, not estimated.** ``LocalChatModel`` reports the
  actual tokenised prompt length and the actual number of generated tokens in
  ``response_metadata``, under the same keys ``BaseAgent._record_tokens``
  already reads for the other providers. The paper's cost and
  quality-per-1k-token metrics are computed from these, so an estimate here
  would silently become a published number.

* **Prompt truncation is explicit and reported.** Ollama has ``num_ctx`` and
  drops tokens past it without complaint; a raw ``transformers`` model has no
  equivalent guard and will run past its trained window, degrading quality with
  no error. Prompts longer than ``config.LOCAL_MAX_INPUT_TOKENS`` are truncated
  from the left (keeping the query and the most recent context, discarding the
  oldest retrieved documents) and the fact is recorded so
  ``truncation_warnings`` still surfaces it.

* **Generation is serialised per model.** The PARALLEL and Graph Engineering topologies fan
  experts out into concurrent LangGraph branches. Different experts hold
  different models, so they genuinely overlap; but two calls into the *same*
  ``transformers`` model from two threads is not safe, so each cached model
  carries its own lock. The lock is per-model, not global, precisely so the
  parallel-vs-sequential comparison still measures real concurrency.

* **The adapter is a switch, not a fact.** ``config.LOCAL_PEFT_USE_ADAPTERS``
  selects the "peft" arm (adapters loaded) or the "base" control arm (identical
  base weights, no adapter). This is what lets the benchmark answer whether
  fine-tuning helped at all, instead of only which topology wins among tuned
  agents. The two arms are cached separately so one process could serve both,
  though the benchmark runs them as separate passes.

Nothing here is imported unless ``GENERATION_PROVIDER=local_peft``, so the
Ollama and DeepSeek paths keep working on a machine with no torch installed.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

ROOT_DIR = Path(__file__).resolve().parent

# Lazy, module-level singletons. Imports are deferred into _import_torch_stack()
# so that merely importing local_models.py on a torch-less machine is harmless.
#
# Each cache entry is a POOL (list) of _LoadedModel replicas, not a single
# instance - see config.LOCAL_MODEL_POOL_SIZE. Pool size 1 (the default) is
# behaviourally identical to the original one-instance-per-key design.
_MODELS: Dict[str, List["_LoadedModel"]] = {}
_LOAD_LOCK = threading.Lock()
# Round-robins which pool replica the next get_loaded_model() call receives.
# Kept separate from _LOAD_LOCK, which only guards first-time pool construction
# - cursor advancement happens on every call and must not serialise on that.
_POOL_CURSORS: Dict[str, int] = {}
_POOL_CURSOR_LOCK = threading.Lock()
_TORCH = None
_TRANSFORMERS = None


class LocalModelUnavailable(RuntimeError):
    """Raised when the local PEFT stack cannot be used, with a fixable message."""


def _import_torch_stack():
    """Import torch/transformers once, with an actionable error if missing."""
    global _TORCH, _TRANSFORMERS
    if _TORCH is not None:
        return _TORCH, _TRANSFORMERS
    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise LocalModelUnavailable(
            "GENERATION_PROVIDER=local_peft needs the local fine-tuning stack. "
            "Install it: pip install -r requirements-finetune.txt "
            f"(import failed: {exc})"
        ) from exc
    _TORCH, _TRANSFORMERS = torch, transformers
    return _TORCH, _TRANSFORMERS


def _compute_dtype():
    torch, _ = _import_torch_stack()
    name = str(config.LOCAL_COMPUTE_DTYPE).strip().lower()
    if name in {"bfloat16", "bf16"}:
        # bf16 needs Ampere or newer. The RTX 4070 (Ada) qualifies; fall back
        # rather than crash on older cards.
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.bfloat16
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    return torch.float32


def _resolve_device(resolved_role: str):
    """Which CUDA device (or 'cpu') a role's weights should load onto.

    Reads config.LOCAL_ROLE_DEVICES (see config.py). On a single-GPU box every
    role defaults to "cuda:0", matching the original design where concurrency
    during the PARALLEL/graph_engineering expert fan-out comes from per-model
    locks timesharing one device. On a multi-GPU host each role can be pinned to
    its own device for real hardware parallelism. Falls back to "cuda:0" if the
    configured index does not exist on this machine, rather than crashing a
    single-GPU dev box that inherited a multi-GPU .env.
    """
    torch, _ = _import_torch_stack()
    requested = str(config.LOCAL_ROLE_DEVICES.get(resolved_role, "cuda:0")).strip().lower()
    if requested == "cpu" or not torch.cuda.is_available():
        return "cpu"
    if requested.startswith("cuda:"):
        try:
            index = int(requested.split(":", 1)[1])
        except ValueError:
            index = 0
        if index >= torch.cuda.device_count():
            print(
                f"[local_models] {resolved_role}: requested {requested} but only "
                f"{torch.cuda.device_count()} GPU(s) visible; falling back to cuda:0"
            )
            index = 0
        return index
    return 0


def resolve_role(role: Optional[str]) -> str:
    """Map an agent role to a registry key.

    The three domain experts map to themselves. Everything else - planner,
    router, memory, aggregator, validator, response, QueryAnalyzer - is a
    coordination node and maps to config.LOCAL_COORDINATOR_ROLE, because a
    fourth 3B model does not fit in 8GB and because keeping specialisation
    confined to the experts is what makes the topology the independent variable.
    """
    key = str(role or "").strip().lower()
    if key in config.LOCAL_PEFT_ROLES:
        return key
    return config.LOCAL_COORDINATOR_ROLE


def uses_adapter(role: Optional[str]) -> bool:
    """Whether this role should have its LoRA adapter applied."""
    if not config.LOCAL_PEFT_USE_ADAPTERS:
        return False
    key = str(role or "").strip().lower()
    # Per-role opt-out, checked before the expert test: an expert listed in
    # LOCAL_UNADAPTED_ROLES runs on its base weights even in the peft arm. See the
    # rationale in config.py - general_qa is there because its adapter degenerates
    # on the long out-of-domain prompts this benchmark serves it.
    if key in config.LOCAL_UNADAPTED_ROLES:
        return False
    if key in config.LOCAL_PEFT_ROLES:
        return True
    # Coordination node: adapter off unless explicitly enabled.
    return config.LOCAL_COORDINATOR_USE_ADAPTER


def adapter_path(role: str) -> Path:
    """Absolute path to a role's adapter directory (may not exist)."""
    spec = config.LOCAL_PEFT_ROLES[resolve_role(role)]
    raw = Path(spec["adapter"])
    return raw if raw.is_absolute() else ROOT_DIR / raw


def describe_roles() -> List[Dict[str, Any]]:
    """Per-role provenance, for /runtime, run_meta.json and the paper's table.

    Reported rather than assumed: an adapter directory that was never trained
    shows up here as adapter_present=False instead of quietly running the base
    model while the run metadata claims otherwise.
    """
    rows = []
    for role, spec in config.LOCAL_PEFT_ROLES.items():
        path = adapter_path(role)
        rows.append(
            {
                "role": role,
                "base_model": spec["base_model"],
                "dataset": spec["dataset"],
                "adapter_dir": str(path),
                "adapter_present": (path / "adapter_config.json").exists(),
                "adapter_requested": uses_adapter(role),
            }
        )
    return rows


def arm_name() -> str:
    """'peft' when adapters are in play, 'base' for the untuned control arm."""
    return "peft" if config.LOCAL_PEFT_USE_ADAPTERS else "base"


class _LoadedModel:
    """One base model (optionally + adapter) plus its tokenizer, lock and stream."""

    def __init__(
        self,
        key: str,
        model,
        tokenizer,
        base_model_id: str,
        adapter_dir: Optional[str],
        stream=None,
    ):
        self.key = key
        self.model = model
        self.tokenizer = tokenizer
        self.base_model_id = base_model_id
        self.adapter_dir = adapter_dir
        # Per-replica, not global: concurrent experts (and, with a pool size > 1,
        # concurrent replicas of the SAME role) hold independent models and must
        # be able to genuinely overlap (see module docstring).
        self.lock = threading.Lock()
        # A dedicated CUDA stream so this replica's kernels can actually run
        # concurrently with another replica's on the GPU's own scheduler, rather
        # than implicitly serialising on PyTorch's shared default stream just
        # because two Python threads happened to call .generate() around the
        # same time. None on CPU or when CUDA is unavailable - streams are a
        # CUDA-only concept.
        self.stream = stream


def _load_model(role: str, want_adapter: bool) -> _LoadedModel:
    """Load one role's model. Caller holds _LOAD_LOCK.

    `role` is the RESOLVED registry key (which weights to load) and
    `want_adapter` is decided from the REQUESTED role by the caller. The two
    must stay separate: a coordination node resolves to the general expert's
    weights but must not inherit its adapter, and deriving want_adapter from the
    resolved role here would silently give it one.
    """
    torch, transformers = _import_torch_stack()
    spec = config.LOCAL_PEFT_ROLES[role]
    base_model_id = spec["base_model"]
    adapter_dir = adapter_path(role) if want_adapter else None

    if want_adapter and not (adapter_dir / "adapter_config.json").exists():
        raise LocalModelUnavailable(
            f"Role '{role}' asks for its LoRA adapter but none exists at "
            f"{adapter_dir}. Train it first:\n"
            f"    python finetune/train_qlora.py --role {role}\n"
            f"or run the untuned control arm with LEGALAI_USE_ADAPTERS=0."
        )

    quant_config = None
    if config.LOCAL_LOAD_IN_4BIT:
        quant_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.LOCAL_QUANT_TYPE,
            bnb_4bit_use_double_quant=config.LOCAL_DOUBLE_QUANT,
            bnb_4bit_compute_dtype=_compute_dtype(),
        )

    print(
        f"[local_models] loading role={role} base={base_model_id} "
        f"4bit={config.LOCAL_LOAD_IN_4BIT} adapter={'yes' if want_adapter else 'no'}"
    )
    started = time.perf_counter()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model_id, trust_remote_code=config.LOCAL_TRUST_REMOTE_CODE
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # trust_remote_code defaults to FALSE, and that is load-bearing rather than
    # merely cautious. All three families here (Llama, Qwen2, Phi3) have native
    # implementations in transformers; with trust_remote_code=True, Phi-3.5-mini
    # instead loads the modeling_phi3.py bundled in its repo, which still uses
    # the pre-4.47 cache API and dies with
    # "'DynamicCache' object has no attribute 'seen_tokens'" on the first
    # generate() under transformers 5.x. Step 0 caught exactly that.
    #
    # sdpa attention rather than eager: same numerics, materially less
    # activation memory, which matters when three models share 8GB.
    device = _resolve_device(role) if torch.cuda.is_available() else "cpu"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quant_config,
        dtype=_compute_dtype(),
        device_map={"": device} if torch.cuda.is_available() else "cpu",
        trust_remote_code=config.LOCAL_TRUST_REMOTE_CODE,
        attn_implementation=config.LOCAL_ATTN_IMPLEMENTATION,
    )

    if want_adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover
            raise LocalModelUnavailable(
                f"LoRA adapters require the 'peft' package: {exc}"
            ) from exc
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)

    model.eval()
    elapsed = time.perf_counter() - started
    print(f"[local_models]   loaded role={role} in {elapsed:.1f}s")

    # Created on the SAME device the model was placed on - torch.cuda.Stream()
    # with no device argument binds to whatever device is "current" for this
    # thread, which is not necessarily this role's configured device on a
    # multi-GPU host (see LOCAL_ROLE_DEVICES).
    stream = torch.cuda.Stream(device=device) if torch.cuda.is_available() else None

    return _LoadedModel(
        key=_cache_key(role, want_adapter),
        model=model,
        tokenizer=tokenizer,
        base_model_id=base_model_id,
        adapter_dir=str(adapter_dir) if adapter_dir else None,
        stream=stream,
    )


def _cache_key(resolved_role: str, want_adapter: bool) -> str:
    """Cache on (base model, adapter-or-not).

    `want_adapter` is passed in rather than derived from `resolved_role`, and
    that distinction is the whole point of this function. The coordination nodes
    resolve to the general expert's registry entry, so a key derived from the
    resolved role alone would be identical to the general expert's - handing
    every aggregator, validator and response call a domain-specialised model,
    silently, with nothing in the output to show it. Keying on the adapter
    decision instead keeps the two as separate cache entries backed by separate
    loads of the same base weights.
    """
    spec = config.LOCAL_PEFT_ROLES[resolved_role]
    suffix = "peft" if want_adapter else "base"
    return f"{spec['base_model']}::{suffix}"


def _pool_size(resolved_role: str) -> int:
    """How many replicas to load for this role. 1 unless configured higher."""
    try:
        size = int(config.LOCAL_MODEL_POOL_SIZE.get(resolved_role, 1))
    except (TypeError, ValueError):
        size = 1
    return max(1, size)


def _next_replica(key: str, pool: List["_LoadedModel"]) -> "_LoadedModel":
    """Round-robin which pool replica the next caller gets.

    Round-robin rather than "pick whichever is free": with pool size 1 (the
    common case) there is only one choice, and correctly detecting "free" would
    need a semaphore per replica for no benefit at that size. At pool size > 1,
    round-robin still spreads load close to evenly across replicas without
    needing that extra machinery - a caller that lands on a busy replica just
    waits on that replica's own lock, exactly as a single-instance role already
    does today.
    """
    if len(pool) == 1:
        return pool[0]
    with _POOL_CURSOR_LOCK:
        index = _POOL_CURSORS.get(key, 0)
        _POOL_CURSORS[key] = (index + 1) % len(pool)
    return pool[index]


def get_loaded_model(role: Optional[str]) -> _LoadedModel:
    """Return a cached model replica for a role, loading the pool on first use.

    Note the asymmetry, which is deliberate: WHICH weights to load comes from
    the resolved role, but WHETHER to attach the adapter comes from the role as
    requested. See _cache_key.

    Returns one replica from that role's pool (see config.LOCAL_MODEL_POOL_SIZE
    and _next_replica). Pool size 1 - the default - returns the same single
    instance every time, identical to the pre-pooling behaviour.
    """
    resolved = resolve_role(role)
    want_adapter = uses_adapter(role)
    key = _cache_key(resolved, want_adapter)
    pool = _MODELS.get(key)
    if pool is None:
        with _LOAD_LOCK:
            pool = _MODELS.get(key)
            if pool is None:
                size = _pool_size(resolved)
                pool = [_load_model(resolved, want_adapter) for _ in range(size)]
                _MODELS[key] = pool
    return _next_replica(key, pool)


def preload_all(roles: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load every expert model up front and report VRAM use.

    This is Step 0 of the pivot plan turned into a callable: the whole design
    rests on three 4-bit models co-residing in 8GB, and that had never been
    tested. Also used by the backend at startup so first-call load time does not
    land inside a measured benchmark run.
    """
    torch, _ = _import_torch_stack()
    targets = roles or list(config.LOCAL_PEFT_ROLES.keys())
    report: Dict[str, Any] = {"roles": [], "cuda_available": bool(torch.cuda.is_available())}

    if report["cuda_available"]:
        torch.cuda.reset_peak_memory_stats()
        free_before, total = torch.cuda.mem_get_info()
        report["vram_total_mib"] = round(total / 1024**2)
        report["vram_free_before_mib"] = round(free_before / 1024**2)

    for role in targets:
        started = time.perf_counter()
        loaded = get_loaded_model(role)
        entry = {
            "role": role,
            "base_model": loaded.base_model_id,
            "adapter_dir": loaded.adapter_dir,
            "pool_size": _pool_size(resolve_role(role)),
            # Time to build the WHOLE pool, not one replica: the first call for
            # a key loads every replica inside _LOAD_LOCK (see get_loaded_model).
            "load_s": round(time.perf_counter() - started, 2),
        }
        if report["cuda_available"]:
            free_now, _ = torch.cuda.mem_get_info()
            entry["vram_free_after_mib"] = round(free_now / 1024**2)
            entry["torch_allocated_mib"] = round(torch.cuda.memory_allocated() / 1024**2)
        report["roles"].append(entry)

    if report["cuda_available"]:
        free_after, total = torch.cuda.mem_get_info()
        report["vram_free_after_mib"] = round(free_after / 1024**2)
        report["vram_used_by_models_mib"] = round(
            (report["vram_free_before_mib"] - free_after / 1024**2)
        )
        report["torch_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2)
    return report


# --------------------------------------------------------------------------- #
# LangChain-compatible chat model
# --------------------------------------------------------------------------- #


class _LocalAIMessage:
    """Minimal AIMessage stand-in.

    Deliberately not a langchain AIMessage subclass: everything downstream in
    this project reads exactly two attributes - `.content` and
    `.response_metadata` - and constructing a real AIMessage would drag pydantic
    validation into the hot generation path for no benefit. `usage_metadata` is
    populated too, for anything that prefers the newer langchain convention.
    """

    def __init__(self, content: str, response_metadata: Dict[str, Any]):
        self.content = content
        self.response_metadata = response_metadata
        self.usage_metadata = {
            "input_tokens": response_metadata.get("prompt_tokens", 0),
            "output_tokens": response_metadata.get("completion_tokens", 0),
            "total_tokens": response_metadata.get("prompt_tokens", 0)
            + response_metadata.get("completion_tokens", 0),
        }
        self.type = "ai"

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"_LocalAIMessage({self.content[:60]!r})"


class LocalChatModel:
    """Chat model backed by a locally loaded (optionally LoRA-adapted) HF model.

    Implements just enough of the langchain Runnable surface for this project:
    ``invoke`` accepting a prompt-value/string/message-list, and ``__ror__`` so
    ``prompt | llm`` keeps working in every agent unchanged.
    """

    def __init__(
        self,
        role: Optional[str] = None,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ):
        self.role = resolve_role(role)
        self.requested_role = role
        self.temperature = float(temperature)
        self.seed = seed
        self.max_new_tokens = int(max_new_tokens or config.LLM_NUM_PREDICT)
        # Load eagerly: an agent constructed at graph-build time should surface a
        # missing adapter then, not halfway through a 540-run benchmark.
        # Pass the REQUESTED role, not self.role (which is already resolved).
        #
        # This one argument caused every coordination node - aggregator,
        # validator, response, router, planner, memory, QueryAnalyzer - to run
        # with the general expert's LoRA adapter attached. resolve_role() maps
        # "aggregator" to "general_qa", and get_loaded_model() derives the adapter
        # decision from the role it is given, so handing it the resolved name made
        # uses_adapter() answer for the EXPERT rather than for the coordination
        # node. get_loaded_model was fixed to consult the requested role; passing
        # the resolved one here silently defeated that fix.
        #
        # The symptom was severe and looked like something else entirely: the
        # aggregator produced word-salad at every prompt length, which was
        # variously misdiagnosed as adapter damage, a train/serve context
        # mismatch, and missing repetition control. The experts were fine
        # throughout - they are supposed to carry adapters.
        #
        # The earlier regression test asserted on get_loaded_model("aggregator")
        # directly and passed, because it never exercised this constructor. See
        # test_coordination_node_via_chat_model_has_no_adapter.
        self._loaded = get_loaded_model(role)

    # -- langchain plumbing -------------------------------------------------
    def __ror__(self, other):  # pragma: no cover - exercised via agents
        """Support `prompt | llm` where prompt is a langchain template."""
        return _PromptChain(other, self)

    def bind(self, **_kwargs):  # pragma: no cover - compatibility shim
        return self

    # -- prompt handling ----------------------------------------------------
    @staticmethod
    def _to_messages(value: Any) -> List[Dict[str, str]]:
        """Normalise langchain prompt values / strings / message lists."""
        # ChatPromptValue and friends expose to_messages().
        if hasattr(value, "to_messages"):
            value = value.to_messages()
        if isinstance(value, str):
            return [{"role": "user", "content": value}]
        if isinstance(value, dict):
            return [{"role": str(value.get("role", "user")), "content": str(value.get("content", ""))}]
        if isinstance(value, (list, tuple)):
            messages = []
            for item in value:
                if isinstance(item, dict):
                    messages.append(
                        {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))}
                    )
                    continue
                item_type = getattr(item, "type", None) or getattr(item, "role", None) or "human"
                mapping = {"human": "user", "ai": "assistant", "system": "system"}
                messages.append(
                    {
                        "role": mapping.get(str(item_type), str(item_type)),
                        "content": str(getattr(item, "content", item)),
                    }
                )
            return messages or [{"role": "user", "content": ""}]
        return [{"role": "user", "content": str(value)}]

    def _encode(self, messages: List[Dict[str, str]]):
        torch, _ = _import_torch_stack()
        tokenizer = self._loaded.tokenizer

        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:  # pragma: no cover - all three chosen models ship a template
            text = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
            text += "\n\nassistant:"

        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        truncated_from = 0
        limit = config.LOCAL_MAX_INPUT_TOKENS
        if input_ids.shape[-1] > limit:
            # Truncate from the LEFT: the query, chat history tail and the
            # generation prompt all live at the end of the rendered template,
            # so dropping the head sheds the oldest retrieved documents first,
            # which is the least damaging thing to lose.
            truncated_from = int(input_ids.shape[-1])
            input_ids = input_ids[:, -limit:]
            attention_mask = attention_mask[:, -limit:]

        device = next(self._loaded.model.parameters()).device
        return (
            input_ids.to(device),
            attention_mask.to(device),
            truncated_from,
        )

    # -- generation ---------------------------------------------------------
    def invoke(self, value: Any, config_dict: Any = None, **_kwargs) -> _LocalAIMessage:
        torch, transformers = _import_torch_stack()
        messages = self._to_messages(value)
        input_ids, attention_mask, truncated_from = self._encode(messages)
        prompt_tokens = int(input_ids.shape[-1])

        do_sample = self.temperature > 0.0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._loaded.tokenizer.pad_token_id,
            "eos_token_id": self._loaded.tokenizer.eos_token_id,
        }
        # Repetition control. transformers applies none by default, whereas Ollama
        # (every pre-pivot run) applies repeat_penalty 1.1 - so without this the
        # move to raw transformers quietly removed a guard the earlier experiments
        # always had, and these 2-3B models collapse into token loops that consume
        # the whole generation budget. See config.LOCAL_REPETITION_PENALTY.
        if config.LOCAL_REPETITION_PENALTY and config.LOCAL_REPETITION_PENALTY != 1.0:
            gen_kwargs["repetition_penalty"] = config.LOCAL_REPETITION_PENALTY
        if config.LOCAL_NO_REPEAT_NGRAM > 0:
            gen_kwargs["no_repeat_ngram_size"] = config.LOCAL_NO_REPEAT_NGRAM
        if do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = 0.95

        # A per-call torch.Generator instead of torch.manual_seed/cuda.manual_seed_all.
        # Those reseed PyTorch's GLOBAL RNG, which was only safe while at most one
        # generate() call could be in flight system-wide - never actually true
        # here, since PARALLEL/graph_engineering already run the legal and news
        # experts concurrently on separate models, and a model-replica pool (see
        # config.LOCAL_MODEL_POOL_SIZE) adds more concurrent callers still. Two
        # threads each calling manual_seed() around their own generate() can
        # interleave and silently reseed each other, breaking the
        # seed-to-reproducible-output guarantee the benchmark's repeats depend
        # on. A Generator is call-local state, so concurrent callers cannot step
        # on each other no matter how many models/replicas run at once.
        generator = None
        if self.seed is not None:
            gen_device = next(self._loaded.model.parameters()).device
            generator = torch.Generator(device=gen_device).manual_seed(int(self.seed))

        started = time.perf_counter()
        stream_ctx = (
            torch.cuda.stream(self._loaded.stream)
            if self._loaded.stream is not None
            else contextlib.nullcontext()
        )
        with self._loaded.lock:
            with stream_ctx, torch.inference_mode():
                output = self._loaded.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generator=generator,
                    **gen_kwargs,
                )
            if self._loaded.stream is not None:
                # The lock is released as soon as this replica's own work is
                # queued; block only long enough for THIS call's kernels to
                # finish before reading `output` below, without forcing every
                # other stream/replica on the device to also drain.
                self._loaded.stream.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000

        generated = output[0][prompt_tokens:]
        completion_tokens = int(generated.shape[-1])
        text = self._loaded.tokenizer.decode(generated, skip_special_tokens=True).strip()

        metadata: Dict[str, Any] = {
            # Same keys BaseAgent._record_tokens already reads for Ollama and
            # DeepSeek, so token telemetry needs no provider-specific branch.
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prompt_eval_count": prompt_tokens,
            "eval_count": completion_tokens,
            "model_name": self._loaded.base_model_id,
            "role": self.role,
            "adapter_dir": self._loaded.adapter_dir,
            "arm": arm_name(),
            "generation_ms": round(elapsed_ms, 2),
            "seed": self.seed,
            "temperature": self.temperature,
            "do_sample": do_sample,
            "repetition_penalty": gen_kwargs.get("repetition_penalty", 1.0),
            # Hitting the cap exactly is the signature of a model that never
            # emitted its stop token - usually a repetition loop. Recorded so the
            # benchmark data shows it rather than leaving it to be noticed by eye.
            "hit_token_cap": completion_tokens >= self.max_new_tokens,
        }
        if truncated_from:
            metadata["input_truncated_from"] = truncated_from
            metadata["input_truncated_to"] = prompt_tokens

        return _LocalAIMessage(text, metadata)

    # Some langchain call sites use .predict/.__call__; keep them working.
    __call__ = invoke


class _PromptChain:
    """`prompt | LocalChatModel` result: renders the template, then generates."""

    def __init__(self, prompt, llm: LocalChatModel):
        self.prompt = prompt
        self.llm = llm

    def invoke(self, variables: Any, config_dict: Any = None, **kwargs):
        rendered = self.prompt.invoke(variables) if hasattr(self.prompt, "invoke") else variables
        return self.llm.invoke(rendered, **kwargs)


def unload_all():
    """Drop every cached model and free VRAM. Used by tests and Step 0.

    Clearing the registry is not sufficient on its own: anything still holding a
    _LoadedModel (a LocalChatModel a caller has not released yet) keeps the
    weights alive, and empty_cache() then frees nothing. The model/tokenizer
    attributes are therefore cleared explicitly, so a stale LocalChatModel
    reference cannot silently pin 2GB of VRAM. It will raise if used afterwards,
    which is the correct outcome - far better than appearing to work while the
    next load OOMs.
    """
    global _MODELS, _POOL_CURSORS
    with _LOAD_LOCK:
        cached = [replica for pool in _MODELS.values() for replica in pool]
        _MODELS = {}
        _POOL_CURSORS = {}
    for entry in cached:
        entry.model = None
        entry.tokenizer = None
    del cached

    import gc

    gc.collect()
    if _TORCH is not None and _TORCH.cuda.is_available():  # pragma: no cover
        _TORCH.cuda.empty_cache()
