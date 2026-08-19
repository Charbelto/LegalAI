"""Tests for the per-role local PEFT model registry.

These cover the resolution logic only - which role gets which model, whether its
adapter is applied, and how the cache is keyed. No torch, no weights, no GPU: the
functions under test are pure, and they are exactly the ones where a mistake is
invisible at runtime. A role silently resolving to the wrong model, or two roles
colliding in the cache, would produce a complete benchmark whose per-expert
claims were simply false, with nothing in the output to show it.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def local_models_module():
    """Load local_models.py by path, like conftest does for the agent modules."""
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


@pytest.fixture()
def cfg(local_models_module):
    return sys.modules["config"]


# ---------------------------------------------------------------------------
# Role resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["legal", "news", "general_qa"])
def test_expert_roles_resolve_to_themselves(local_models_module, role):
    """Each domain expert must get its OWN model, not a shared one."""
    assert local_models_module.resolve_role(role) == role


@pytest.mark.parametrize(
    "role",
    ["planner", "router", "memory", "aggregator", "validator", "response", "query_analyzer", None, ""],
)
def test_coordination_roles_share_the_coordinator_model(local_models_module, cfg, role):
    """Non-expert nodes must not each load a model - 8GB does not allow it."""
    assert local_models_module.resolve_role(role) == cfg.LOCAL_COORDINATOR_ROLE


def test_expert_base_models_differ(local_models_module, cfg):
    """The three experts must resolve to three different underlying models."""
    resolved = {
        role: cfg.LOCAL_PEFT_ROLES[local_models_module.resolve_role(role)]["base_model"]
        for role in ("legal", "news", "general_qa")
    }
    assert len(set(resolved.values())) == 3, resolved


# ---------------------------------------------------------------------------
# Adapter application and the peft/base arm switch
# ---------------------------------------------------------------------------


def test_experts_use_adapters_in_the_peft_arm(local_models_module, cfg, monkeypatch):
    monkeypatch.setattr(cfg, "LOCAL_PEFT_USE_ADAPTERS", True)
    # The per-role opt-out (general_qa runs unadapted in deployment) is a
    # separate decision; clear it so this exercises the mechanism itself.
    monkeypatch.setattr(cfg, "LOCAL_UNADAPTED_ROLES", set())
    for role in ("legal", "news", "general_qa"):
        assert local_models_module.uses_adapter(role) is True
    assert local_models_module.arm_name() == "peft"


def test_base_arm_disables_every_adapter(local_models_module, cfg, monkeypatch):
    """The control arm must be the identical base weights, with nothing attached.

    If any role kept its adapter here, the 'base' arm would be a mixture and RQ2
    would be comparing PEFT against partially-PEFT.
    """
    monkeypatch.setattr(cfg, "LOCAL_PEFT_USE_ADAPTERS", False)
    monkeypatch.setattr(cfg, "LOCAL_COORDINATOR_USE_ADAPTER", True)
    for role in ("legal", "news", "general_qa", "aggregator", None):
        assert local_models_module.uses_adapter(role) is False
    assert local_models_module.arm_name() == "base"


def test_coordination_nodes_are_unadapted_by_default(local_models_module, cfg, monkeypatch):
    """Specialisation stays confined to the experts.

    The coordination nodes share the general expert's base weights; if they also
    inherited its adapter, the aggregator/validator/response would be
    domain-tuned too, and 'topology combines specialised experts' would no longer
    be what the experiment isolates.
    """
    monkeypatch.setattr(cfg, "LOCAL_PEFT_USE_ADAPTERS", True)
    # The per-role opt-out (general_qa runs unadapted in deployment) is a
    # separate decision; clear it so this exercises the mechanism itself.
    monkeypatch.setattr(cfg, "LOCAL_UNADAPTED_ROLES", set())
    monkeypatch.setattr(cfg, "LOCAL_COORDINATOR_USE_ADAPTER", False)
    assert local_models_module.uses_adapter("aggregator") is False
    assert local_models_module.uses_adapter("validator") is False
    assert local_models_module.uses_adapter(None) is False
    # ...while the expert of the same base model still does.
    assert local_models_module.uses_adapter("general_qa") is True


def test_coordination_node_is_not_handed_the_experts_adapter(
    local_models_module, cfg, monkeypatch
):
    """The decisive test: what does get_loaded_model actually resolve to?

    This is a regression test for a real bug. get_loaded_model resolved the role
    first ("aggregator" -> "general_qa") and then asked uses_adapter() about the
    RESOLVED role, which is an expert and therefore answers True. Every
    aggregator, validator and response call was consequently served by the
    domain-specialised model, with nothing in the output to reveal it.

    Asserting on uses_adapter("aggregator") alone did NOT catch this - that
    function was always correct. The bug lived in how its answer was used, so the
    assertion has to be on the load path itself.
    """
    monkeypatch.setattr(cfg, "LOCAL_PEFT_USE_ADAPTERS", True)
    # The per-role opt-out (general_qa runs unadapted in deployment) is a
    # separate decision; clear it so this exercises the mechanism itself.
    monkeypatch.setattr(cfg, "LOCAL_UNADAPTED_ROLES", set())
    monkeypatch.setattr(cfg, "LOCAL_COORDINATOR_USE_ADAPTER", False)
    monkeypatch.setattr(cfg, "LOCAL_COORDINATOR_ROLE", "general_qa")

    loaded = {}

    def _fake_load(resolved_role, want_adapter):
        loaded["resolved_role"] = resolved_role
        loaded["want_adapter"] = want_adapter
        return object()

    monkeypatch.setattr(local_models_module, "_load_model", _fake_load)
    monkeypatch.setattr(local_models_module, "_MODELS", {})

    local_models_module.get_loaded_model("aggregator")
    assert loaded["resolved_role"] == "general_qa", "must reuse the general expert's weights"
    assert loaded["want_adapter"] is False, "but must NOT attach its adapter"

    # And the expert itself still does get one, from the same base weights.
    loaded.clear()
    local_models_module.get_loaded_model("general_qa")
    assert loaded["resolved_role"] == "general_qa"
    assert loaded["want_adapter"] is True


def test_coordinator_and_expert_occupy_different_cache_slots(
    local_models_module, cfg, monkeypatch
):
    """Same base weights, different adapter decision -> different cache keys.

    If they collided, whichever loaded first would be served to both.
    """
    monkeypatch.setattr(cfg, "LOCAL_PEFT_USE_ADAPTERS", True)
    # The per-role opt-out (general_qa runs unadapted in deployment) is a
    # separate decision; clear it so this exercises the mechanism itself.
    monkeypatch.setattr(cfg, "LOCAL_UNADAPTED_ROLES", set())
    monkeypatch.setattr(cfg, "LOCAL_COORDINATOR_USE_ADAPTER", False)
    monkeypatch.setattr(cfg, "LOCAL_COORDINATOR_ROLE", "general_qa")

    expert_key = local_models_module._cache_key(
        local_models_module.resolve_role("general_qa"),
        local_models_module.uses_adapter("general_qa"),
    )
    coordinator_key = local_models_module._cache_key(
        local_models_module.resolve_role("aggregator"),
        local_models_module.uses_adapter("aggregator"),
    )

    assert expert_key != coordinator_key
    assert expert_key.endswith("::peft")
    assert coordinator_key.endswith("::base")


def test_cache_key_separates_the_two_arms(local_models_module, cfg, monkeypatch):
    """peft and base variants of one base model must never share a cache slot."""
    monkeypatch.setattr(cfg, "LOCAL_PEFT_USE_ADAPTERS", True)
    peft_key = local_models_module._cache_key("legal", local_models_module.uses_adapter("legal"))
    monkeypatch.setattr(cfg, "LOCAL_PEFT_USE_ADAPTERS", False)
    base_key = local_models_module._cache_key("legal", local_models_module.uses_adapter("legal"))

    assert peft_key != base_key
    assert peft_key.endswith("::peft")
    assert base_key.endswith("::base")


def test_every_expert_gets_its_own_weights(local_models_module, cfg, monkeypatch):
    """The three experts must resolve to three separate cache entries.

    Two experts sharing a cache slot would mean two 'specialists' backed by one
    model, quietly turning the three-model design into a two-model one.
    """
    monkeypatch.setattr(cfg, "LOCAL_PEFT_USE_ADAPTERS", True)
    keys = {
        role: local_models_module._cache_key(
            local_models_module.resolve_role(role), local_models_module.uses_adapter(role)
        )
        for role in ("legal", "news", "general_qa")
    }
    assert len(set(keys.values())) == 3, keys


# ---------------------------------------------------------------------------
# Provenance reporting
# ---------------------------------------------------------------------------


def test_describe_roles_reports_missing_adapters_honestly(local_models_module, cfg, monkeypatch, tmp_path):
    """An untrained adapter must report adapter_present=False, not be assumed.

    benchmark.py's preflight refuses to run the peft arm when an adapter is
    missing; that guard is only as good as this flag, and a default of True here
    would let 270 runs of base models be labelled 'peft'.
    """
    monkeypatch.setattr(cfg, "LOCAL_ADAPTER_DIR", str(tmp_path / "adapters"))
    monkeypatch.setattr(
        cfg,
        "LOCAL_PEFT_ROLES",
        {
            role: {**spec, "adapter": str(tmp_path / "adapters" / role)}
            for role, spec in cfg.LOCAL_PEFT_ROLES.items()
        },
    )
    monkeypatch.setattr(cfg, "LOCAL_PEFT_USE_ADAPTERS", True)
    # The per-role opt-out (general_qa runs unadapted in deployment) is a
    # separate decision; clear it so this exercises the mechanism itself.
    monkeypatch.setattr(cfg, "LOCAL_UNADAPTED_ROLES", set())

    rows = local_models_module.describe_roles()
    assert len(rows) == 3
    for row in rows:
        assert row["adapter_present"] is False
        assert row["adapter_requested"] is True

    # Now create one, and only that one flips.
    trained = tmp_path / "adapters" / "legal"
    trained.mkdir(parents=True)
    (trained / "adapter_config.json").write_text("{}", encoding="utf-8")

    rows = {row["role"]: row for row in local_models_module.describe_roles()}
    assert rows["legal"]["adapter_present"] is True
    assert rows["news"]["adapter_present"] is False


def test_adapter_path_is_absolute(local_models_module):
    """Relative adapter paths break the server subprocess, whose cwd may differ."""
    for role in ("legal", "news", "general_qa"):
        assert local_models_module.adapter_path(role).is_absolute()


# ---------------------------------------------------------------------------
# Model replica pooling (LOCAL_MODEL_POOL_SIZE) - exploits spare VRAM on a
# bigger GPU for genuine intra-role concurrency. Pool size 1 (the default) must
# be exactly the pre-pooling single-instance behaviour.
# ---------------------------------------------------------------------------


def test_pool_size_defaults_to_one_for_every_role(cfg):
    """The 8GB-laptop design must be unchanged unless someone opts in."""
    for role in ("legal", "news", "general_qa"):
        assert cfg.LOCAL_MODEL_POOL_SIZE[role] == 1


def test_pool_size_never_goes_below_one(local_models_module, cfg, monkeypatch):
    """A misconfigured 0/negative/non-numeric pool size must not disable a role."""
    monkeypatch.setitem(cfg.LOCAL_MODEL_POOL_SIZE, "general_qa", 0)
    assert local_models_module._pool_size("general_qa") == 1

    monkeypatch.setitem(cfg.LOCAL_MODEL_POOL_SIZE, "general_qa", -3)
    assert local_models_module._pool_size("general_qa") == 1


def test_next_replica_always_returns_the_only_replica_at_pool_size_one(local_models_module):
    """No round-robin bookkeeping should kick in when there is nothing to round-robin."""
    only = object()
    for _ in range(5):
        assert local_models_module._next_replica("k", [only]) is only
    # And it must not have touched the cursor table at all.
    assert "k" not in local_models_module._POOL_CURSORS


def test_next_replica_round_robins_across_a_larger_pool(local_models_module, monkeypatch):
    """Load should spread evenly across replicas, wrapping back to the start."""
    monkeypatch.setattr(local_models_module, "_POOL_CURSORS", {})
    pool = [object(), object(), object()]
    seen = [local_models_module._next_replica("k", pool) for _ in range(7)]
    assert seen == [pool[0], pool[1], pool[2], pool[0], pool[1], pool[2], pool[0]]


def test_get_loaded_model_builds_a_pool_of_the_configured_size(
    local_models_module, cfg, monkeypatch
):
    """Raising a role's pool size must load that many replicas, once, and cycle them."""
    monkeypatch.setattr(cfg, "LOCAL_PEFT_USE_ADAPTERS", False)
    monkeypatch.setitem(cfg.LOCAL_MODEL_POOL_SIZE, "general_qa", 3)
    monkeypatch.setattr(local_models_module, "_MODELS", {})
    monkeypatch.setattr(local_models_module, "_POOL_CURSORS", {})

    built = []

    def _fake_load(resolved_role, want_adapter):
        replica = object()
        built.append((resolved_role, want_adapter, replica))
        return replica

    monkeypatch.setattr(local_models_module, "_load_model", _fake_load)

    replicas = [local_models_module.get_loaded_model("aggregator") for _ in range(4)]

    # Exactly 3 replicas built (the configured pool size), not 4 - the pool is
    # built once on first use and then reused, never reloaded per call.
    assert len(built) == 3
    assert all(role == "general_qa" and adapter is False for role, adapter, _ in built)

    pool = [replica for _, _, replica in built]
    # 4 calls over a 3-replica pool: round-robin, wrapping back to the first.
    assert replicas == [pool[0], pool[1], pool[2], pool[0]]


def test_unload_all_clears_every_replica_in_every_pool(local_models_module, monkeypatch):
    """A pool of N replicas must release all N, not just one."""
    replica_a = local_models_module._LoadedModel(
        key="k::peft", model=object(), tokenizer=object(), base_model_id="m", adapter_dir=None
    )
    replica_b = local_models_module._LoadedModel(
        key="k::peft", model=object(), tokenizer=object(), base_model_id="m", adapter_dir=None
    )
    monkeypatch.setattr(local_models_module, "_MODELS", {"m::peft": [replica_a, replica_b]})
    monkeypatch.setattr(local_models_module, "_POOL_CURSORS", {"m::peft": 1})

    local_models_module.unload_all()

    assert local_models_module._MODELS == {}
    assert local_models_module._POOL_CURSORS == {}
    assert replica_a.model is None and replica_a.tokenizer is None
    assert replica_b.model is None and replica_b.tokenizer is None
