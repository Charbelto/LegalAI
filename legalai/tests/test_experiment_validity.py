"""Regression tests for the experiment-validity fixes.

Each test corresponds to a defect that silently invalidated earlier benchmark
runs. They are deliberately cheap: no Ollama, no ChromaDB, no network.
"""

import numpy as np
import pandas as pd
import pytest

import config


ABSTENTION = config.ABSTENTION_SENTENCE


# ---------------------------------------------------------------------------
# 1. Aggregator: abstention must not veto the whole ensemble
# ---------------------------------------------------------------------------


def test_single_abstaining_expert_does_not_veto_others(aggregator_agent):
    """One abstaining expert must not silence experts that did answer.

    This is the bug that made every topology containing the legal expert return a
    7-word string while router-only and expert-free topologies were untouched.
    """
    agent, chain = aggregator_agent
    state = {
        "query": "What are the obligations for high-risk AI systems?",
        "route": "legal",
        "agent_outputs": {
            "legal": ABSTENTION,
            "news": "Recent coverage describes phased enforcement during 2026.",
            "general_qa": "High-risk systems face conformity assessment duties.",
        },
        "retrieved_docs": [],
        "chat_history": [],
    }

    result = agent.invoke(state)

    assert result["abstained"] is False
    assert result["draft_response"] == "MOCK AGGREGATED ANSWER"
    assert result["abstained_experts"] == ["legal"]
    assert result["experts_run"] == 3
    assert result["expert_abstention_rate"] == pytest.approx(1 / 3, abs=1e-4)

    # The abstaining expert's text must not be fed to the aggregator as content.
    passed_expert_output = chain.calls[0]["expert_output"]
    assert "Recent coverage" in passed_expert_output
    assert "conformity assessment" in passed_expert_output
    assert passed_expert_output.count(ABSTENTION) == 0


def test_non_expert_outputs_are_not_counted_as_experts(aggregator_agent):
    """Only the three domain experts may count toward experts_run.

    `agent_outputs` is a shared channel: agents/router.py stores its route
    decision under "router". Counting every key made experts_run 4 in a
    three-expert topology, and that inflated denominator fed straight into
    expert_abstention_rate (abstained / experts_run) - a metric the paper
    reports - understating it by a quarter on every single run.
    """
    agent, _chain = aggregator_agent
    state = {
        "query": "What are the obligations for high-risk AI systems?",
        "route": "legal",
        "agent_outputs": {
            "router": "legal",            # not an expert
            "legal": ABSTENTION,
            "news": "Recent coverage describes phased enforcement.",
            "general_qa": "Conformity assessment duties apply.",
        },
        "retrieved_docs": [],
        "chat_history": [],
    }

    result = agent.invoke(state)

    assert result["experts_run"] == 3, "the router's output must not count as an expert"
    assert result["abstained_experts"] == ["legal"]
    # 1 of 3, not 1 of 4.
    assert result["expert_abstention_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_router_output_alone_is_not_treated_as_an_expert(aggregator_agent):
    """A topology that ran no experts must report zero, even though the router ran."""
    agent, _chain = aggregator_agent
    state = {
        "query": "Summarise the retrieved context.",
        "route": "general",
        "agent_outputs": {"router": "general"},
        "retrieved_docs": [],
        "chat_history": [],
    }

    result = agent.invoke(state)

    assert result["experts_run"] == 0
    assert result["expert_abstention_rate"] == 0.0
    assert result["abstained"] is False


def test_all_experts_abstaining_still_propagates_abstention(aggregator_agent):
    """When every expert abstains, the system genuinely abstains."""
    agent, chain = aggregator_agent
    state = {
        "query": "Does Article 999 apply to my toaster?",
        "route": "legal",
        "agent_outputs": {"legal": ABSTENTION, "news": ABSTENTION},
        "retrieved_docs": [],
        "chat_history": [],
    }

    result = agent.invoke(state)

    assert result["abstained"] is True
    assert result["draft_response"] == ABSTENTION
    assert result["abstained_experts"] == ["legal", "news"]
    assert result["expert_abstention_rate"] == 1.0
    assert chain.calls == []  # no wasted aggregation call


def test_expert_free_topology_is_not_treated_as_abstention(aggregator_agent):
    """verify_only runs no experts; that is not an abstention."""
    agent, _chain = aggregator_agent
    state = {
        "query": "Summarise the retrieved context.",
        "route": "general",
        "agent_outputs": {},
        "retrieved_docs": [],
        "chat_history": [],
    }

    result = agent.invoke(state)

    assert result["abstained"] is False
    assert result["experts_run"] == 0
    assert result["expert_abstention_rate"] == 0.0


def test_empty_expert_output_is_ignored_not_counted(aggregator_agent):
    """Blank expert outputs must not count as abstentions or as content."""
    agent, _chain = aggregator_agent
    state = {
        "query": "What is the EU AI Act?",
        "route": "legal",
        "agent_outputs": {"legal": "   ", "news": "A real news analysis."},
        "retrieved_docs": [],
        "chat_history": [],
    }

    result = agent.invoke(state)

    assert result["experts_run"] == 1
    assert result["abstained_experts"] == []
    assert result["abstained"] is False


def test_experts_run_survives_the_compiled_graph(aggregator_agent):
    """Aggregator telemetry must reach the graph's final state, not just its own return value.

    AggregatorAgent.invoke() sets experts_run/abstained_experts/expert_abstention_rate
    correctly on the dict it is handed (see test_single_abstaining_expert_does_not_veto_others
    above) - but that test calls agent.invoke() directly on a plain dict, so it never
    exercises LangGraph. LangGraph's StateGraph derives its tracked state channels from
    the AgentState TypedDict passed to StateGraph(), not from whatever keys a node
    happens to write; a field the aggregator sets that isn't declared in AgentState is
    silently dropped when state propagates to the next node. That was the actual defect:
    benchmark.py read a default of 0 for experts_run on every single row, in every mode,
    because AgentState never declared the field. This test runs a real (minimal)
    StateGraph built from the project's own AgentState schema so it would have caught that.
    """
    from langgraph.graph import StateGraph, END
    from state import AgentState

    agent, _chain = aggregator_agent

    def setup_node(state):
        return {
            "agent_outputs": {
                "legal": ABSTENTION,
                "news": "Recent coverage describes phased enforcement during 2026.",
                "general_qa": "High-risk systems face conformity assessment duties.",
            },
            "route": "legal",
            "retrieved_docs": [],
            "chat_history": [],
        }

    def aggregator_node(state):
        return agent.invoke(dict(state))

    graph = StateGraph(AgentState)
    graph.add_node("setup", setup_node)
    graph.add_node("aggregator", aggregator_node)
    graph.set_entry_point("setup")
    graph.add_edge("setup", "aggregator")
    graph.add_edge("aggregator", END)
    compiled = graph.compile()

    result = compiled.invoke({"query": "What are the obligations for high-risk AI systems?"})

    assert result["experts_run"] == 3
    assert result["abstained_experts"] == ["legal"]
    assert result["expert_abstention_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert result["abstained"] is False


def test_truncation_warnings_survives_expert_node_narrowing(monkeypatch):
    """truncation_warnings must reach the graph's final state through the real
    legal/news/general_qa node wrappers, not just via a correct AgentState schema.

    This field had two stacked bugs: (1) AgentState never declared
    truncation_warnings, so - exactly like experts_run above - LangGraph silently
    dropped it between nodes regardless of what a node returned; and (2)
    legal_node/news_node/general_qa_node in graph/workflow.py returned a narrowed
    dict (agent_outputs/agent_timings/thinking_log/agent_tokens only) that omitted
    the key entirely, so a correct schema alone would not have helped those three
    nodes specifically - it would still vanish whenever a legal/news/general_qa
    agent was the one that appended the warning. This test loads the real
    graph/workflow.py, swapping in cheap fake agent classes for the heavy real
    ones (the same technique conftest.py already uses to stub the 'agents'
    package so tests avoid Ollama/ChromaDB), so a regression in either fix would
    be caught. legal_node is exercised here; news_node/general_qa_node share the
    identical shape.
    """
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    class _NoOpAgent:
        def __init__(self, *_args, **_kwargs):
            pass

        def invoke(self, state):
            return state

    class _LegalAgent(_NoOpAgent):
        def invoke(self, state):
            state.setdefault("agent_outputs", {})["legal"] = "Legal analysis."
            state.setdefault("truncation_warnings", []).append(
                {"agent": "legal", "prompt_tokens": 7900}
            )
            return state

    agents_module = sys.modules["agents"]
    for name in (
        "PlannerAgent",
        "RouterAgent",
        "MemoryAgent",
        "RetrievalAgent",
        "NewsAgent",
        "GeneralQAAgent",
        "AggregatorAgent",
        "ValidationAgent",
        "ResponseAgent",
    ):
        monkeypatch.setattr(agents_module, name, _NoOpAgent, raising=False)
    monkeypatch.setattr(agents_module, "LegalAgent", _LegalAgent, raising=False)

    sys.modules.pop("graph.workflow", None)
    spec = importlib.util.spec_from_file_location("graph.workflow", root / "graph" / "workflow.py")
    workflow = importlib.util.module_from_spec(spec)
    sys.modules["graph.workflow"] = workflow
    try:
        spec.loader.exec_module(workflow)

        compiled = workflow.create_legal_ai_graph()
        result = compiled.invoke({
            "query": "What are the obligations for high-risk AI systems?",
            "session_id": "test",
            "route": "legal",
            "expert_execution_mode": "legal_first",
            "chat_history": [],
            "retrieved_docs": [],
            "agent_outputs": {},
            "agent_timings": {},
            "thinking_log": [],
            "validation_result": {},
        })

        assert result["truncation_warnings"] == [{"agent": "legal", "prompt_tokens": 7900}]
    finally:
        sys.modules.pop("graph.workflow", None)


# ---------------------------------------------------------------------------
# 2. Response agent: substring matching must not discard good answers
# ---------------------------------------------------------------------------


def test_answer_mentioning_the_abstention_sentence_survives(response_agent):
    """A complete answer that merely quotes the sentence must not be discarded."""
    agent, chain = response_agent
    draft = (
        "Providers must complete a conformity assessment under Article 43. "
        f'If evidence is thin the system should reply "{ABSTENTION}" instead of guessing.'
    )
    state = {"draft_response": draft, "query": "Explain conformity assessment", "abstained": False}

    result = agent.invoke(state)

    assert result["final_response"] == "MOCK POLISHED ANSWER"
    assert len(chain.calls) == 1


def test_explicit_abstention_flag_is_propagated(response_agent):
    agent, chain = response_agent
    state = {"draft_response": ABSTENTION, "query": "Unanswerable", "abstained": True}

    result = agent.invoke(state)

    assert result["final_response"] == ABSTENTION
    assert chain.calls == []


# ---------------------------------------------------------------------------
# 3. Configuration guards
# ---------------------------------------------------------------------------


def test_compl_ai_canned_answers_are_off_by_default():
    """Canned COMPL-AI answers bypass the workflow and fake telemetry."""
    assert config.COMPL_AI_ENABLED is False


def test_context_window_is_explicit_and_large_enough():
    assert config.LLM_NUM_CTX >= 8192


def _llm_param(llm, name):
    """Read a model parameter from either the real ChatOllama or the test stub."""
    kwargs = getattr(llm, "kwargs", None)
    if isinstance(kwargs, dict) and name in kwargs:
        return kwargs[name]
    return getattr(llm, name, None)


def test_agents_apply_num_ctx_and_runtime_seed(base_module, monkeypatch):
    """Agents must pass num_ctx to Ollama and rebuild when the seed changes.

    num_ctx and seed are both Ollama-only concepts (a hosted DeepSeek model has
    its own fixed context window and no documented seed support - see
    agents/base.py build_chat_llm), so this test forces the ollama provider
    regardless of what GENERATION_PROVIDER the environment/.env has active.
    """
    monkeypatch.setattr(config, "GENERATION_PROVIDER", "ollama")

    class Probe(base_module.BaseAgent):
        def invoke(self, state):  # pragma: no cover - not exercised
            return state

    agent = Probe(temperature=0.4)
    try:
        base_module.set_runtime_seed(1234)
        first = agent.llm
        assert _llm_param(first, "num_ctx") == config.LLM_NUM_CTX
        assert _llm_param(first, "seed") == 1234

        base_module.set_runtime_seed(1235)
        second = agent.llm
        assert second is not first, "changing the seed must rebuild the chat model"
        assert _llm_param(second, "seed") == 1235
    finally:
        base_module.set_runtime_seed(None)


# ---------------------------------------------------------------------------
# 4. SINGLE mode must run exactly one expert
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "route,expected",
    [
        ("legal", "legal"),
        ("news", "news"),
        ("general", "general_qa"),
        ("legal,news", "legal"),          # multi-label: primary label only
        ("news,legal", "news"),
        ("legal,news,general", "legal"),
        ("general,legal", "general_qa"),
        ("", "general_qa"),
        (None, "general_qa"),
        ("nonsense", "general_qa"),
    ],
)
def test_single_mode_selects_exactly_one_expert(route, expected):
    """SINGLE must never fan out, or the single-vs-multi claim is unfalsifiable."""
    from graph.routing import select_single_expert

    selected = select_single_expert(route)
    assert selected == expected
    assert isinstance(selected, str)  # a list here would mean parallel fan-out


# ---------------------------------------------------------------------------
# 5. Validator must not fail open
# ---------------------------------------------------------------------------


def test_unparseable_validator_output_is_a_failure(base_module):
    """A malformed judgement used to be silently recorded as PASS."""
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if "agents.validator" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "agents.validator", root / "agents" / "validator.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["agents.validator"] = module
        spec.loader.exec_module(module)
    module = sys.modules["agents.validator"]

    agent = module.ValidationAgent()

    garbage = agent._parse_validation("I think the answer looks fine overall.")
    assert garbage["pass"] is False
    assert garbage["parsed"] is False

    proper = agent._parse_validation("PASS: true\nISSUES: None\nSOURCE_RELEVANT: true")
    assert proper["pass"] is True
    assert proper["parsed"] is True

    explicit_fail = agent._parse_validation("PASS: false\nISSUES: Missing article citation")
    assert explicit_fail["pass"] is False
    assert explicit_fail["parsed"] is True


# ---------------------------------------------------------------------------
# 6. Statistics: the experimental unit is the query, not the repeat
# ---------------------------------------------------------------------------


def _mode_rows(mode, n_queries, repeats, judge_value, rng, noise=0.01, arm="peft", latency=20.0):
    rows = []
    for q in range(n_queries):
        for rep in range(repeats):
            rows.append(
                {
                    "query_id": f"q{q:02d}",
                    "query_type": "simple",
                    "mode": mode,
                    "arm": arm,
                    "repeat": rep,
                    "judge_average": judge_value + rng.normal(0, noise),
                    "rouge_l": 0.30 + rng.normal(0, noise),
                    "elapsed_s": latency + rng.normal(0, noise),
                    "cost": 0.001,
                    "abstained_flag": 0,
                    "judge_per_1k_tokens": judge_value / 2.0 + rng.normal(0, noise),
                }
            )
    return rows


def _synthetic_runs(n_queries, repeats, single_value, other_value, noise=0.01):
    """Two-topology frame where 'all' scores `single_value` and 'graph_engineering' `other_value`.

    Named after the two topologies that are actually compared now. `paired_tests`
    sorts mode names, so the comparison comes out as "all_vs_graph_engineering" and a positive
    Cliff's delta means ALL scored higher.
    """
    rng = np.random.default_rng(0)
    rows = _mode_rows("all", n_queries, repeats, single_value, rng, noise, latency=20.0)
    rows += _mode_rows("graph_engineering", n_queries, repeats, other_value, rng, noise, latency=40.0)
    return pd.DataFrame(rows)


def test_pairs_are_queries_not_repeats(analyze_module):
    """8 queries x 5 repeats must give 8 pairs, not 40.

    Pairing on (query_id, repeat) inflated n eightfold and broke the independence
    assumption of the Wilcoxon signed-rank test.
    """
    df = _synthetic_runs(n_queries=8, repeats=5, single_value=4.0, other_value=3.0)

    sig = analyze_module.paired_tests(df, analyze_module.TEST_METRICS)

    assert len(sig) == 1
    row = sig.iloc[0]
    assert row["comparison"] == "all_vs_graph_engineering"
    assert row["n_queries_paired"] == 8
    assert row["judge_average_n"] == 8
    assert row["judge_average_p"] < 0.05
    assert row["judge_average_cliffs_delta"] == pytest.approx(1.0)
    assert row["judge_average_underpowered"] is np.False_ or row["judge_average_underpowered"] is False


def test_all_topology_pairs_are_compared(analyze_module):
    """Three topologies must yield three comparisons, not two against a baseline.

    The pivot removed SINGLE from the compared set, so there is no baseline to
    test everything against; the design is now all-pairs among ALL / PARALLEL /
    Graph Engineering. A regression to baseline-style testing would silently drop the
    PARALLEL-vs-Graph Engineering comparison, which is the one the paper's ranking depends on.
    """
    rng = np.random.default_rng(1)
    rows = []
    for mode, value in (("all", 3.0), ("parallel", 3.5), ("graph_engineering", 4.0)):
        rows += _mode_rows(mode, 10, 3, value, rng)
    sig = analyze_module.paired_tests(pd.DataFrame(rows), analyze_module.TEST_METRICS)

    assert set(sig["comparison"]) == {"all_vs_graph_engineering", "all_vs_parallel", "graph_engineering_vs_parallel"}
    assert len(sig) == 3


def test_single_topology_subset_is_not_tested(analyze_module):
    """One topology cannot be compared with anything; return empty, not a self-test."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame(_mode_rows("all", 10, 3, 4.0, rng))

    assert analyze_module.paired_tests(df, analyze_module.TEST_METRICS).empty


def test_underpowered_comparison_is_not_reported_as_a_null_result(analyze_module):
    """With 3 queries, report not-tested (NaN) rather than p = 1.0."""
    df = _synthetic_runs(n_queries=3, repeats=5, single_value=4.0, other_value=3.0)

    sig = analyze_module.paired_tests(df, analyze_module.TEST_METRICS)
    row = sig.iloc[0]

    assert row["n_queries_paired"] == 3
    assert bool(row["judge_average_underpowered"]) is True
    assert np.isnan(row["judge_average_p"])
    assert np.isnan(row["judge_average_p_holm"])


def test_identical_outputs_are_not_reported_as_significant(analyze_module):
    """Deterministic decoding gives identical text; that is not a finding."""
    df = _synthetic_runs(n_queries=8, repeats=5, single_value=4.0, other_value=4.0, noise=0.0)

    sig = analyze_module.paired_tests(df, analyze_module.TEST_METRICS)
    row = sig.iloc[0]

    assert np.isnan(row["judge_average_p"])
    assert row["judge_average_median_diff"] == pytest.approx(0.0)


def test_rows_with_missing_judge_scores_are_dropped_from_judge_tests(analyze_module):
    """Failed judge calls (NaN) must shrink n, not be scored as 1/1/1."""
    df = _synthetic_runs(n_queries=8, repeats=5, single_value=4.0, other_value=3.0)
    df.loc[df["query_id"].isin(["q00", "q01"]), "judge_average"] = np.nan

    sig = analyze_module.paired_tests(df, analyze_module.TEST_METRICS)
    row = sig.iloc[0]

    assert row["judge_average_n"] == 6      # judge metric loses the excluded queries
    assert row["elapsed_s_n"] == 8          # latency keeps every query
    assert row["n_queries_paired"] == 8


def test_holm_correction_is_applied_across_mode_comparisons(analyze_module):
    """The family of pairwise topology comparisons must be family-wise corrected."""
    rng = np.random.default_rng(3)
    rows = []
    for mode, value in (("all", 4.0), ("parallel", 3.0), ("graph_engineering", 3.4)):
        rows += _mode_rows(mode, 10, 3, value, rng)

    sig = analyze_module.paired_tests(pd.DataFrame(rows), analyze_module.TEST_METRICS)

    assert len(sig) == 3
    for _, row in sig.iterrows():
        assert row["judge_average_p_holm"] >= row["judge_average_p"]


def test_cliffs_delta_direction_is_signed(analyze_module):
    """Positive delta must mean the LEFT side of the comparison scores higher."""
    better = analyze_module.cliffs_delta([5, 5, 5], [1, 1, 1])
    worse = analyze_module.cliffs_delta([1, 1, 1], [5, 5, 5])

    assert better == pytest.approx(1.0)
    assert worse == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 7. The PEFT ablation arm (RQ2)
# ---------------------------------------------------------------------------


def test_arm_ablation_is_paired_within_each_topology(analyze_module):
    """peft vs base must be tested per topology, paired on query.

    Pooling the arms across topologies would let a topology difference show up as
    a specialisation effect (and vice versa), which is exactly the confound the
    per-topology pairing exists to avoid.
    """
    rng = np.random.default_rng(4)
    rows = []
    for mode in ("all", "parallel", "graph_engineering"):
        rows += _mode_rows(mode, 10, 3, 4.0, rng, arm="peft")
        rows += _mode_rows(mode, 10, 3, 3.0, rng, arm="base")

    sig = analyze_module.arm_tests(pd.DataFrame(rows), analyze_module.TEST_METRICS)

    assert len(sig) == 3
    assert set(sig["mode"]) == {"all", "parallel", "graph_engineering"}
    for _, row in sig.iterrows():
        assert row["left"] == "peft" and row["right"] == "base"
        assert row["n_queries_paired"] == 10
        assert row["judge_average_p"] < 0.05
        # peft scored higher, so the signed effect size must be positive.
        assert row["judge_average_cliffs_delta"] == pytest.approx(1.0)
        assert row["judge_average_median_diff"] > 0


def test_single_arm_run_reports_no_ablation(analyze_module):
    """One arm means RQ2 is unanswerable; return empty rather than a fake control."""
    rng = np.random.default_rng(5)
    rows = []
    for mode in ("all", "parallel", "graph_engineering"):
        rows += _mode_rows(mode, 10, 3, 4.0, rng, arm="peft")

    assert analyze_module.arm_tests(pd.DataFrame(rows), analyze_module.TEST_METRICS).empty


def test_ablation_without_arm_column_is_not_invented(analyze_module):
    """A pre-pivot runs file has no arm; that must not silently produce a result."""
    df = _synthetic_runs(n_queries=8, repeats=3, single_value=4.0, other_value=3.0)
    df = df.drop(columns=["arm"])

    assert analyze_module.arm_tests(df, analyze_module.TEST_METRICS).empty


# ---------------------------------------------------------------------------
# 8. Benchmarked topology set and arm bookkeeping
# ---------------------------------------------------------------------------


def test_benchmarked_topologies_are_exactly_the_three_compared():
    """The paper compares ALL / PARALLEL / Graph Engineering and nothing else.

    SINGLE and the other topologies stay implemented (see the routing test above,
    which still passes) but must not re-enter the benchmark sweep: doing so would
    put topologies in the results that the paper does not describe, and would
    quietly change the size of the Holm-corrected family.
    """
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if "benchmark" not in sys.modules:
        spec = importlib.util.spec_from_file_location("benchmark", root / "benchmark.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["benchmark"] = module
        spec.loader.exec_module(module)
    benchmark = sys.modules["benchmark"]

    assert benchmark.MODES == ["all", "parallel", "graph_engineering"]
    assert "single" not in benchmark.MODES
    assert benchmark.ARMS == ["peft", "base"]


def test_local_peft_roles_are_three_distinct_model_families():
    """Each expert must have its own base model, or 'specialised agents' is a fiction.

    Two roles pointing at the same base model would mean two experts sharing
    weights, which quietly turns the three-model design into a two-model one and
    makes any per-role claim in the paper wrong.
    """
    assert set(config.LOCAL_PEFT_ROLES) == {"legal", "news", "general_qa"}
    base_models = [spec["base_model"] for spec in config.LOCAL_PEFT_ROLES.values()]
    assert len(set(base_models)) == 3, f"expected 3 distinct base models, got {base_models}"
    for role, spec in config.LOCAL_PEFT_ROLES.items():
        assert spec["adapter"], f"role {role} has no adapter path"
        assert spec["dataset"], f"role {role} has no fine-tuning dataset recorded"


def test_resume_key_includes_the_arm(tmp_path):
    """Resuming a two-arm run must not treat arm 2's cells as already done.

    The resume set is keyed on (query_id, mode, repeat, arm). Without the arm,
    arm 1's completed rows would match arm 2's planned cells and 270 runs would
    be silently skipped - the benchmark would report success having measured only
    half the design. Rows written before the pivot carry no arm and are treated as
    the peft arm's predecessors rather than matching every arm.
    """
    import json

    runs_file = tmp_path / "runs.jsonl"
    rows = [
        {"query_id": "q01", "mode": "all", "repeat": 0, "arm": "peft", "success": True},
        {"query_id": "q01", "mode": "all", "repeat": 0, "arm": "base", "success": True},
        {"query_id": "q02", "mode": "graph_engineering", "repeat": 1, "success": True},   # pre-pivot row
        {"query_id": "q03", "mode": "all", "repeat": 0, "arm": "peft", "success": False},
    ]
    runs_file.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    # Mirrors benchmark.py's resume parsing.
    existing = set()
    for line in runs_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        if data.get("success"):
            existing.add(
                (data["query_id"], data["mode"], data["repeat"], data.get("arm", "peft"))
            )

    assert ("q01", "all", 0, "peft") in existing
    assert ("q01", "all", 0, "base") in existing
    # A completed peft cell must NOT mask the same cell in the base arm.
    assert ("q03", "all", 0, "peft") not in existing, "failed runs must be retried"
    assert ("q03", "all", 0, "base") not in existing
    # Pre-pivot rows attach to peft only, never to every arm.
    assert ("q02", "graph_engineering", 1, "peft") in existing
    assert ("q02", "dag", 1, "base") not in existing


def test_arm_is_recorded_on_every_row():
    """Each row must carry its arm, or the ablation cannot be reconstructed.

    benchmark.py records the arm rather than inferring it later, because the two
    arms are indistinguishable from a row's contents alone - the same query, mode
    and repeat appear in both.
    """
    import importlib.util
    import inspect
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if "benchmark" not in sys.modules:
        spec = importlib.util.spec_from_file_location("benchmark", root / "benchmark.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["benchmark"] = module
        spec.loader.exec_module(module)
    benchmark = sys.modules["benchmark"]

    source = inspect.getsource(benchmark._execute_one)
    assert '"arm": arm' in source, "the success row must record the arm"
    # And the failure row too: a run that errored still belongs to an arm, and
    # dropping it there would make failure counts un-attributable.
    assert source.count('"arm": arm') >= 2, "the failure row must record the arm as well"

    # The per-request timeout must accommodate validator retries, which are the
    # common case rather than the exception on local models.
    assert benchmark.REQUEST_TIMEOUT_S >= 1800, benchmark.REQUEST_TIMEOUT_S


@pytest.mark.parametrize(
    "mode,validator_should_run",
    [
        ("all", False),
        ("parallel", False),
        ("legal_news_parallel", False),
        ("legal_first", False),
        ("verify_only", False),
        ("planner_based", False),
        ("graph_engineering", True),
        ("graph", True),
        ("dag", True),
    ],
)
def test_terminal_validator_runs_only_for_graph_engineering(monkeypatch, mode, validator_should_run):
    """Loop Engineering (the terminal validator/reflection pass) must be exclusive
    to graph_engineering, or it is a constant present in every arm rather than the
    thing the graph_engineering-vs-ALL/PARALLEL comparison is testing.

    Uses the same real-graph-with-stub-agents technique as
    test_truncation_warnings_survives_expert_node_narrowing so this exercises the
    actual conditional edges in graph/workflow.py, not a reimplementation of them.
    """
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    class _NoOpAgent:
        def __init__(self, *_args, **_kwargs):
            pass

        def invoke(self, state):
            return state

    agents_module = sys.modules["agents"]
    for name in (
        "PlannerAgent",
        "RouterAgent",
        "MemoryAgent",
        "RetrievalAgent",
        "LegalAgent",
        "NewsAgent",
        "GeneralQAAgent",
        "AggregatorAgent",
        "ValidationAgent",
        "ResponseAgent",
    ):
        monkeypatch.setattr(agents_module, name, _NoOpAgent, raising=False)

    sys.modules.pop("graph.workflow", None)
    spec = importlib.util.spec_from_file_location("graph.workflow", root / "graph" / "workflow.py")
    workflow = importlib.util.module_from_spec(spec)
    sys.modules["graph.workflow"] = workflow
    try:
        spec.loader.exec_module(workflow)

        compiled = workflow.create_legal_ai_graph()
        result = compiled.invoke({
            "query": "What are the obligations for high-risk AI systems?",
            "session_id": "test",
            "route": "legal",
            "expert_execution_mode": mode,
            "chat_history": [],
            "retrieved_docs": [],
            "agent_outputs": {},
            "agent_timings": {},
            "thinking_log": [],
            "validation_result": {},
        })

        ran = "validator" in result.get("agent_timings", {})
        assert ran is validator_should_run, (
            f"mode={mode!r}: expected validator-ran={validator_should_run}, got {ran}"
        )
    finally:
        sys.modules.pop("graph.workflow", None)


def test_live_legal_search_is_off_by_default():
    """A live EUR-Lex lookup makes runs non-reproducible; it must be opt-in.

    Same reasoning as fetch_news=False in the benchmark: if the legal expert's
    context depends on what the web returned that minute, two runs of the same
    query are not comparable and a topology difference may just be a fetch
    difference.
    """
    assert config.EURLEX_LIVE_SEARCH_ENABLED is False
