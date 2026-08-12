# P0 fixes applied — 2026-07-25

Addresses the blocking items from `DIAGNOSTIC_REPORT.md` §10 (P0). Every fix has a
regression test in `legalai/tests/test_experiment_validity.py` (15 tests, no Ollama
needed). Old invalid artifacts are archived under
`legalai/archive/pre_fix_20260725/`.

---

## 1. Aggregator abstention veto — the bug that produced the headline result

`agents/aggregator.py` replaced the *entire* draft with the 7-word abstention
sentence whenever **any** expert emitted it. Only the legal expert can emit it, so
every topology containing the legal expert was silenced while `single` (when routed
elsewhere) and `verify_only` were immune. The published "single agent wins" result
was largely a reproduction of this bug.

Now:

- the system abstains only when **every** expert that ran abstained;
- an abstaining expert is dropped from the aggregation and its abstention is passed
  to the aggregator prompt as a note, not as content;
- abstention is reported as data: `abstained`, `abstained_experts`, `experts_run`,
  `expert_abstention_rate` flow through the API into every benchmark row, and
  `abstained_flag` / `expert_abstention_rate` are aggregated per mode and tested for
  significance like any other metric.

`agents/response.py` no longer substring-matches the sentence, which used to discard
complete answers that merely quoted or referenced it.

## 2. COMPL-AI canned answers gated off

`compl_ai.py` matched bare substrings ("obligation", "evidence") and returned a
fixed template with fabricated telemetry (`backend_ms = 1.0`), bypassing the whole
workflow — it hit q08/q11/q13 identically in all 8 modes.

- New flag `LEGALAI_ENABLE_COMPL_AI`, **off by default** (`config.COMPL_AI_ENABLED`).
- `benchmark.py` forces it off in the environment the server inherits, and a
  **preflight check aborts the run** if `/config` reports it enabled — so this can
  never silently contaminate an experiment again.

## 3. Context window set explicitly

`LEGALAI_NUM_CTX` (default **8192**) is now passed to every agent's `ChatOllama`.
Previously Ollama fell back to its small default and truncated long retrieval
contexts with no error. Any prompt within 10% of `num_ctx` now logs a warning and
records a `truncation_warnings` entry on the run row.

## 4. Determinism / repeats — why the CIs were ±0.00

With `LEGALAI_DETERMINISTIC=1` decoding is greedy, so all 5 repeats returned
byte-identical text: every text-quality CI collapsed to ±0.00 and every paired test
degenerated.

- Per-request `seed` field on `/chat` and `/chat/stream`; agents rebuild their chat
  model when the runtime seed changes (`agents/base.set_runtime_seed`).
- `benchmark.py` sends `seed = 1000 + repeat`, defaults the server to
  `LEGALAI_DETERMINISTIC=0`, and warns if a deterministic server is used with
  repeats > 1.
- Two documented regimes: **greedy** (repeats measure latency variance only) or
  **sampling with per-repeat seeds** (repeats measure generation variance — use this
  for the paper's confidence intervals).

## 5. Statistics

`analyze_results.py`:

- **The experimental unit is now the query.** Repeats are averaged within each
  `(query_id, mode)` cell before pairing, instead of pairing on
  `(query_id, repeat)` — which inflated n fivefold and broke the independence
  assumption of the Wilcoxon signed-rank test.
- Comparisons with fewer than 6 paired queries are reported as **not-tested (NaN)**,
  never as `p = 1.0`, which reads as evidence of no difference when it is absence of
  data.
- **H2 interaction tests added**: the same paired tests are now run *within each
  query type* (`simple` / `decomposable` / `routing`). This is where a multi-agent
  advantage should appear if it exists at all; a pooled test hides it.
- Failed LLM-judge calls are excluded from judge metrics instead of being scored as
  a real 1/1/1 (`llm_judge.judge` now returns `ok: False` and does not cache
  failures). They still contribute latency and cost data.
- Holm correction applies only across tests that actually ran; Cliff's delta,
  per-metric n, median difference, and an `_underpowered` flag are reported.
- `llm_judge` warns when `JUDGE_MODEL == GOLD_MODEL` (self-preference bias — both
  currently default to `llama3.1:8b`).
- Duplicate metric entries removed (they produced duplicate columns in
  `by_query_type.csv`).

## 6. Benchmark harness

- Smoke runs write to `benchmark_runs_smoke.jsonl` / `run_meta_smoke.json`, so they
  can never be mixed into or mistaken for a full run.
- One discarded warm-up request per mode, so first-call model loading is not
  measured.
- `run_meta.json` records **measured** duration, seeds, planned vs completed runs,
  server config, env vars (including `OLLAMA_NUM_PARALLEL`), and platform — replacing
  the hardcoded `duration_s: 300.0`.
- Rows now carry `seed`, abstention fields, truncation warnings, and gold provenance
  (`gold_status`, `gold_needs_review`).

## 7. LaTeX table

`evaluate_workflows.generate_latex_table`:

- **Backend latency was printed in milliseconds under a "(s)" label** — now scaled
  (67520 ms → 67.52 s), with the CI scaled to match.
- Real measured `prompt_tokens` / `completion_tokens` rows replace the
  `words × 1.3` estimate.
- Abstention-rate rows added; missing metrics are skipped rather than crashing;
  caption states that ± is a 95% CI margin.

---

## How to produce defensible numbers

```powershell
cd "C:\Users\Charbel\Desktop\Legal AI\legalai"
.\.venv\Scripts\Activate.ps1

# 0. sanity check the fixes (fast, no Ollama)
python -m pytest tests -q

# 1. environment for a variance-bearing run
$env:LEGALAI_DETERMINISTIC = "0"     # sample, so repeats differ
$env:LEGALAI_NUM_CTX       = "8192"
$env:LEGALAI_ENABLE_COMPL_AI = "0"   # benchmark.py enforces this anyway
$env:OLLAMA_NUM_PARALLEL   = "4"     # or parallel topologies are secretly serialized
$env:JUDGE_MODEL           = "qwen2.5:7b"   # must differ from the gold-answer model

# 2. smoke test first (8 runs, separate output file)
python benchmark.py --smoke

# 3. full run: 30 queries x 8 modes x 5 repeats = 1200 runs
python benchmark.py            # --resume if it dies partway

# 4. analysis -> analysis_summary.csv, by_query_type.csv, significance.csv, results.json
python analyze_results.py

# 5. figures + LaTeX table from that same run
python evaluate_workflows.py
python scripts/package_results.py
```

Budget the wall-clock: at ~35 s/run, 1200 runs is roughly 12 hours. Run it once,
overnight, and do not mix it with anything else.

---

---

# Round 2 — 2026-07-26

## 8. SINGLE is now a true single agent

`graph/workflow.py` fanned out to 2–3 experts in parallel whenever the router
emitted a multi-label route, so the "single agent" baseline was really a small
ensemble and the central claim was unfalsifiable. SINGLE now runs exactly one
expert — the router's primary label. The decision logic moved to
`graph/routing.py::select_single_expert` so it can be unit tested without
langgraph; 10 parametrised cases cover it.

## 9. Judge / gold / system model separation

`llm_judge.py` now warns when the judge shares a family with either the model
under test or the model that drafted the reference answers. All three are
recorded in `results.json` under `provenance`, so any published score can be
traced to the models that produced it.

## 10. Vector-store wipe fixed

`embed.py` deleted **every** id in the collection on a news fetch, destroying the
EU AI Act corpus — after which every legal query abstains legitimately and the
run looks like a modelling result. It now deletes only chunks with
`source_type == "news"`, reports how many non-news chunks were preserved, and
warns loudly if the collection ends up empty.

## 11. Validator no longer fails open

`agents/validator.py` read `metadata['source']` when ingestion writes `name`, so
the validator judged source relevance against a list of "Unknown"s. It now reads
the correct key. Unparseable validator output was silently treated as PASS; it is
now recorded as FAIL with the raw output logged and a `validator_parse_failures`
counter on the state.

## 12. Figures consolidated for Overleaf

New `make_paper_figures.py` builds **four** composite PNGs from
`analysis_summary.csv` and `by_query_type.csv`, replacing the ~15 separate images
the draft pulled in through nested `subfigure` environments — which is what was
hitting your compile timeout:

| Figure | Shows | Why it earns its place |
|---|---|---|
| `fig1_operational.png` | latency, per-node decomposition, steps, real token cost | attributes cost to a specific stage; the table can only give totals |
| `fig2_quality.png` | BLEU, ROUGE, judge scores, abstention rate | puts lexical and judged rankings side by side so disagreement is visible |
| `fig3_tradeoff.png` | quality vs latency, with the "slower and no better" region shaded | converts per-metric columns into a decision |
| `fig4_query_type.png` | quality by topology × task class | the H2 interaction, invisible in any pooled table |

The 31 diagnostic charts still generate; they just are not in the paper.

## 13. Paper rewritten — `main.tex` + `references.bib`

Every false claim is gone: `verify_only` no longer "skips retrieval" (retrieval
runs, experts are skipped), planner-based and DAG are promoted out of future work
into the main comparison, the "32 metrics" claim is reconciled, the gold standard
is no longer called "expert-curated", and SINGLE is described honestly. Added: a
methodology section that defines planner vs router and specifies all eight
topologies, an explicit statistical protocol, a reproducibility section, a
threats-to-validity section, and 31 verified citations.

**Every number is a red `\TBD{}` placeholder.** Nothing citable is left in the
file. `metrics_table.tex` is `\input{}`, so the generated table drops in.

## 14. One-command run

`run_experiment.ps1` sets the environment, verifies Ollama and the judge model,
runs the validity tests, benchmarks, analyses, builds figures and prints a
significance summary. `-Smoke` for a dry run first.

---

## The literature check changes the paper's positioning

I verified the prior work (see `CITATION_NOTES.md`). Two findings you need before
you write another word of the paper:

**"A single agent can beat multi-agent" is already published — three times in
2026 alone.** Jwalapuram et al. (arXiv 2606.13003), Tran & Kiela (arXiv
2604.02460) and Xu et al. (arXiv 2601.12307) all report it, on top of the 2024
scaling-law caveats. Leading with this as the headline finding invites a
"known result" rejection.

**"Topology matters" is an established subfield, not a gap.** GPTSwarm, MacNet,
DyLAN, AgentPrune, G-Designer, ARG-Designer and AgentDropout are all real papers
whose entire contribution is topology design, and there is already a 2026 survey
of the area. The claim that "there is nothing in the literature about how to
structure agents" is not defensible.

What survives, and what `main.tex` now argues:

1. **Domain.** Nobody has run this comparison on retrieval-grounded regulatory
   compliance QA, where abstention is correct behaviour rather than failure.
2. **A retrieval-matched single-agent baseline.** The topology papers compare
   multi-agent systems to each other; none benchmarks a learned topology against
   a single agent given identical retrieval.
3. **Abstention propagation.** The aggregator bug is itself a publishable
   structural failure mode: one abstaining expert silences an ensemble unless the
   aggregator is designed against it, which biases every naive single-vs-multi
   comparison in a domain that has an abstention mechanism.
4. **The task-class interaction.** Whether the multi-agent gap narrows on
   decomposable queries is the one question the prior work leaves open here.

If the interaction in item 4 shows up in your data, lead with it.

---

## Still yours to do (not code fixes)

1. **Gold answers.** All 30 are `"needs_review": true` and were drafted by
   `llama3.1:8b`. Human-review them, and stop calling them "expert-curated" in the
   paper until you have. Set `JUDGE_MODEL` to a different model than the drafter.
2. **Retrieval metrics** (P@5, R@5, MRR) are self-referential — annotate true
   relevant chunks or cut the retrieval claims from the paper.
3. **Vector-store wipe** (`DIAGNOSTIC_REPORT.md` §3.1): the first news fetch can
   destroy the EU AI Act knowledge base. Fix before any long unattended run, or run
   with `fetch_news=False` throughout (the benchmark already does).
4. **Paper edits after the re-run:** promote `planner_based` and `dag` out of future
   work (both are implemented and measured), fix the `verify_only` description
   (retrieval *does* run; experts are skipped), add citations and a bibliography,
   add a reproducibility paragraph (model, quantization, hardware, Ollama version,
   `OLLAMA_NUM_PARALLEL`, seeds, N/R), reconcile the "32 metrics" claim, and either
   force one expert in `single` mode or rename it honestly — it currently fans out
   2–3 experts on multi-label routes.
