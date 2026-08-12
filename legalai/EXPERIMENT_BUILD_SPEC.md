# Build Spec — Single-Agent vs Multi-Agent Topology Experiment

**Audience:** an AI coding agent implementing changes in this repository (`legalai/`).
**Goal:** turn the current one-query, one-run demo benchmark into a statistically valid
experiment that compares a single-agent router against several multi-agent topologies,
then package it so it runs reproducibly (local + Docker + optional HPC batch).

Implement in the order given. This now **includes** authoring the gold answers (§13) and
running the full benchmark against live Ollama (§14). Everything should be code-complete
and pass the acceptance criteria in §9.

---

## 0. Current state (read before changing)

- `benchmark.py` — runs **one hard-coded query**, once, across 6 modes; writes
  `benchmark_results.json` (a dict keyed by mode).
- `evaluate_workflows.py` — scores each mode's single response against one hard-coded
  `GOLD_STANDARD` string; generates 29 charts + a LaTeX table. Metric functions
  (`calculate_bleu`, `calculate_rouge_n`, `calculate_rouge_l`, `word_jaccard`,
  `char_jaccard`, `cosine_similarity_tf`, `levenshtein_similarity`) are correct and reusable.
- `graph/workflow.py` — builds the LangGraph. Topology is chosen by
  `state["expert_execution_mode"]` inside `route_from_retrieval`, `route_after_legal`,
  `route_after_news`. Sequential-only by design ("avoiding parallel branches for state safety").
- `agents/base.py` — `BaseAgent.__init__` builds `ChatOllama(model, base_url, temperature)`
  with **no seed and no num_predict** (non-reproducible).
- `backend/models.py` — `expert_execution_mode` is a `Literal[...]` of the 6 modes; response
  models define returned fields.
- `backend/service.py` — `process_query` streams the graph; `_normalize_expert_mode` (line ~107)
  whitelists the 6 modes; the return dict (lines ~483–497) is what `/chat` sends back.
- `config.py` — env-driven config + all agent prompts. `EXPERT_EXECUTION_MODE` whitelist is
  only `{"all","single"}` (stale).
- Cost (`$0.0001` constant) and tokens (`word_count*1.3`) in `evaluate_workflows.py` are
  **fabricated proxies** — replace with real values (§4).

---

## 1. Research design this code must support

Hypotheses the output must be able to confirm/reject:
- **H1 (non-inferiority):** single-agent is not worse than the best multi-agent topology on
  answer quality.
- **H2 (interaction):** multi-agent beats single only for *decomposable* queries, not *simple* ones.
- **H3 (overhead):** latency/cost rises with coordination without a matching quality gain.

Design: same model, corpus, decoding, prompts across all topologies; vary only the graph.
Test set of N≥30 queries (target 100), balanced across query types, each with a gold answer
and a gold set of source doc IDs. Each (query × mode) run R=5 times.

---

## 2. New file — `eval_dataset.json`

Create a template with 6 example rows (human fills the rest + gold text). Schema:

```json
[
  {
    "id": "q01",
    "type": "simple",
    "query": "What is a high-risk AI system under the EU AI Act?",
    "gold": "<expert-written reference answer>",
    "gold_doc_ids": ["eu_ai_act_art6", "eu_ai_act_annex3"]
  }
]
```

- `type` ∈ {`simple`, `decomposable`, `routing`}. Keep counts roughly balanced.
- `gold_doc_ids` = the chunk/source IDs that *should* be retrieved (for retrieval precision/recall).
  Use the same ID scheme the retrieval layer emits (see §4). If unknown at authoring time,
  allow an empty list and skip retrieval metrics for that row.

Add JSON-schema validation in `analyze_results.py` startup; fail fast on malformed rows.

---

## 3. Reproducibility — `agents/base.py` + `config.py`

In `config.py` add:
```python
LLM_SEED = int(os.getenv("LEGALAI_LLM_SEED", "42"))
LLM_NUM_PREDICT = int(os.getenv("LEGALAI_NUM_PREDICT", "1024"))
DETERMINISTIC = os.getenv("LEGALAI_DETERMINISTIC", "1") == "1"
```
Extend the `EXPERT_EXECUTION_MODE` whitelist to all 8 modes (§5).

In `agents/base.py` `__init__`, pass seed + num_predict so runs are repeatable:
```python
self.llm = ChatOllama(
    model=model or config.OLLAMA_MODEL,
    base_url=config.OLLAMA_BASE_URL,
    temperature=0.0 if config.DETERMINISTIC else temperature,
    seed=config.LLM_SEED if config.DETERMINISTIC else None,
    num_predict=config.LLM_NUM_PREDICT,
)
```
Keep the per-agent `temperature` argument, but when `DETERMINISTIC=1` it is overridden to 0.0
so the main comparison is noise-free. (A separate consistency pass can set `DETERMINISTIC=0`.)

---

## 4. Backend — return real metrics (`backend/service.py`, `backend/models.py`)

The experiment needs real tokens and retrieved doc IDs, not proxies.

1. **Retrieved doc IDs.** In `process_query`, after the graph runs, read the final state's
   `retrieved_docs` and emit their stable IDs. Add to the `/chat` return dict:
   `"retrieved_ids": [doc.metadata["id"] for doc in final_state.get("retrieved_docs", [])]`.
   Ensure the retrieval layer (`agents/retrieval.py`) attaches a stable `id` in each doc's
   metadata (chunk id or source filename + chunk index). Document the scheme in the README.
2. **Token counts.** `ChatOllama` responses expose `response_metadata` with
   `prompt_eval_count` and `eval_count`. Accumulate these per node into the state (extend
   `record_agent_timing` pattern with a parallel `record_agent_tokens`), then return totals:
   `"prompt_tokens": ..., "completion_tokens": ...`.
3. **Models.** Add `retrieved_ids: list[str]`, `prompt_tokens: int | None`,
   `completion_tokens: int | None` to the `/chat` response model in `backend/models.py`.
4. **Cost.** Compute a real proxy from tokens in the analysis layer (configurable
   $/1k in/out rates), not a constant.

---

## 5. Add two topologies — `graph/workflow.py`, `backend/models.py`, `backend/service.py`, `config.py`, `agents/planner.py`, `state.py`

Add modes `"planner_based"` and `"dag"`.

1. `backend/models.py` line ~18 — extend the `Literal[...]` with `"planner_based", "dag"`.
2. `backend/service.py` `_normalize_expert_mode` (line ~110) — add both to the allowed set.
3. `config.py` — add both to the `EXPERT_EXECUTION_MODE` whitelist.
4. `graph/workflow.py` `_effective_expert_mode` (line ~262) — add both to the valid set.
5. `route_from_retrieval` (line ~268) — add branches:
```python
elif mode == "planner_based":
    return state.get("plan") or ["legal"]      # planner decided which experts
elif mode == "dag":
    return ["legal", "news"]                    # then fan into general_qa via route_after_news
```
6. **Planner that actually plans.** In `agents/planner.py`, when mode is `planner_based`,
   call the LLM (use `PLANNER_PROMPT`, extend it to ask for a JSON list of needed experts from
   {legal, news, general_qa}) and store `state["plan"] = [...]`. Add `plan: list[str]` to
   `state.py` `AgentState`.
7. **Parallel-safe state.** True parallel fan-out (`parallel`, `legal_news_parallel`,
   `planner_based`, `dag`) requires concurrent writes to `agent_outputs` to merge, not clobber.
   In `state.py`, annotate the field with a reducer:
```python
from typing import Annotated
def _merge_dicts(a: dict, b: dict) -> dict: return {**(a or {}), **(b or {})}
agent_outputs: Annotated[dict, _merge_dicts]
```
   Do the same for `agent_timings` and `thinking_log` (list concat reducer). Verify the existing
   "parallel" modes still pass after this change.
8. For `dag`, customize `route_after_legal`/`route_after_news` so the legal+news results
   converge into `general_qa` before `aggregator`, giving a genuine diamond dependency graph
   distinct from flat `parallel`.

---

## 6. Rewrite `benchmark.py` — dataset × modes × repeats → JSONL

Replace the single-query block (lines ~45–86). Keep the server start/stop logic.

```python
import itertools
REPEATS = int(os.getenv("BENCH_REPEATS", "5"))
MODES = ["all","single","parallel","legal_news_parallel","legal_first",
         "verify_only","planner_based","dag"]

with open(ROOT_DIR/"eval_dataset.json", encoding="utf-8") as f:
    dataset = json.load(f)

with open(ROOT_DIR/"benchmark_runs.jsonl", "w", encoding="utf-8") as out:
    for item, mode, rep in itertools.product(dataset, MODES, range(REPEATS)):
        payload = {"message": item["query"],
                   "session_id": f"bench_{mode}_{item['id']}_{rep}",  # unique => no memory carryover
                   "fetch_news": False, "expert_execution_mode": mode}
        start = time.perf_counter()
        try:
            data = requests.post(URL, json=payload, timeout=180).json()
            row = {"query_id": item["id"], "query_type": item["type"], "mode": mode,
                   "repeat": rep, "gold": item["gold"],
                   "gold_doc_ids": item.get("gold_doc_ids", []),
                   "response": data.get("response",""),
                   "elapsed_s": round(time.perf_counter()-start,3),
                   "backend_ms": data.get("workflow_elapsed_ms"),
                   "route": data.get("route"), "timings": data.get("agent_timings_ms",{}),
                   "steps": len(data.get("thinking_log",[])),
                   "retrieved_ids": data.get("retrieved_ids",[]),
                   "prompt_tokens": data.get("prompt_tokens"),
                   "completion_tokens": data.get("completion_tokens"),
                   "success": True}
        except Exception as e:
            row = {"query_id": item["id"], "mode": mode, "repeat": rep,
                   "success": False, "error": str(e)}
        out.write(json.dumps(row, ensure_ascii=False)+"\n"); out.flush()
```
Add `--resume` support (skip rows already present in `benchmark_runs.jsonl`) so long runs
survive interruption. Add a `--smoke` flag that runs 1 query × all modes × 1 repeat.

---

## 7. New file — `llm_judge.py` (correctness scorer)

BLEU/ROUGE only measure word overlap. Add an LLM-as-judge for correctness.

- Function `judge(query, gold, answer) -> dict` returning integer 1–5 scores for
  `accuracy`, `completeness`, `groundedness`, plus a one-line rationale.
- Use a **different / stronger** model than the system-under-test (configurable
  `JUDGE_MODEL` env), temperature 0, strict JSON output, retry-on-parse-fail.
- Cache by `hash(query+gold+answer)` to avoid re-scoring identical text.
- Provide `--validate` mode that prints judge-vs-human agreement when a human-scored
  subset CSV is supplied (human authors ~20 rows).

---

## 8. New file — `analyze_results.py` (aggregation + statistics)

Consumes `benchmark_runs.jsonl`. Reuse metric functions imported from `evaluate_workflows.py`.

Required steps:
1. Load JSONL → DataFrame; drop `success=False` (report their count).
2. **Per-run, per-query-gold** quality: compute BLEU-1, ROUGE-1/2/L, Jaccard, cosine,
   Levenshtein of each `response` vs **its own `gold`** (not a global string).
3. Add `llm_judge` scores (call `llm_judge.judge`).
4. **Retrieval metrics:** precision@k, recall@k, MRR of `retrieved_ids` vs `gold_doc_ids`.
5. **Operational:** elapsed_s, backend_ms, steps, real tokens, computed cost.
6. **Aggregate per mode:** mean, std, n, **95% CI** for every metric.
7. **Paired significance** single vs each other mode, matched on `(query_id, repeat)`:
   Wilcoxon signed-rank; **Holm-correct** the p-values across the family of comparisons;
   report **Cliff's delta** effect size.
8. **Interaction (H2):** group means by `(query_type, mode)`; emit the table and a grouped bar chart.
9. **Consistency:** per `(query_id, mode)` variance of quality across repeats.
10. Write outputs: `analysis_summary.csv` (mode × metric mean±CI), `significance.csv`
    (pairwise p, p_holm, cliff_delta), `by_query_type.csv`, and a `results.json` bundle.

Keep `evaluate_workflows.py` for plotting, but **feed it the aggregated means** (refactor its
`main()` to accept a summary DataFrame instead of recomputing from a single response). All
bar charts must render **error bars** from the 95% CI; chart 28 (quality-vs-latency) must plot
CI whiskers on both axes.

---

## 9. Acceptance criteria (definition of done)

- [ ] `python benchmark.py --smoke` produces `benchmark_runs.jsonl` with one row per mode and
      all 8 modes present, each `success=True`, each carrying non-null `retrieved_ids` and
      token counts.
- [ ] Re-running `--smoke` twice with `LEGALAI_DETERMINISTIC=1` yields **identical responses**
      for the same (query, mode) — proves seeding works.
- [ ] `planner_based` and `dag` modes execute end-to-end; `dag` shows legal+news converging into
      general_qa in `thinking_log`; no concurrent-write errors from the reducers.
- [ ] `analyze_results.py` runs on a ≥2-query, ≥2-repeat sample and emits all four output files,
      with mean±CI, Holm-corrected p-values, and Cliff's delta populated.
- [ ] `by_query_type.csv` contains a mode × query_type matrix (enables the H2 claim).
- [ ] All charts regenerate with error bars; no chart depends on the old single-`GOLD_STANDARD`.
- [ ] No fabricated metrics remain: cost and tokens trace to real values.
- [ ] Unit tests: metric functions (known inputs), retrieval precision/recall, Cliff's delta,
      JSONL round-trip, dataset schema validation. `pytest` green.
- [ ] README updated (§ below). `docker compose up` still serves the app and `/health` is green.
- [ ] `eval_dataset.json` fully populated: ≥30 rows (target 100), balanced across query types,
      every row has a source-grounded `gold`, a non-empty `gold_doc_ids`, and provenance fields (§13).
- [ ] Every `gold_doc_id` resolves to a real chunk in the vector store (validator passes).
- [ ] Full benchmark completed: `benchmark_runs.jsonl` has N×8×5 rows with **zero** `success=false`
      (failures investigated/re-run), and `analyze_results.py` + `evaluate_workflows.py` outputs plus
      `run_meta.json` are written to `results/` (§14).

---

## 10. Out of scope for the agent (human-owned)

- Authoring the ~20-row human-scored subset to validate the judge (needs a legal SME).
- **Final legal sign-off** on the agent-drafted gold answers — the agent drafts and
  self-checks them (§13), a human SME approves before results are published.
- Securing HPC/GPU for larger-model passes.

---

## 11. Deployment & run instructions (add to README)

### Local (dev)
```bash
# 1. Ollama with pinned models
ollama pull qwen2.5
ollama pull nomic-embed-text
ollama pull llama3.1:8b           # JUDGE_MODEL (use a different/stronger model)

# 2. Python env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# add to requirements.txt: scipy, statsmodels, pytest

# 3. Reproducible run
export LEGALAI_DETERMINISTIC=1 LEGALAI_LLM_SEED=42 JUDGE_MODEL=llama3.1:8b
python benchmark.py --smoke           # sanity
python benchmark.py                   # full run -> benchmark_runs.jsonl
python analyze_results.py             # -> analysis_summary.csv, significance.csv, by_query_type.csv
python evaluate_workflows.py          # -> charts in evaluation_assets/ + metrics_table.tex
```

### Docker
- Add the analysis deps to `requirements.txt` (rebuild image).
- The benchmark talks to the API over HTTP; either run `benchmark.py` on the host against the
  container's `:8000`, or add a one-shot `benchmark` service to `docker-compose.yml`:
```yaml
  benchmark:
    build: { context: ., dockerfile: Dockerfile }
    depends_on: { legalai: { condition: service_healthy } }
    environment:
      URL_BASE: http://legalai:8000
      LEGALAI_DETERMINISTIC: "1"
      JUDGE_MODEL: llama3.1:8b
    volumes:
      - ./eval_dataset.json:/app/eval_dataset.json
      - ./results:/app/results          # persist benchmark_runs.jsonl + csv outputs
    profiles: ["bench"]                  # only runs with: docker compose --profile bench up
    command: ["python", "benchmark.py"]
```
  Parameterize `URL` in `benchmark.py` from `URL_BASE` env (default `http://127.0.0.1:8000`).
- Ollama stays on the host; container reaches it via the existing
  `OLLAMA_BASE_URL=http://host.docker.internal:11434`.

### HPC / larger models (optional, for the "bigger models" experiment)
- Make `OLLAMA_MODEL` and `JUDGE_MODEL` fully env-driven (already mostly true).
- Provide a SLURM batch script `scripts/run_benchmark.sbatch` that: loads the module/conda env,
  starts Ollama (or points to a node-local server), pulls the target model, then runs
  `benchmark.py` + `analyze_results.py`, writing to a job-scoped `results/<jobid>/` dir.
- Keep `--resume` so requeued jobs continue. Record `OLLAMA_MODEL`, seed, git commit, and
  timestamp into `results/<jobid>/run_meta.json` for provenance.

---

## 12. Suggested build order
1. §3 reproducibility + §4 real metrics (backend).  2. §6 benchmark loop + §2 dataset template.
3. §8 analysis + stats (works on smoke data).  4. §7 judge.  5. §5 planner/DAG topologies.
6. §8 chart refactor with error bars.  7. tests + README + Docker (§9, §11).
8. §13 author + validate the gold dataset.  9. §14 run the full benchmark + analysis, commit `results/`.

---

## 13. Gold answer authoring (agent task)

Produce a source-grounded `gold` and `gold_doc_ids` for **every** row of `eval_dataset.json`.
Golds must be derived from authoritative text, never invented.

**Procedure (script `scripts/build_gold.py`):**
1. Load each query. Run the **retrieval layer** (`agents/retrieval.py`) against the existing
   corpus (`articles/` + `chroma_storage/`) to get the top candidate passages **and their stable
   chunk IDs** (the same ID scheme exposed in §4).
2. Draft the gold answer with a **strong model** (`JUDGE_MODEL` or larger), instructed to use
   **only** the retrieved passages / official EU AI Act text, cite specific Articles, and abstain
   if support is missing. Reuse the hand-written `GOLD_STANDARD` block in `evaluate_workflows.py`
   (lines ~27–55) as the style/scope exemplar for compliance-type queries.
3. Match the answer shape to the query `type`: IRAC-style and Article-cited for `legal`/
   `decomposable`; short and factual for `simple`; route-appropriate for `routing`.
4. Set `gold_doc_ids` to the chunk IDs that actually support each claim (a subset of step 1).
5. Write back into `eval_dataset.json` adding provenance fields per row:
   `"gold_status": "draft"`, `"needs_review": true`, `"gold_model": "<model>"`,
   `"gold_sources": [<doc ids cited>]`.

**Self-check (must pass before §14):**
- Every sentence of a gold maps to at least one cited Article or `gold_doc_id`; flag any that don't.
- `gold` is non-empty; `gold_doc_ids` is non-empty and **every ID exists in the vector store**.
- Query types are balanced (no type < 20% of rows).
- No claim contradicts the retrieved text (run the `llm_judge` groundedness check of each gold
  against its own sources; reject golds scoring < 4/5 and redraft).

Leave `needs_review: true` so a human SME can approve; the agent does **not** flip it to false.

---

## 14. Full benchmark execution (agent task)

Run the complete experiment against live Ollama and commit the artifacts.

**Preconditions (assert and fail fast):**
- Ollama reachable at `OLLAMA_BASE_URL`; `OLLAMA_MODEL`, `OLLAMA_EMBEDDING_MODEL`, and
  `JUDGE_MODEL` all pulled (`ollama list`). Pull any missing model.
- `eval_dataset.json` passes the §13 validator. API `/health` is green.
- `LEGALAI_DETERMINISTIC=1`, `LEGALAI_LLM_SEED=42` exported.

**Run (script `scripts/run_full.sh`):**
```bash
export LEGALAI_DETERMINISTIC=1 LEGALAI_LLM_SEED=42 JUDGE_MODEL=${JUDGE_MODEL:-llama3.1:8b}
python benchmark.py --smoke            # gate: must be all-green before the full run
python benchmark.py --resume           # full N x 8 modes x 5 repeats -> benchmark_runs.jsonl
python analyze_results.py              # -> analysis_summary.csv, significance.csv, by_query_type.csv
python evaluate_workflows.py           # -> charts + metrics_table.tex
```

**During/after:**
- Use `--resume` so the run survives interruption (long: N×8×5 model calls — expect hours; run
  unattended/overnight and log progress + ETA).
- Re-run any `success=false` rows until zero remain, or record why a row is unrecoverable.
- Write `results/run_meta.json` with: `OLLAMA_MODEL`, `JUDGE_MODEL`, seed, git commit hash,
  SHA-256 of `eval_dataset.json`, start/end timestamps, row counts, and library versions.
- Copy `benchmark_runs.jsonl`, all CSV/JSON outputs, charts, and `metrics_table.tex` into
  `results/<timestamp>/` and commit them.

**Done when:** `results/<timestamp>/` contains the run with zero failed rows, the four analysis
outputs, regenerated charts with error bars, and `run_meta.json`.
