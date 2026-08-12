# Legal AI — Full Project Diagnostic Report

**Scope:** Every source file, data artifact, document, and configuration in `C:\Users\Charbel\Desktop\Legal AI` (root + `legalai/`).
**Date:** 2026-07-18.
**Method:** Complete manual read of all 40+ source files; forensic cross-checking of the benchmark data (`benchmark_runs.jsonl`, 324 rows), the published tables/charts, both paper drafts, the proposal, and the advisor feedback documents; hash comparisons, dataset statistics, and syntax checks (`compileall` clean, Python 3.13.7).

---

## 0. Executive verdict

The engineering skeleton of this project is genuinely good: a clean modular multi-agent LangGraph pipeline, a well-structured FastAPI backend, a capable React frontend, a benchmark harness with resume support, and an analysis script that (on paper) does the right statistics (Wilcoxon signed-rank, Holm correction, Cliff's delta, 95% CIs). The experiment *design* in `EXPERIMENT_BUILD_SPEC.md` (H1 non-inferiority / H2 interaction / H3 overhead, N≥30, R=5) is publication-worthy.

**However, the experiment as executed does not currently support any claim in the paper.** The evidence chain is broken in five independent places, any one of which a reviewer would treat as disqualifying:

1. The paper's results table is built from **N = 1** (one query, one repeat per mode, 6 of 8 modes) — `benchmark_results.json`.
2. The committed "publication-grade" charts, `metrics_table.tex`, and `significance.csv` come from a **different, degenerate smoke run** (n = 1 per mode) in which **7 of 8 topologies returned the literal abstention string** `"Insufficient authoritative support -- recommend expert review."` (word count = 7, ROUGE-L = 0.0, all p-values = 1.0).
3. The **real** benchmark data (324 runs, 9 queries × 8 modes × 5 repeats) **was never analyzed** — `analyze_results.py` was last run before it existed (judge cache holds 2 entries).
4. The current 30-query dataset **has never been benchmarked at all**, and its gold answers are machine-drafted (`llama3.1:8b`), all flagged `"needs_review": true` — while the paper calls them "expert-curated."
5. **10% of benchmark queries (q08, q11, q13) are hijacked** by a hardcoded keyword-matched template (`compl_ai.py`) that bypasses the entire multi-agent workflow and reports fabricated timings (`{"compl_ai": 1.0}`).

The good news: the paper's *thesis* ("a single agent is sometimes better; structure matters more than agent count") is plausible, the advisor's restructuring is already largely implemented in `main.tex.txt`, and the infrastructure to produce defensible numbers already exists. What's needed is a set of targeted bug fixes followed by **one clean, full re-run** and a rewrite of the results section from that run. A concrete roadmap is in §10.

---

## 1. What the project actually is

- **Application:** local-first (Ollama: `qwen2.5` chat + `nomic-embed-text` embeddings) multi-agent RAG assistant for EU AI Act compliance Q&A. ChromaDB vector store, FastAPI backend with SSE streaming, Vite/React frontend, Docker deployment, JSON-file session persistence.
- **Research artifact:** an experiment comparing 8 execution topologies of the same 10-node LangGraph (`all`, `single`, `parallel`, `legal_news_parallel`, `legal_first`, `verify_only`, `planner_based`, `dag`) on latency, cost, lexical quality vs gold answers, LLM-judge quality, and retrieval metrics.
- **Papers:** `main.tex.txt` (root; canonical draft, post-advisor-feedback title *"When Is One Agent Enough?..."*) and `legalai/overleaf_paper.tex` (older draft). `PROPOSAL.md` is an earlier, much broader proposal. `To do.docx` / `Extra to do.txt` / `Gaps.docx` contain advisor feedback and gap analysis.

### Architecture as-built (verified in `graph/workflow.py`)

All topologies are conditional routes through **one** compiled graph: `planner → router → memory → retrieval → {experts by mode} → aggregator → validator → (retry loops) → response`. The modes differ only in which experts run and in what order:

| Mode | Expert path after retrieval |
|---|---|
| `all` | legal → news → general_qa (fully sequential) |
| `single` | router-selected expert(s); **multi-label routes fan out in parallel** |
| `parallel` | legal ∥ news ∥ general_qa |
| `legal_news_parallel` | legal ∥ news |
| `legal_first` | legal → news (no general_qa) |
| `verify_only` | *(none — straight to aggregator; retrieval still runs)* |
| `planner_based` | LLM-planned subset, fan-out if >1 |
| `dag` | (legal ∥ news) → general_qa → aggregator (diamond) |

---

## 2. CRITICAL findings — scientific validity (the paper is currently unsupported)

### 2.1 Three generations of results that disagree, none valid

| Artifact | Provenance | Status |
|---|---|---|
| `benchmark_results.json` | Oldest run: **1 query × 1 repeat × 6 modes** | **Source of every number in both paper drafts** (e.g., 79.07 s vs 46.51 s, "~40% faster", all BLEU/ROUGE values). N=1; no repeats, no CIs, no significance. |
| `metrics_table.tex`, all 31 charts, `analysis_summary.csv`, `significance.csv`, `results.json`, `results/20260713_160543/` | Smoke-scale run analyzed 2026-07-13: **n = 1 per mode** (n = 4 for `all`) | Degenerate. 7/8 modes answered with the 7-word abstention string (see §2.3), so ROUGE-L = 0.0, judge accuracy = 2.0 uniformly, every p-value = 1.0, every CI = ±0.00. `run_meta.json` records `completed_runs: 11` and a **hardcoded** `duration_s: 300.0` (`scripts/package_results.py:64`). |
| `benchmark_runs.jsonl` | Newest real run: **324 rows, 313 successes** — 9 queries (q01–q09) × 8 modes × 5 repeats (q07/q08 have 35, q09 only 3) | **Never analyzed.** `judge_cache.json` holds 2 entries; `analysis_summary.csv` predates it. Also polluted by the compl_ai hijack (§2.2) and mixes smoke-run rows into repeat 0 (§4.6). |

Meanwhile `eval_dataset.json` now contains **30 queries** (10 simple / 10 decomposable / 10 routing) that have never been run. The README and spec claim "N ≥ 30"; no artifact reflects that.

**Consequence:** every quantitative claim in `main.tex.txt` §Results — including the abstract's headline "single agent achieves the strongest reference alignment while running roughly 40% faster" — rests on a single unrepeated observation per mode, and the figures the paper embeds (`chart1…chart29`) were regenerated later from *different, broken* data that contradicts the paper's own table.

### 2.2 The `compl_ai.py` canned-answer hijack corrupts the benchmark

`backend/service.py:337-362`: before running the workflow, `process_query()` checks `compl_ai.get_compl_ai_response(query)`. `compl_ai.py:49-67` matches **bare substrings** — any query containing "obligation" or "evidence" returns a fixed COMPL-AI marketing-style template, **bypassing the entire multi-agent system**, and reports fabricated telemetry: `agent_timings_ms = {"compl_ai": 1.0}`, `workflow_elapsed_ms = 1.0`.

- Verified in data: all 35 successful q08 runs in `benchmark_runs.jsonl` contain the identical canned response with `backend_ms = 1.0` across **all 8 modes**.
- In the current 30-query dataset, **q08, q11, q13** ("transparency **obligations**…", "compliance **obligations** for providers…", "…what **obligations** correspond to each tier") all trigger it. The canned text (a GPAI provider tier table) does not answer any of those three questions.
- It also silently poisons the 4-layer harness: Layer-2's turn-2 query "What extra **obligations** apply…" is hijacked, and its keyword-based success check then passes automatically (`evaluate_harness.py:210`).

**Consequence:** 10% of the dataset produces identical, off-topic, zero-latency answers in every topology — deflating between-mode differences on quality and corrupting latency/token/cost aggregates. For a demo feature this is defensible; inside an experiment it is fatal. **Fix:** gate behind an env flag that is off by default and force-off in `benchmark.py`, or use exact-phrase matching plus an explicit UI entry point.

### 2.3 Aggregator "abstention veto" structurally biases the experiment against multi-agent modes

`agents/aggregator.py:140-143`: after merging expert outputs, if the *concatenated* text contains the abstention sentence anywhere, the **entire** draft is replaced by the abstention string — even when other experts produced good answers. Only the legal agent is instructed to emit that sentence (`config.py:82`), and it abstains whenever retrieved context looks insufficient. So:

- In `all`, `parallel`, `legal_first`, `legal_news_parallel`, `dag`, and `planner_based` modes, **one abstaining legal expert vetoes everything**.
- `single` (when routed away from legal) and `verify_only` (no experts) are immune.

This is not hypothetical: the committed `metrics_table.tex` shows word count = 7, TTR = 1.0, ROUGE = 0.0 for exactly the seven modes that include the legal expert, while `verify_only` produced an 82-word real answer and judge accuracy 5.0. **The one experimental result the committed artifacts actually demonstrate is this bug.** Any single-vs-multi comparison run before fixing it measures abstention propagation, not topology quality. **Fix:** abstain only if *all* (or the route-primary) experts abstain; treat per-expert abstention as a signal, not a veto; and report abstention rate as its own metric.

### 2.4 Gold answers: circular provenance, and misdescribed in the paper

`scripts/build_gold.py`:
- Golds are drafted by **`llama3.1:8b`** — the same model used as the LLM judge (`llm_judge.py:14`). Judge-favors-its-own-text (self-preference) bias is well documented; at minimum gold-writer and judge must differ, ideally with a human pass.
- The drafting prompt explicitly permits falling back to **pre-trained knowledge** when retrieval is thin (`build_gold.py:115`), with a self-groundedness check that **defaults to passing (4/5) when the check errors** (`:57`) and **accepts the draft even when it scores < 4 after two attempts** (loop at `:129-141` has no rejection path).
- All 30 items carry `"gold_status": "draft", "needs_review": true`. The spec (§10) explicitly marks "final legal sign-off on the agent-drafted gold answers" as **human-owned — never done**.
- Both paper drafts call this an "**expert-curated** Gold Standard" (`main.tex.txt:24,69`; `overleaf_paper.tex:24,41`). As written, that description is false and is the most dangerous sentence in the paper.

### 2.5 Retrieval metrics are self-referential (always 1.0)

`gold_doc_ids` are simply **whatever the system's own retriever returned at gold-build time** (`build_gold.py:100-104`). Benchmark-time P@5 / R@5 / MRR then compare the same retriever against its own earlier output — hence the committed table's uniform `1.0000` for every mode. This measures ID stability, not retrieval quality. Additionally, chunk IDs are Chroma insertion indices (`embed.py:127`), so any re-embedding silently invalidates all `gold_doc_ids`. **Fix:** human-annotate relevant chunks per query (or pooled relevance judgments), or drop retrieval claims from the paper.

### 2.6 The "four-layer evaluation" can fabricate results via mocks

`evaluate_harness.py:26-130`: if Ollama is unreachable, the harness silently swaps in `MockChatOllama` / `MockChroma` with hardcoded IRAC answers and hardcoded retrieval hits, then "evaluates" them. The committed `evaluation_assets/four_layer_eval_results.json` reports perfect 1.0 scores with `total_elapsed_seconds: 0.08` — **it ran on mocks.** Metric design is also broken independent of mocks: Layer-3 groundedness can never fall below 0.5 and awards 1.0 to abstentions and to any text containing "document"/"article"/"http" (`:229-236`); Layer-2 turn-1 passes if the answer contains the substring `"10"` (`:209`); Layer-4 routing counts `any()` overlap as correct (`:270`). If any of these numbers are cited anywhere, retract them. **Fix:** make mock mode an explicit `--mock` flag that stamps `"mocked": true` into the output file; redesign the layer metrics.

### 2.7 Statistics: the current pipeline can't support significance claims

- **Unit-of-analysis error:** `analyze_results.py:335-389` pairs on `(query_id, repeat)`, treating repeats as independent samples. Repeats of the same query are highly correlated (they'd be *identical* if seeding worked — see §2.8), so the effective N is the number of queries (9, eventually 30), not queries×repeats (45/150). Aggregate repeats per query (mean/median) before the Wilcoxon, or fit a mixed-effects model with query as a random effect.
- Mode-level CIs (`:311-319`) pool across queries and repeats — same pseudoreplication.
- Judge failures return all-1s (`llm_judge.py:120-126`) and are averaged in silently — a transient judge outage would systematically depress a mode. Exclude-and-log instead.
- No power analysis; with N=9 (current data) even clean stats would be underpowered. The spec's target of 100 queries is the right direction.
- `INPUT_COST_PER_1K`/`OUTPUT_COST_PER_1K` (`analyze_results.py:20-21`) applies GPT-4o-mini-style API prices to a local model. Label as "hypothetical API-equivalent cost" or compute energy/GPU-time instead.

### 2.8 "Deterministic" configuration that isn't — and isn't random either

`config.py:10-12` + `agents/base.py:22-24` force `temperature=0.0, seed=42` for **all** agents (constructor temperatures 0.2–0.5 are dead parameters). Yet in `benchmark_runs.jsonl`, **0 of 63** (query, mode) groups have 5 identical repeat responses — Ollama does not guarantee determinism under GPU batching. So the run inherits the worst of both: it *claims* determinism (fixed seed) while actually sampling **uncontrolled hardware noise**, which is what your repeat-variance and "consistency" metrics then measure. **Fix:** either (a) vary the seed per repeat (`seed = base + repeat`) and describe repeats as sampled diversity, or (b) document the nondeterminism and treat repeats as system-noise replicates. Also set `num_ctx` explicitly (see §3.1) since silent prompt truncation is another variance source.

### 2.9 Paper-specific defects (`main.tex.txt`)

1. **No citations at all.** Chatlaw, LegalGPT, PAKTON, LegalBench-RAG, LexRAG, COMPL-AI, HERA are named with zero `\cite` commands and no bibliography — instant desk-reject in IEEE format. (Also verify the more exotic names from the proposal — HalluGraph, RAGShield, Legal-DC "published March–April 2026" — before citing; I could not corroborate them from the repo.)
2. **Methodology omits every reproducibility fact:** model name/size, quantization, hardware, Ollama version, corpus size, N queries, R repeats, prompt versions. The spec has all of it; the paper has none.
3. **`verify_only` is misdescribed** as "bypassing the retrieval nodes entirely" and "completes in 22.27 s precisely because it skips retrieval" (`:64,117`). In code (`workflow.py:289-290`) retrieval **runs**; the *experts* are skipped. The frontend label ("Retrieval Only (Bypass Experts)") is correct; the paper's causal interpretation of its speed is wrong.
4. **Planner-based and DAG are "deferred to future work"** (`:66,271`) — but both are fully implemented (`workflow.py`) and have 40 runs each in the data. Integrate them; deferring implemented-and-measured topologies undermines credibility.
5. **"32 metrics" claim** vs 22 rows in the paper's own table (the newer pipeline computes ~45). Reconcile.
6. **"SINGLE" is not always single:** for multi-label routes, `route_from_retrieval` fans out 2–3 experts in parallel (`workflow.py:308-321`). Either force one expert in `single` mode or rename/describe it honestly ("routed dynamic subset").
7. Figures/table mismatch: the embedded `figures/chartN.png` set will not reproduce the table (different runs; see §2.1). After the re-run, regenerate both from one pipeline execution.
8. `Estimated Tokens (Words × 1.3)` in the table while the backend now reports true token counts — use the real ones.

---

## 3. CRITICAL findings — application correctness

### 3.1 First news-fetch **permanently destroys the EU AI Act knowledge base**

Chain: `service.process_query` → `fetch_with_progress(..., clear_existing=True)` (`service.py:395-401`) → `embed_articles_from_files(..., clear_existing=True)` (`auto_fetcher.py:355-359`) → **deletes every chunk in the Chroma collection** (`embed.py:173-176`), then re-embeds only the scraped news articles. The AI Act is never re-ingested: `ensure_eu_ai_act_loaded` short-circuits because the store is non-empty (`service.py:123-127`).

Trigger conditions are broad: route == news, or the query contains "recent", "latest", "update", "current", "2024/2025/2026" — or the substring **"new"** (which matches "news", "newest", even "renewable") (`query_analyzer.py:59-79`). One casual question permanently strips the legal corpus for all later legal questions. Benchmarks avoid it only because `benchmark.py` sets `fetch_news: False`.

**Fix:** never `clear_existing=True` from the auto-fetch path; delete only stale `source_type=="news"` chunks (Chroma `where` filter); re-verify AI Act presence by metadata name, not by count > 0.

### 3.2 Context window never configured — silent truncation likely

No `num_ctx` anywhere in project code; `ChatOllama` therefore uses Ollama's default context (2048–4096 tokens depending on version). The legal prompt (5 × ~1000-char chunks + history + IRAC scaffold) approaches it; the **aggregator** prompt in `all`/`parallel` (history + context + up to 3 expert outputs) almost certainly exceeds it, causing Ollama to silently truncate the prompt head. This both degrades multi-expert modes (another anti-multi-agent bias, compounding §2.3) and adds unexplained variance. **Fix:** set `num_ctx=8192` (or measured max) in `agents/base.py`, and add prompt-token logging + truncation of retrieved context via the existing (unused) `utils.truncate_text`.

### 3.3 "Parallel" wall-clock claims depend on Ollama server concurrency

LangGraph does execute fanned-out nodes concurrently (threads), but a single local Ollama serializes generations unless `OLLAMA_NUM_PARALLEL > 1` (and VRAM allows). Nothing records or pins this. The paper's parallel-vs-sequential latency deltas are thus a function of an undocumented server setting. **Fix:** record `OLLAMA_NUM_PARALLEL`, GPU, and VRAM in `run_meta.json`; state it in the paper; ideally run both settings.

---

## 4. MAJOR findings — pipeline, evaluation harness, and benchmark mechanics

### 4.1 Workflow / state (graph correctness)
- **In-place state mutation throughout** (`add_thinking_step` appends to the shared list; agents mutate and return the same dict). Under parallel fan-out, all branches share the same underlying objects; correctness currently depends on GIL atomicity and the dedup reducer (`state.py:29-57`) masking duplicates. LangGraph's contract is "return new partial updates; never mutate." Works today by accident; will break under async or checkpointing.
- The thinking-log dedup key `(step, details)` drops legitimately repeated steps (e.g., a second retrieval pass in a retry loop logs nothing), and `steps = len(thinking_log)` then feeds the "Execution Steps" and "latency-per-step" metrics — undercounting retries.
- Validator loop: on fail it re-enters **planner** (full re-run) with the same fixed seed — the retry regenerates nearly the same answer, except the aggregator sees a `[Note: Please address these issues…]` suffix (`aggregator.py:160-162`). Retries also **overwrite** `agent_tokens` per agent (dict reducer, last-write) while `agent_timings` **accumulate** (`workflow.py:53`) — token/cost undercounts retried runs; timings don't. Inconsistent semantics.
- The **response agent rewrites the answer *after* validation** (`workflow.py:233-244`) — the validator never sees what ships (validate-then-mutate anti-pattern).
- `route` is typed `Literal["legal","news","general",""]` (`state.py:86`) but the router writes `"legal, news"` combined strings (`router.py:51`) — the type lies; route checks are substring-based everywhere.

### 4.2 Agents
- **Validator is cosmetic in the failure path:** it's an LLM self-check parsed by regex where **any parse failure silently means PASS** (`validator.py:27-38`). Worse, it reads sources from `metadata['source']`, but ingestion writes `name`/`url`/`source_type` — **no `source` key exists** (`embed.py:96-122`), so the validator always sees "- Unknown" and its `SOURCE_RELEVANT`/`RETRY_FETCH` verdicts are judgments about nothing.
- Aggregator/GeneralQA duplicate small-talk sets; four near-identical `_format_history`/`_format_context` implementations (legal/news/general_qa/aggregator) should live in `BaseAgent`; the *good* citation-grade formatter (`retrieval.format_context`, with authority + effective dates) is used **only** by gold building — the actual experts see a poorer context format than the gold-writer did (another asymmetry).
- `MemoryAgent` needlessly instantiates a `ChatOllama` it never uses (inherits `BaseAgent`); its `add_exchange` is dead code (service persists directly); its per-instance cache dict only grows.
- Router prompt asks for comma-separated labels, then parsing scans substrings — a chatty model answering "this is generally legal" yields `route="legal, general"`.

### 4.3 Retrieval & ingestion
- **BM25 index is rebuilt from the entire collection on every query** via the private API `self.db._collection.get()` (`retrieval.py:150-165`) — O(corpus) per request; fine at 5 MB, quadratic pain later. Build once, invalidate on ingest.
- Hybrid fusion discards the actual dense similarity scores and uses linear rank weights `(10-rank)/10` with a 999 sentinel (`:208-210`); standard RRF is one line and better-behaved.
- Recency boost triggers on hardcoded strings incl. "2025"/"2026" (`:200`); dates will rot.
- `embed.py` chunking splits on single `\n` and greedily packs ≤1000 **chars** with **no overlap** (`:26-46`) — PDF extraction makes lines, not paragraphs, so chunks are arbitrary; legal texts deserve article-aware chunking.
- **Chunk IDs are `str(count..count+n)`** (`:127`) — collide after any deletion; not content-addressed; the whole `gold_doc_ids` scheme sits on this sand.
- Metadata date heuristics have substring bugs: `"article 5" in chunk` also matches Articles 50–59; `"chapter v"` matches chapters VI–VIII; any chunk containing "high-risk" gets `effective_from=2026-08-02` (`embed.py:113-118`).
- `scraper.py`: no robots.txt handling, browser-impersonating UA, and `ddgs` text results have no `source` field (always "Unknown"). `auto_fetcher._fetch_single_article` has an operator-precedence bug: `title or soup.title.string if soup.title else 'Unknown'` → returns 'Unknown' whenever the page lacks `<title>`, even if the search result had one (`auto_fetcher.py:251`); content is truncated at 10,000 chars mid-sentence.
- **Corpus hygiene:** `articles/` holds 20 files — **8 are 0 bytes**, 10 are < 1 KB, and several are plainly off-topic scrapes ("2026 Lebanon war - Wikipedia", "2026 in the United Kingdom") that get embedded into the *legal* knowledge base and can be retrieved as "sources." `PROPOSAL.md`'s "16 pre-fetched documents (~3,500+ tokens)" doesn't match the folder.

### 4.4 Backend & API
- **Path traversal in production static serving** (`backend/main.py:205-214`): `FRONTEND_DIST / full_path` is served if it `exists()`, with no `resolve()`/containment check — encoded `../` sequences can escape `frontend/dist` when `LEGALAI_SERVE_STATIC=1` (the Docker image sets it!). Fix: `candidate.resolve().is_relative_to(FRONTEND_DIST.resolve())`.
- **Unauthenticated destructive endpoints:** `POST /admin/clear` deletes articles + entire vector store; `DELETE /sessions` wipes history. Fine on localhost, indefensible with the compose file's default `LEGALAI_ALLOWED_ORIGINS=*` and 0.0.0.0 binding. Add a token check at minimum.
- SSE: worker thread runs to completion even after client disconnect (the frontend "Stop" only aborts the HTTP response; the LLM workflow keeps burning GPU); no heartbeat events (long fetch+embed phases can hit proxy idle timeouts); `except Exception:` around `graph.stream` silently swallows the error and **re-runs the whole workflow** via `invoke` on the same (possibly mutated) initial state (`service.py:448-470`).
- `self._graph_lock` serializes all chat requests process-wide — one user at a time per worker (`service.py:448`). Document or remove ("production mode" claims otherwise).
- Route is classified **twice** per request by two different prompts (`service._classify_route` one-word vs in-graph RouterAgent multi-label) — extra LLM call, and they can disagree (the saved session's `route` may not match what the graph did).
- `session_store._write_json` is non-atomic (no tmp+rename) and the lock is per-process (`--workers 2` → last-write-wins races). `datetime.utcnow()` is deprecated on 3.12+. `HTTPException(500, detail=str(exc))` leaks internals.
- `_safe_session_id` is done right (regex allowlist) — no traversal there. Pydantic models and input clamps are solid.

### 4.5 Evaluation harness code quality
- Custom BLEU/ROUGE implementations (`evaluate_workflows.py:39-167`) are close-but-nonstandard (punctuation-stripping tokenizer, β=1 ROUGE-L). Numbers won't be comparable to literature — use `sacrebleu` + `rouge-score` and state versions.
- **`generate_latex_table` backend-latency bug:** comment says "Scaled inside row" but no scaling exists (`evaluate_workflows.py:502`) — the table prints **milliseconds labeled "(s)"** (39597.49 "s" in the committed file). Chart 2 scales correctly; the table doesn't.
- "Total Tokens Used" (table + chart 7) is completion tokens only.
- Temporal-reference regex omits August–December (`analyze_results.py:244`); bullet-point regex `[-*+•\d+\.]` is a char class that counts any digit-initial line as a bullet (`:245`).
- Radar chart normalizes Speed/Steps as max/x (can exceed the axis, clipped at 1.1) and mixes them with raw 0-1 metrics — decorative, not principled.
- `metrics_to_agg` contains duplicates ("bleu_1", "bleu_4" twice) → duplicated columns in the CSVs.
- `net_overhead` is clamped at ≥ 0 (`:271`) — negative values would signal timing bugs; don't hide them.

### 4.6 Benchmark mechanics
- `run_full.ps1`/`.sh` run `--smoke` (truncates `benchmark_runs.jsonl` with mode "w") then `--resume` (appends, skipping completed combos) — so **q01/repeat-0 rows in the final dataset are the smoke run's rows**, including cold-start latency for the first mode. Separate smoke output from experiment output, and add an explicit warm-up request per model before timing.
- Fixed execution order (`itertools.product(dataset, MODES, repeats)`) — no randomization/interleaving; ordering effects (model load/unload between chat and embed models, thermal state) systematically favor later runs.
- The run captured only 9 of the intended queries with unbalanced coverage (q09: 3 rows, all in one mode) and 11 silent failures — `analyze_results` drops failures without reporting *which* mode/query failed; unbalanced pairs silently shrink the Wilcoxon.
- Client `timeout=600` with server route-classification and session writes inside `elapsed_s` — fine, but note `backend_ms` excludes the service-level route LLM call, so "network overhead" (chart 3) actually contains an LLM call.
- Docker `benchmark` service sets `LEGALAI_DETERMINISTIC`/`JUDGE_MODEL` on the **benchmark container** where nothing consumes them (the *server* container needed them), and writes `benchmark_runs.jsonl` to an **unmounted** `/app` path — results are lost when the container exits (`docker-compose.yml:30-45`).

---

## 5. Frontend (`frontend/src/App.jsx`, 1136 lines)

Generally the healthiest part of the codebase: fetch-stream SSE parsing with clean fallback to `/chat`, AbortController wiring, ReactMarkdown **without** `rehype-raw` (model HTML is not rendered → no XSS), export/import, session manager, quick prompts, `aria-live` on messages.

Issues:
- **`VALID_MODES` omits `planner_based` and `dag`** (`App.jsx:32`) — the two newest topologies can't be selected; `sanitizeExpertMode` coerces them to "all". The mode dropdown (`:1109-1114`) likewise lacks them.
- The init `useEffect` depends on `[sourceLimit]` and its cleanup aborts the in-flight request (`:184-190`) — adjusting "Source preview limit" while a response streams **kills the stream** and re-fires all five status loads per keystroke.
- `frontend/dist` is a stale build (2026-05-20; bundle predates the newest modes) yet ships in the repo and in production mode. Rebuild on release; don't version `dist/`.
- Minor: Enter-to-send doesn't work (textarea + submit only); `App.jsx` is a single 1,100-line component (split panels into components); duplicate final-payload handling between stream/non-stream paths.

---

## 6. Deployment & reproducibility

- **`requirements.txt` is completely unpinned** (only `chromadb>=0.5.0`, `langgraph>=0.2.0` floors). For a research artifact this is the #1 reproducibility killer — the langchain ecosystem breaks monthly. Ship `pip freeze`-style pins (or a `uv`/`pip-tools` lock) + record versions in `run_meta.json`.
- Dockerfile: runs as **root**, `COPY . .` (mitigated by `.dockerignore`, which is decent), sets `LEGALAI_SERVE_STATIC=1` (→ exposes §4.4 traversal), no `HEALTHCHECK` in-image (compose has one). Multi-stage frontend build is good.
- docker-compose: defaults `LEGALAI_ALLOWED_ORIGINS: "*"`; `host.docker.internal` doesn't resolve on Linux without `extra_hosts: ["host.docker.internal:host-gateway"]`; no optional `ollama` service (the stack isn't actually self-contained); benchmark-service issues per §4.6.
- **Two virtualenvs** (`.venv` at root and `legalai/.venv`, 814 MB) — pick one.
- `app.py` launcher is well done (Windows npm shim resolution, sibling-process monitoring, graceful termination). Minor: `--reload` default in dev with `--workers` silently ignored under reload is handled correctly.

---

## 7. Repository hygiene & provenance

There is **no version control**: no git repo at root or in `legalai/`. For a thesis project this is the single most dangerous operational fact — the three-generations-of-results confusion in §2.1 is exactly what git history prevents. Also missing: `.gitignore`, `LICENSE`, `CITATION.cff`, any CI.

Things currently mixed into the tree that don't belong in source control (once git exists):
- `chroma_storage/` (5.6 MB binary DB), `legalai/.venv/` (814 MB), `.pytest_cache/`, `frontend/dist/`, `frontend/node_modules/`
- 330 benchmark/chat session JSONs in `sessions/`
- `results/20260713_160543/` — byte-identical duplicate of root-level CSVs + all 31 charts (~4 MB duplicated)
- `benchmark_runs.jsonl` (560 KB), `judge_cache.json`, `benchmark_results.json` — keep, but under `results/` with provenance
- `Legal_AI_Benchmark_Report.docx` (467 KB) at `legalai/` root
- `_chk.tex` / `_chk.log` at root — leftovers from a failed local `pdflatex` probe (missing `IEEEtran.cls`); delete (and compile the paper on Overleaf or install `texlive-publishers`)
- Root-level loose docs (`Extra to do.txt`, `To do.docx`, `Gaps.docx`, `main.tex.txt`) — move to `docs/` and `paper/`; rename `main.tex.txt` → `paper/main.tex`

`.env` is byte-identical to `.env.example` and contains no secrets (good).

Suggested layout:

```
Legal AI/
├─ .git/  .gitignore  LICENSE  CITATION.cff  README.md
├─ paper/         main.tex, refs.bib, figures/ (generated), advisor-feedback/ (To do.docx, Gaps.docx, Extra to do.txt)
├─ docs/          PROPOSAL.md, EXPERIMENT_BUILD_SPEC.md, research_gaps_filled.txt
├─ legalai/       (code only: agents/ backend/ graph/ frontend/ scripts/ *.py)
├─ data/          eval_dataset.json, articles/ (cleaned)
└─ results/       <timestamped runs incl. benchmark_runs.jsonl + run_meta.json with REAL env capture>
```

---

## 8. Documentation drift

- **PROPOSAL.md promises far more than exists** (fine for a proposal, but nothing marks the delta): user study with 30 legal professionals, eye-tracking, fairness/demographic-parity audits, third-party security audit, Legal-DC / Legal RAG Bench / LawBench evaluation, "Responsible AI Council," ISO 42001/23894 alignment, F1 ≥ 0.85 targets. None are implemented. Add a status table to the proposal or an explicit scope-narrowing note — the current paper's narrower scope is *better*, but the trail should say so.
- README is largely accurate (a rarity) but: claims "expert" gold workflow via `build_gold.py` without the review caveat; "Multi-Agent vs Single-Agent Experiment" section says N ≥ 30 (never run); the stable-ID scheme it documents is the fragile count-based one (§4.3).
- `research_gaps_filled.txt` says the project "**proves**" a ~41% latency reduction — from N=1. Soften until the re-run.
- Advisor feedback (`To do.docx`, `Extra to do.txt`) vs `main.tex.txt`: title/framing/definitions/interpretation-sections are **done**. Still open: integrate planner+DAG results (data exists), real literature review with citations, topology diagrams in methodology, broader metrics (task success, consistency, scalability), literature check for prior "single-agent beats multi-agent" findings, HPC/bigger-model experiments.

---

## 9. Local dev-environment issue (outside the repo, worth fixing today)

Your Claude Code environment carries model-override variables pointing at nonexistent models — `ANTHROPIC_MODEL=deepseek-v4-pro[1m]`, `ANTHROPIC_DEFAULT_SONNET_MODEL=deepseek-v4-pro[1m]`, `ANTHROPIC_DEFAULT_OPUS_MODEL=deepseek-v4-pro[1m]`, `ANTHROPIC_DEFAULT_HAIKU_MODEL=deepseek-v4-flash`, `CLAUDE_CODE_SUBAGENT_MODEL=deepseek-v4-flash` — while `ANTHROPIC_BASE_URL` points at the real Anthropic API. Result: **every subagent/background agent fails to launch** (this audit had to run single-threaded). These look like leftovers from a proxy/router experiment; unset them (System Properties → Environment Variables, or wherever they're exported) unless you're actively routing through a proxy that understands those names.

---

## 10. Remediation roadmap

### P0 — before any number is shown to an advisor/reviewer (≈ 2–4 days of work + 1 re-run)
1. **Disable the compl_ai hijack in the experiment path** (env-gate, off by default; force-off in `benchmark.py`). *(§2.2)*
2. **Fix the aggregator abstention veto** — abstain only when all experts abstain; log abstention as a metric. *(§2.3)*
3. **Set `num_ctx` explicitly** (≥ 8192) in `agents/base.py`; add prompt-length logging. *(§3.2)*
4. **Fix `generate_latex_table` backend-ms scaling.** *(§4.5)*
5. **Decide the determinism story** (per-repeat seeds recommended) and document Ollama nondeterminism. *(§2.8)*
6. **Fix analysis statistics:** aggregate repeats per query before paired tests; exclude judge-failure rows; keep Holm + Cliff's delta; add per-query-type (H2) tests. *(§2.7)*
7. **Re-run everything cleanly:** fresh `benchmark_runs.jsonl`, 30 queries × 8 modes × 5 repeats, warm-up excluded, smoke output kept separate → `analyze_results.py` → `evaluate_workflows.py` → `package_results.py` (with real duration/env capture, not the hardcoded 300.0/30). One provenance chain, one results folder.
8. **Gold answers:** human-review pass over all 30 (spec §10 says this is yours to do); regenerate with a model ≠ judge (or at least judge with a different model than the gold-writer); change every "expert-curated" claim to match reality.
9. **Retrieval metrics:** annotate true relevant chunks or cut retrieval claims. *(§2.5)*
10. **Update the paper** from the new run only: include planner_based + dag columns, fix the verify_only description, add citations + bibliography, add a reproducibility paragraph (model, quantization, hardware, Ollama version, `OLLAMA_NUM_PARALLEL`, seeds, N/R), regenerate figures and table from the same run.

### P1 — correctness and safety (application)
11. Fix the vector-store wipe path (targeted news-chunk deletion; AI-Act presence check by name). *(§3.1)*
12. Validator: fix `source`→`name` metadata read; treat unparseable validator output as FAIL-with-log, not PASS; validate the *final* response or drop the post-validation rewrite. *(§4.2, §4.1)*
13. Static-file traversal fix + token on `/admin/clear` and `DELETE /sessions`. *(§4.4)*
14. Stop mutating LangGraph state in place; return partial updates; align `agent_tokens` (sum) with `agent_timings`. *(§4.1)*
15. SSE: cancel workflow on client disconnect (or document), add heartbeats; atomic session writes. *(§4.4)*
16. Keyword bugs: `"new"` substring trigger, `"article 5"`/`"chapter v"` metadata misclassification, router substring parsing → word-boundary regex or exact-set matching. *(§3.1, §4.3)*
17. Chunking with overlap (and ideally article-aware splitting); content-hash chunk IDs; persistent BM25 index; RRF fusion. *(§4.3)*
18. Purge empty/off-topic articles; add a min-length + domain filter at save time (`utils.save_online_articles`).

### P2 — hygiene and polish
19. `git init` + `.gitignore` (venvs, chroma, dist, sessions, caches, results) + `LICENSE` + `CITATION.cff`; commit in logical units; tag the state used for each paper draft.
20. Pin `requirements.txt`; delete one of the two venvs; record package versions in `run_meta.json`.
21. Repo restructure per §7; delete `_chk.*`, duplicate `results/20260713_160543`, stale `frontend/dist`.
22. Frontend: add `planner_based`/`dag` to `VALID_MODES` + dropdown; decouple the abort-on-`sourceLimit` effect; Enter-to-send; split the monolith component.
23. Docker: non-root user, drop default CORS `*`, `extra_hosts` for Linux, mount the benchmark output path, optional ollama service.
24. Replace custom BLEU/ROUGE with `sacrebleu`/`rouge-score`; fix month/bullet regexes; add `pytest` coverage for topology wiring (assert node execution order per mode with a mocked LLM — cheap and catches routing regressions), for the aggregator abstention logic, and for the API endpoints; add `ruff` + pre-commit.
25. Logging module instead of `print` throughout; deduplicate `_format_history`/`_format_context`/small-talk sets into `BaseAgent`.

### Research extensions (advisor-aligned, post-P0)
- Scale N toward the spec's 100 queries; add a second corpus/domain to test the generalization claim the new title makes.
- Repeat on ≥ 2 model sizes (e.g., qwen2.5:3b vs 7b vs 14b) — the advisor's HPC point; the single-vs-multi gap plausibly narrows or widens with capacity, which would be a genuinely novel result.
- Validate the LLM judge against human ratings using the already-built (never used) `llm_judge.py --validate` path; report agreement (MAE/Spearman) in the paper.
- Add task-success/abstention-rate/consistency metrics per the advisor list; consider cost as tokens×latency rather than fictional API dollars.
- Literature to position against (verify before citing): AutoGen (Wu et al., 2023), MetaGPT (Hong et al., 2023), CAMEL (Li et al., 2023), "More Agents Is All You Need" (2024), "Are More LLM Calls All You Need?" (2024), "Why Do Multi-Agent LLM Systems Fail?" / MAST taxonomy (2025), LLM-as-judge bias literature (Zheng et al., 2023; self-preference bias 2024), LegalBench (Guha et al., 2023), Chatlaw (Cui et al., 2023), LegalBench-RAG (2024). The proposal's HalluGraph / RAGShield / Legal-DC citations could not be corroborated — verify they exist before citing.

---

## 11. What's already good (keep it)

- Clear modular decomposition (agents / graph / backend / frontend / scripts) and a genuinely readable codebase.
- `EXPERIMENT_BUILD_SPEC.md` is an excellent experiment charter — hypotheses, acceptance criteria, and explicitly human-owned steps. The failure was execution drift, not design.
- The statistics *machinery* (Wilcoxon + Holm + Cliff's delta + CIs) is the right toolkit; it just needs the right units and real data.
- Resume-able benchmark with JSONL append + flush; judge caching keyed on content; schema validation of the dataset; unit tests for the metric math (all 5 pass).
- Backend API design (pydantic models, health/readiness/runtime endpoints, SSE with graceful fallback) and the launcher's Windows-aware process management.
- Frontend streaming implementation and safe markdown rendering.
- `.env.example` hygiene; no secrets anywhere in the tree; session-ID sanitization done correctly.

---

## Appendix A — file-by-file inventory

| File | Verdict | Key notes (§refs) |
|---|---|---|
| `legalai/config.py` | ⚠ | Forced temp=0/seed=42 (§2.8); prompts decent; MAX_ITERATIONS shared by two loop types |
| `legalai/state.py` | ⚠ | Reducers OK; `route` type wrong (§4.1); unused `validation_issues` field |
| `legalai/graph/workflow.py` | ⚠ | All 8 topologies wired as claimed; in-place mutation (§4.1); step-log dedup undercounts |
| `legalai/agents/base.py` | ⚠ | Deterministic override kills per-agent temps; no `num_ctx` (§3.2); no timeout/retry on LLM calls |
| `legalai/agents/planner.py` | ✓/⚠ | Verbose state init; JSON plan parsing with sane fallback |
| `legalai/agents/router.py` | ⚠ | Substring parsing (§4.2) |
| `legalai/agents/retrieval.py` | ⚠ | Per-query BM25 rebuild, private API, rank fusion, duplicate imports (§4.3) |
| `legalai/agents/legal.py` / `news.py` / `general_qa.py` | ⚠ | Copy-pasted helpers; no context truncation; legal abstention string (§2.3) |
| `legalai/agents/aggregator.py` | ✗ | Abstention veto (§2.3); duplicated small-talk set |
| `legalai/agents/validator.py` | ✗ | Parse-fail = PASS; reads nonexistent `source` metadata (§4.2) |
| `legalai/agents/response.py` | ⚠ | Rewrites after validation (§4.1) |
| `legalai/agents/memory_agent.py` | ⚠ | Unused LLM; dead `add_exchange`; growing cache |
| `legalai/compl_ai.py` | ✗ | Keyword hijack — corrupts benchmark + demo (§2.2) |
| `legalai/backend/service.py` | ✗ | Hijack call-site; store-wipe on fetch (§3.1); silent stream fallback; global graph lock; double routing |
| `legalai/backend/main.py` | ⚠/✗ | Static traversal in prod (§4.4); unauth destructive endpoints; SSE no-cancel |
| `legalai/backend/session_store.py` | ✓/⚠ | Safe IDs ✓; non-atomic writes; per-process lock |
| `legalai/backend/models.py` | ✓ | Clean |
| `legalai/app.py` | ✓ | Solid launcher |
| `legalai/utils.py` | ⚠ | Saves empty articles; unused truncate helper; odd `JS`/`OS` aliases |
| `legalai/embed.py` | ✗ | Count IDs; no overlap; date-substring bugs; clear-all path (§3.1, §4.3) |
| `legalai/scraper.py` | ⚠ | No robots.txt; UA spoof; `source` never populated |
| `legalai/auto_fetcher.py` | ✗ | `clear_existing=True` default (§3.1); title precedence bug |
| `legalai/query_analyzer.py` | ⚠ | `"new"` substring trigger; new LLM per helper call |
| `legalai/benchmark.py` | ⚠ | Goes through HTTP (hijack applies); fixed order; no warmup; smoke/resume mixing (§4.6) |
| `legalai/analyze_results.py` | ⚠ | Right tests, wrong units (§2.7); judge-failure poisoning; regex bugs; fictional cost |
| `legalai/evaluate_workflows.py` | ⚠ | Custom metrics; **table ms bug** (§4.5); radar normalization |
| `legalai/llm_judge.py` | ⚠ | Judge = gold model (§2.4); all-1s fallback; single judge |
| `legalai/evaluate_harness.py` | ✗ | Silent mock mode; broken layer metrics (§2.6) |
| `legalai/test_experiment.py` | ✓ | 5 passing math tests; no pipeline coverage |
| `legalai/scripts/build_gold.py` | ✗ | Circular golds; no rejection path; same-model judge (§2.4, §2.5) |
| `legalai/scripts/package_results.py` | ⚠ | Hardcoded `duration_s=300.0`, `total_queries=30` (§2.1) |
| `legalai/scripts/run_full.ps1` / `.sh` | ⚠ | Smoke rows leak into results (§4.6); parity OK |
| `legalai/eval_dataset.json` | ⚠ | 30 balanced queries ✓; all golds draft/needs-review; 3 hijacked queries (§2.2) |
| `legalai/benchmark_runs.jsonl` | ⚠ | 324 rows / 9 queries; never analyzed (§2.1) |
| `legalai/benchmark_results.json` | ⚠ | Legacy N=1 run — the paper's actual data source (§2.1) |
| `legalai/analysis_summary.csv`, `significance.csv`, `by_query_type.csv`, `results.json`, `metrics_table.tex`, `evaluation_assets/*` (31 charts), `results/20260713_160543/*` | ✗ | Degenerate smoke-run artifacts; duplicate copies (§2.1, §7) |
| `legalai/evaluation_assets/four_layer_eval_results.json` | ✗ | Produced by mocks in 0.08 s (§2.6) |
| `legalai/judge_cache.json` | ⚠ | 2 entries — proof analysis never ran on real data |
| `legalai/articles/*` | ⚠ | 8/20 empty, off-topic content embedded (§4.3) |
| `legalai/sessions/*` (330) | 🗑 | Run artifacts; don't version |
| `legalai/chroma_storage/` | 🗑 | 5.6 MB binary DB; regenerable |
| `legalai/frontend/src/App.jsx` | ✓/⚠ | Good streaming/XSS-safety; missing 2 modes; sourceLimit-abort bug (§5) |
| `legalai/frontend/*` (config, dist) | ⚠ | Sensible config; `dist/` stale (§5) |
| `legalai/requirements.txt` | ✗ | Unpinned (§6) |
| `legalai/Dockerfile`, `docker-compose.yml`, `.dockerignore` | ⚠ | Root user; CORS `*` default; Linux host-gateway; benchmark volume bug (§6, §4.6) |
| `legalai/.env`, `.env.example` | ✓ | Identical, no secrets |
| `legalai/README.md` | ✓/⚠ | Mostly accurate; N≥30 claim unmet |
| `legalai/EXPERIMENT_BUILD_SPEC.md` | ✓ | Excellent; §10 human tasks not done |
| `main.tex.txt` (root) | ✗→fixable | N=1 numbers, no citations, verify_only misdescribed, "expert-curated" claim (§2.9) |
| `legalai/overleaf_paper.tex` | 🗑 | Superseded draft — archive |
| `PROPOSAL.md` | ⚠ | Promises >> reality; add scope note (§8) |
| `To do.docx`, `Gaps.docx`, `Extra to do.txt` | ✓ | Advisor feedback captured; move to `paper/advisor-feedback/` |
| `research_gaps_filled.txt` | ⚠ | "Proves" from N=1 (§8) |
| `Legal_AI_Benchmark_Report.docx` | ? | 467 KB generated report — regenerate after the clean run |
| `_chk.tex`, `_chk.log` | 🗑 | Failed pdflatex probe (missing IEEEtran.cls) — delete |
| `.venv/` + `legalai/.venv/` | ⚠ | Two venvs (814 MB); keep one |

Legend: ✓ sound · ⚠ issues · ✗ critical defect · 🗑 remove/relocate · ? review

---

## Appendix B — the five headline evidence items, verbatim from the repo

1. `benchmark_runs.jsonl` q08 (all 35 rows): `"timings": {"compl_ai": 1.0}, "backend_ms": 1.0` — canned template identical across all 8 modes.
2. `analysis_summary.csv`: `elapsed_s_n` = 1 for 7 modes, 4 for `all`; `rouge_l_mean` = 0.0 except `verify_only` = 0.318; `word_count_mean` = 7.0 (= the abstention sentence).
3. `significance.csv`: every `*_p` and `*_p_holm` = 1.0.
4. `metrics_table.tex` line 13: `Backend Graph Latency (s) … 39597.49` (milliseconds mis-labeled as seconds).
5. `eval_dataset.json`: all 30 items `"gold_status": "draft", "needs_review": true, "gold_model": "llama3.1:8b"` — vs the paper's "expert-curated gold standard."
