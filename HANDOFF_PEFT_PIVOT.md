# Legal AI — PEFT Pivot: complete handoff (A to Z)

**Written 2026-08-04.** Self-contained. Paste this into a new chat, or just say:
*"Read `HANDOFF_PEFT_PIVOT.md` in the project root and continue."*

---

## 0. TL;DR — state right now

- **Implementation: complete.** 101 tests pass. Paper rewritten.
- **Adapters: trained and validated.** Legal and news work well. `general_qa`
  runs **unadapted** by user decision (its adapter degenerates — see §6).
- **Not yet run: the 540-run benchmark.** Everything upstream of it is verified.
- **Nothing is running.** GPU free.

**Immediate next step:** re-time one full run (§9 step 1), expect ~8 min/run.

---

## 1. What the project is

Charbel's thesis experiment. A LangGraph multi-agent RAG system answering EU AI
Act compliance questions, feeding the paper at `main.tex`.

**Old question (previous paper, done):** does multi-agent coordination beat a
single well-configured agent? Answer: no. But every "specialist" there was the
*same* model behind a different prompt.

**New question (this pivot):** given three *genuinely different*, separately
fine-tuned expert models, **which coordination topology combines them best?**

Directed by the advisor. Spec: `PEFT_Pivot_Implementation_Plan.md` (root).

- `SINGLE` and 4 other topologies remain implemented but are **out of scope**.
- Generation is **fully local**. DeepSeek is **judge-only** — that is what
  retires the self-preference-bias caveat of the previous paper.

---

## 2. Experimental design

**30 queries × 3 topologies × 3 repeats × 2 arms = 540 runs**

- **Topologies:** `all` (sequential chain), `parallel` (concurrent), `dag`
  (converging). Structurally distinct; nothing else is benchmarked.
- **Arms:** `peft` (adapters loaded) and `base` (identical models, no adapters).
  The base arm is the control that makes RQ2 answerable.
- Arms run as **two consecutive passes** (six models won't fit in 8 GB), both
  appended to one `benchmark_runs.jsonl` distinguished by an `arm` column.

**RQs:** RQ1 which topology wins and at what cost; RQ2 did specialisation help;
RQ3 operational cost per topology.

**Stats:** experimental unit is the **query** (repeats averaged first). Two
families, Holm-corrected separately: pairwise topologies *within* an arm, and
peft-vs-base *within* a topology. Positive median diff / Cliff's delta = the
**first-named** side scored higher.

---

## 3. Hardware and models

RTX 4070 Laptop, **8 GB VRAM**, Windows 11. Interpreter:
`legalai\.venv\Scripts\python.exe` — **never bare `python`** (see §7 bug 8).

| role | base model | 4-bit VRAM | KV/token | adapter |
|---|---|---|---|---|
| legal | `unsloth/Llama-3.2-3B-Instruct` | 2206 MiB | 112 KiB | **yes** |
| news | `Qwen/Qwen2.5-3B-Instruct` | 1992 MiB | 36 KiB | **yes** |
| general_qa | `ibm-granite/granite-3.1-2b-instruct` | 1781 MiB | 80 KiB | **NO** (§6) |

All three resident: **5979 MiB**, 1089 MiB free, 912 MiB KV at 4096 tokens.
**Measured concurrency speedup: 1.07×** (not ~3×) — three models on one GPU
serialise regardless of how the graph dispatches.

Coordination nodes (planner, router, memory, aggregator, validator, response,
QueryAnalyzer) share the **general expert's base weights, adapter disabled**.
A fourth model doesn't fit, and keeping specialisation confined to experts is
what makes topology the independent variable.

**Model substitutions forced by reality (needs advisor sign-off):**
- **Ministral 3B has no open weights.** Mistral open-weighted only the 8B.
  `ministral/Ministral-3b-instruct` was created 2024-03-14, seven months *before*
  Mistral announced Ministral — unrelated model reusing the name.
- **Phi-3.5-mini replaced by Granite 3.1 2B.** Phi has *no grouped-query
  attention* (32 KV heads) → 384 KiB/token. All three fitted in VRAM with 252 MiB
  spare, but their combined KV cache needed ~2.1 GB. **GQA, not parameter count,
  is the binding constraint.** A residency check is not a feasibility test.
- `meta-llama/Llama-3.2-3B-Instruct` is `gated: manual`; the Unsloth mirror is
  the same weights, ungated.

---

## 4. Data

**Corpus:** 644 chunks, 599,775 chars, from **CELEX:32024R1689** (Regulation (EU)
2024/1689, Official Journal text) via the **EU Publications Office Cellar**
endpoint. Rebuild: `python embed.py --replace`.

> The old Parliament URL returns HTTP 202/empty and `eur-lex.europa.eu` serves
> non-browser clients an AWS WAF challenge. Cellar needs an explicit
> `Accept-Language: eng`. Note this corpus is **not** chunk-comparable to the
> previous study's 351 chunks — disclosed in Threats to Validity.

**Fine-tuning sets** (`finetune/data/`, ~1425 examples each):

| domain | source | median tokens |
|---|---|---|
| legal | LegalBench `nguha/legalbench`, 102 tasks round-robin | 299 |
| news | NewsQA `lucadiliello/newsqa` | 642 |
| general | Dolly-15k, biased 70% to context-bearing rows | 301 |

**Two data defects, both fixed, both invisible in the loss curve:**
1. **Raw LegalBench answers average 3.8 chars** (105/112 tasks ≤10 chars, median
   3: "Yes"/"No"/"UCC"). Training on them teaches one-word answers, which the
   judge scores as incomplete — presenting as *"specialisation hurt quality"*.
   Fixed: `--legal-format irac` renders targets into the deployed response
   structure using **only real dataset content**. Citations are **omitted, never
   invented**.
2. **Targets sit at the END of each example**, so anything over
   `--max-seq-length` is right-truncated into an input with **no answer**. At 512
   tokens this hit **83% of NewsQA**; that adapter learned nothing (loss flat
   1.920→1.956). Fixed: 2800-char input cap, NewsQA articles **windowed around
   the answer**, and `prepare_datasets.py` now tokenises with each model's own
   tokenizer and refuses above 10% over-limit. Final: 0.0% / 0.0% / 2.5%.

---

## 5. Training

**Final hyperparameters (identical across all three — this matters):**

```
epochs 1 | lr 5e-5 | max_seq_length 1024 | LoRA r=16 alpha=32 dropout=0.05
effective batch 8 (1 × 8 grad-accum) | 4-bit NF4 + double quant
paged AdamW 8-bit | cosine schedule, 3% warmup | ~1425 examples
```

**The consistency rule (user was emphatic):** every adapter must get the *same*
budget. If one expert gets more epochs/tokens/examples, a topology difference is
partly *"which position holds the best-trained expert"* — a self-inflicted
confound. **Changing any parameter means retraining all three.**
`train_qlora.py` resumes by default but **aborts** on a budget mismatch.

**Results (~91 min total):**

| adapter | eval curve (25→100%) | train loss | trainable | time |
|---|---|---|---|---|
| legal | 1.310 → 1.203 → 1.165 → **1.160** | 1.409 | 24.3 M (0.75%) | 18.6 min |
| news | 1.934 → 1.909 → 1.904 → **1.904** | 1.949 | 29.9 M (0.96%) | 28.2 min |
| general_qa | 3.457 → 2.044 → 1.739 → **1.721** | 3.669 | 28.2 M (1.10%) | 25.3 min |

Base (unadapted) held-out loss for reference: legal 2.958, news 2.994,
general_qa 3.449.

**Loss is NOT a quality signal here.** An earlier adapter set cut held-out loss
46–72% and generated word-salad. Judge quality against the **base model** and at
**real prompt length** only.

**Validation (real agent prompts, `--realistic-length` default):**

| role | probe tokens | divergence | words tuned/base | citations |
|---|---|---|---|---|
| legal | 1674 | 98.3% | **165 vs 7** | 1.0 vs 0.0 |
| news | 243 | 93.3% | 214 vs 72 | 0/0 |
| general_qa | 262 | 89.7% | 120 vs 59 | 0/0 |

**The legal base model's "7 words" is the abstention sentence.** At a real prompt
the unadapted model *declines*; the adapter answers citing Article 43 / Article 8.
Direct probe: tuned legal gave **687 words of correct IRAC citing Article 6(3)**;
base gave 165 words citing **Article 2(1)(b) — a fabricated citation with a
fabricated quote**. That is a behavioural change in abstention, which this paper
treats as a first-class outcome. The benchmark, not two probes, settles it.

---

## 6. Why `general_qa` runs unadapted (user decision, 2026-08-04)

At its real 1368-token serving prompt the adapted Granite emitted invented
non-words ("considerallation", "frontrunnability", "obsolecies") and ran to the
full 1024-token cap (156 s). **Identical base weights answered correctly in 9 s**,
citing Annex III and EU database registration.

Cause: its Dolly training data is generic instructions (~300-token median) while
the node is served ~1370 tokens of statutory text — far out of distribution.
Legal and news are unaffected because their training data resembles what they're
served. Granite is also smallest and moved furthest from base.

**Three fixes tried, then accepted as a limit — do not retry blindly:**
- Bias Dolly to context-bearing rows (median 175→301): helped a lot
  (213 s→78 s, stopped capping) but output stayed degenerate.
- Distinct pad token (Granite's `<|end_of_text|>` id 0 is eos+bos+unk):
  **no effect** — eval curves near-identical before/after. Hypothesis refuted.
- Gentler hyperparameters: already in use.

**Implementation:** `config.LOCAL_UNADAPTED_ROLES` (env
`LEGALAI_UNADAPTED_ROLES`, default `general_qa`). Set to `""` to adapt all three.

**Consequence:** the PEFT arm is 2 adapted + 1 unadapted. RQ2 measures
specialising the legal and news roles. Written into `main.tex` methodology.

---

## 7. Every bug found and fixed (do not reintroduce)

| # | bug | why it mattered |
|---|---|---|
| 1 | **`LocalChatModel.__init__` passed the *resolved* role** to `get_loaded_model` | **THE big one.** `resolve_role("aggregator")→"general_qa"`, so every coordination node ran with the general expert's LoRA. Cause of all the word-salad. Fix: pass the **requested** role. After fix: aggregator 159 s→42 s, output correct. An earlier test passed because it called `get_loaded_model` directly, bypassing the constructor. |
| 2 | **No repetition penalty** | `transformers` defaults to 1.0 (none); Ollama, used by every pre-pivot run, applies 1.1. Small models looped ("Article Article Article…" ×80). Fixed: `LOCAL_REPETITION_PENALTY=1.1` to match Ollama. Also records `hit_token_cap`. |
| 3 | **`experts_run` counted `router`** | `agents/router.py` writes into shared `agent_outputs`; aggregator counted every key → denominator of `expert_abstention_rate` (a **reported paper metric**) 25% too large. Pre-existing, not from the pivot. Fixed via `EXPERT_KEYS`. |
| 4 | **`.env` had `GENERATION_PROVIDER=deepseek`** | Script printed `local_peft` while the server loaded DeepSeek. The arm cross-check aborted the run rather than recording hosted answers as local. Fixed + script now reads `.env` and exports the resolved value. |
| 5 | **`run_experiment.ps1` used bare `python`** | System interpreter has torch but **not** peft/accelerate/bitsandbytes → no model ever loaded, with a message saying to install something already installed (in the venv). All 10 invocations now use the venv + a preflight import check. |
| 6 | **Server-start timeout 25 attempts** | Server became healthy just after benchmark.py gave up; log filled with identical tracebacks. Now 300 s, sparse logging, detects subprocess death. |
| 7 | **Request timeout 600 s** | Validator retries are the *common* case (508/630 pre-pivot rows had 20+ steps vs 14–17 for one pass). 10 min would record ordinary behaviour as failures and bias toward whichever topology retries least. Now 3600 s. |
| 8 | **Corpus was 44 stale news chunks** | A news fetch runs `clear_existing=True` and replaces the Act. Every legal query would abstain. |
| 9 | **Act download URL dead** | See §4. `embed.py` now tries multiple sources and **raises** rather than embedding an empty doc. |
| 10 | **`validate_adapters.py` gave two wrong verdicts** | (a) 50-token probes passed broken adapters; (b) an improvised prompt scaffold failed *working* ones (adapters are format-sensitive). Now uses the **real agent templates** and prints word ratio + citations next to the verdict. |
| 11 | **Probe reused one `session_id`** | Memory agent replayed the previous answer, so a retrained adapter reproduced the old one verbatim and looked unchanged. `benchmark.py` was always safe (unique id per arm/mode/query/repeat). |
| 12 | **`unload_all()` didn't free with a live reference** | `validate_adapters` loads 6 models sequentially → would OOM. |
| 13 | **`snapshot_run.py compare()` keyed on `mode` only** | 6 rows (3 modes × 2 arms) collapsed to 3; one arm silently overwrote the other. |
| 14 | **Preflight leaked the server subprocess** | An aborted check left uvicorn listening; the next arm would reuse it and measure the wrong models. |
| 15 | **git dependency** | Removed per user request. `snapshot_run.py` now uses a source-file hash instead of `git rev-parse`. `.git` **moved** (not deleted) to `Desktop\LegalAI_git_history_backup_20260804`. Project is git-free. |

**Refuted hypotheses (don't re-run these):** adapter over-training; global
train/serve context mismatch (true only for `general_qa`); pad/eos collision.

---

## 8. Traps that make a run silently invalid

| trap | consequence |
|---|---|
| Ollama down | `RetrievalAgent` logs `Dense vector search failed` and **silently continues on BM25 only**. Completes fine; nothing in the output says so. |
| `LEGALAI_ENABLE_COMPL_AI=1` | canned answers bypass the graph, fake 1 ms telemetry |
| `LEGALAI_DETERMINISTIC=1` with repeats | identical repeats, CIs collapse to zero |
| `LEGALAI_ENABLE_EURLEX_LIVE=1` during a run | context depends on what the web returned that minute |
| Setting `LEGALAI_USE_ADAPTERS` by hand for a benchmark | rows get the wrong arm label; use `benchmark.py --arm` |
| Adding topologies to `benchmark.py MODES` | changes the Holm-corrected family size |
| Pooling the two arms into one mean | averages a system with its own control |
| Another GPU process | ~180 MiB spare VRAM → OOM, not just noise |
| Battery power | throttling distorts the latency the paper reports (step times went 5.2→9.7 s under sustained load) |

---

## 9. What is left to do, in order

**1. Re-time one run** (~10 min). GPU should be free first.
```powershell
cd "C:\Users\Charbel\Desktop\Legal AI\legalai"
# start server (peft arm)
$env:GENERATION_PROVIDER="local_peft"; $env:LEGALAI_USE_ADAPTERS="1"
$env:LEGALAI_DETERMINISTIC="0"; $env:LEGALAI_ENABLE_COMPL_AI="0"
.\.venv\Scripts\python.exe -u -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
# in another shell:
.\.venv\Scripts\python.exe -u probe_one_run.py all
```
Was **17.1 min/run** with `general_qa` broken (it alone was 56% of the clock).
Expect **~8 min** now. That number sets the benchmark estimate.

**2. Smoke test both arms** (~20 min). `.\run_experiment.ps1 -Smoke`
Then inspect answers — **stop** if: abstention everywhere (retrieval empty);
`peft` == `base` (arm switch broken); all topologies identical (sampling off);
canned GPAI template (COMPL-AI on).

**3. Full benchmark.** `.\run_experiment.ps1 -BenchmarkOnly`
Resumable: `-Resume -BenchmarkOnly`. Rows flush per run, so a crash at 400 keeps
399. **Estimate ~3 days at 8 min/run.**

**4. Analysis.** `analyze_results.py` → `evaluate_workflows.py` →
`make_paper_figures.py` → `make_topology_figure.py` →
`scripts/print_summary.py`. Verified end-to-end on synthetic data; the stats
recover injected effects correctly. Judge cost ~$0.21 of $2.76 remaining.

**5. Fill the paper.** ~27 `\TBD{}` in `main.tex`, from `results.json` — never by
hand. A `\TBD{}` in a submitted PDF is a bug.

### Charbel-only (blocks analysis, not the benchmark)
- **Review the 30 reference answers** in `eval_dataset.json` — all flagged
  `needs_review: true`. Every quality metric measures against them.
- **Relevance annotation** (`scripts/annotate_relevance.py`), or retrieval
  metrics stay suppressed. Do not re-enable any other way: old labels were copies
  of the retriever's own output, making precision@5 = 1.0 by construction.
- **Advisor sign-off:** title, RQ wording, the two model substitutions (§3), and
  `general_qa` running unadapted (§6).

---

## 10. Runtime reality — why not "a couple of hours"

Arithmetic, not tuning. ~15 tok/s per 4-bit 3B model; ~8 model calls per run;
a few hundred output tokens each.

```
540 runs × 8 calls × 300 tokens ÷ 15 tok/s ≈ 24 h   ← absolute floor
```
In 2 hours you get ~25 tokens per call. The pre-pivot run did 630 runs in 40 min
because generation was a **hosted API**. Running three fine-tuned 3B models on one
8 GB laptop is ~100× slower per token — that trade came with the pivot.

Levers if needed (each a **disclosed methodology change**):
`LLM_NUM_PREDICT` 1024→384 (~halves), `MAX_ITERATIONS` 2→1 (~30–40%),
repeats 3→2 (weakens the CIs). User has declined cutting repeats/queries.

**Latency-drift caveat:** the arms run consecutively, so thermal drift lands on
whichever ran second, and `elapsed_s` is a tested metric. Every run records
`started_at_utc`; the analysis reports within-arm drift (first vs last decile)
and warns above 15%. It **reports rather than corrects** — a regression fix would
hide a measurement problem behind a statistical one. Topology comparisons are
unaffected (modes interleave within an arm).

---

## 11. Key files

```
main.tex                     rewritten; ~27 \TBD{}
references.bib               +7 verified entries (LoRA, QLoRA, NewsQA, Dolly,
                             Llama3, Qwen2.5, Granite)
PEFT_Pivot_Implementation_Plan.md   original spec
RUN_HANDOFF.md               operational run guide (current)

legalai/
  config.py                  GENERATION_PROVIDER=local_peft, role registry,
                             LOCAL_UNADAPTED_ROLES, repetition penalty
  local_models.py            NEW — 4-bit loader + LoRA, LangChain wrapper,
                             token telemetry, per-model locks
  eurlex_search.py           NEW — live EU-source search, off by default (works;
                             only eur-lex is WAF-blocked)
  benchmark.py               3 topologies, --arm, arm cross-check, timeouts
  analyze_results.py         pairwise + arm_tests, drift check
  evaluate_workflows.py      metrics_table.tex + metrics_table_ablation.tex
  make_paper_figures.py      fig1–fig5 (fig5 = RQ2 ablation)
  agents/base.py             role-aware build_chat_llm
  agents/aggregator.py       EXPERT_KEYS fix
  finetune/                  prepare_datasets, train_qlora, check_vram,
                             validate_adapters, base_eval_loss,
                             backfill_eval_history
  notebooks/                 5 notebooks + _build_notebooks.py (regenerate with
                             `python notebooks/_build_notebooks.py`)
  probe_one_run.py           full-graph timing probe
  probe_legal_expert.py      isolates the legal expert
  probe_general_qa.py        isolates general_qa / token-cap check
  probe_aggregator.py        isolates aggregator at rising prompt sizes
  adapters/{legal,news,general_qa}/   trained LoRA weights + training_meta.json
  tests/                     101 tests
```

---

## 12. Findings that survive regardless of the benchmark outcome

Measured, in the paper, independent of how RQ1/RQ2 land:

1. **GQA, not parameter count, bounds multi-expert VRAM.** Phi-3.5-mini:
   384 KiB/token vs Llama's 112. Residency ≠ feasibility.
2. **Single-GPU concurrency is ~1.07×, not ~3×.** A "parallel" topology pays full
   coordination cost for a property the hardware doesn't deliver.
3. **Fine-tuning on short-answer benchmarks transfers answer *format* along with
   domain knowledge**, and where deployment expects long-form output that
   transfer can dominate the gain.
4. **PEFT on a small model with generic instruction data degrades it on long
   out-of-domain context**, to the point of unusability (§6).
5. **Adapter loss curves are not a quality signal** — 46–72% loss reduction
   alongside unusable generation.
6. **EUR-Lex now WAF-blocks non-browser clients**; the Cellar endpoint works.
