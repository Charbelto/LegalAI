# Handoff brief: run the PEFT topology benchmark

Paste this whole file to an AI agent that has shell access to Charbel's Windows
machine (Claude Code, Cursor, etc.). It has everything needed to execute the run
and report back.

---

## Your task

Execute a benchmark that compares three LLM agent coordination topologies over
three separately fine-tuned expert models, then report the results. You are
running an experiment whose numbers will go into an academic paper, so
**correctness matters more than completion**: if something looks wrong, stop and
report it rather than working around it.

**Absolute rule: never invent, estimate, extrapolate or hand-edit a result.** If a
step fails, report the failure. A missing number is fine; a fabricated one ends the
project.

## Environment

- Windows, PowerShell.
- Project root: `C:\Users\Charbel\Desktop\Legal AI`
- Code: `C:\Users\Charbel\Desktop\Legal AI\legalai`
- Virtualenv: `C:\Users\Charbel\Desktop\Legal AI\legalai\.venv`
- GPU: RTX 4070 Laptop, 8 GB. This is the binding constraint on everything below.
- **Generation: fully local.** `GENERATION_PROVIDER=local_peft`. Three separate
  4-bit models, one per domain expert, each with its own LoRA adapter:

  | role | base model | adapter trained on |
  |---|---|---|
  | legal | `unsloth/Llama-3.2-3B-Instruct` | LegalBench |
  | news | `Qwen/Qwen2.5-3B-Instruct` | NewsQA |
  | general_qa | `ibm-granite/granite-3.1-2b-instruct` | Dolly-15k |

  Coordination nodes (planner, router, memory, aggregator, validator, response)
  share the general expert's **base** weights with the adapter disabled — a
  fourth resident model does not fit in 8 GB, and keeping specialisation confined
  to the experts is what makes topology the independent variable.
- Embeddings still run on **local Ollama** (`nomic-embed-text`). Ollama must be
  running or retrieval silently returns zero documents.
- Vector store: ChromaDB at `legalai\chroma_storage`, the EU AI Act.
- The system is a FastAPI backend + LangGraph agent graph. `benchmark.py` starts
  the server itself if one is not already running.

## What the experiment is

One agent graph (router, retrieval, three domain experts, aggregator, validator,
response) is executed under three coordination topologies — `all` (sequential),
`parallel` (concurrent), `dag` (converging) — in **two arms**:

- **`peft`** — each expert loads its LoRA adapter.
- **`base`** — the identical base models, no adapters. This is the control that
  makes RQ2 ("did specialisation help?") answerable.

30 queries x 3 topologies x 3 repeats x 2 arms = **540 runs**.

The other topologies (`single`, `legal_first`, `planner_based`, `verify_only`,
`legal_news_parallel`) remain implemented and selectable, but are **not** part of
the benchmarked set. Do not add them back.

## Realistic timings — read before starting

Local 3B models are far slower per token than the hosted API used in the previous
study. Measure before committing:

- The previous DeepSeek run did 630 runs in ~40 minutes.
- Local generation runs roughly 14 tokens/second per expert, and a single run
  makes ~8 model calls.
- **Expect several minutes per run, i.e. on the order of 24-40 hours for 540
  runs.** Get a real per-run figure from the smoke test (Step 2) and multiply,
  rather than trusting this estimate.

The run is resumable (`-Resume`), so it can be stopped and continued.

## Step 0 — preconditions

```powershell
cd "C:\Users\Charbel\Desktop\Legal AI\legalai"
.\.venv\Scripts\Activate.ps1

# Ollama up (needed for EMBEDDINGS even though generation is local)
curl http://localhost:11434/api/tags

# Corpus present: expect several hundred chunks
python -c "import utils; print('chunks:', utils.get_db_document_count())"

# Code invariants (75 tests, no GPU needed). Must all pass.
python -m pytest tests -q
```

**If `chunks` is 0 or implausibly small (tens rather than hundreds)**, the corpus
is missing or has been overwritten. A live news fetch with `clear_existing=True`
replaces the Act with news chunks — this has happened before. Re-ingest:

```powershell
python embed.py --replace
```

`embed.py` pulls CELEX:32024R1689 (Regulation (EU) 2024/1689, the Official
Journal text) from the EU Publications Office's Cellar endpoint. Note that
`eur-lex.europa.eu` answers non-browser clients with an AWS WAF challenge and the
old Parliament URL now returns HTTP 202 with an empty body, so those sources will
not work headless. `embed.py` raises rather than embedding an empty document if
every source fails — do not "fix" that by lowering the length floor.

**If any test fails, stop and report.** The tests guard the exact defects that
invalidated earlier runs.

## Step 1 — hardware and adapter checks

```powershell
python finetune\check_vram.py --concurrent
```

Must print `Step 0 verdict: PASS`. It confirms all three models are resident, all
three generate, and their combined KV cache fits. Known-good numbers: 5979 MiB of
weights, 1089 MiB free, 912 MiB of KV cache at the concurrent-phase context
length.

**If it fails on KV cache rather than residency**, the cause is almost certainly
a model without grouped-query attention. Check the `KiB/tok` column it prints:
anything near 384 KiB/token (as Phi-3.5-mini is) will not co-reside. Fallbacks in
order: lower `LEGALAI_LOCAL_MAX_INPUT_TOKENS`, swap for a smaller GQA model, and
only as a last resort sequential loading — which must then be disclosed in the
paper's Threats to Validity.

If the adapters do not exist yet:

```powershell
python finetune\prepare_datasets.py --domain all
python finetune\train_qlora.py --role all
python finetune\validate_adapters.py
```

The defaults are the settings actually used, so no flags are needed: 1,500
examples per domain (1,425 after dedupe), 3 epochs, 512-token sequence limit,
LoRA r=16 / alpha=32. Measured cost on the experiment GPU: 537 optimiser steps at
~5.9 s/step, so **roughly 50 minutes per adapter and ~2.5 hours for all three**.
The earlier 1024-token setting took ~4 hours per adapter and was abandoned.

**If you change any training parameter, retrain all three.** An expert trained
with more epochs, longer sequences or more examples than its peers makes any
topology difference partly "which position holds the best-trained expert" — a
confound of your own making, stacked on the model-position one the paper already
discloses. This is why `--role all` is the default and why a partially-trained set
should be deleted rather than topped up.

`validate_adapters.py` must not report any adapter as `inert`. An inert adapter
means the `peft` arm would be measuring the base models under a `peft` label,
which silently destroys RQ2.

Check `finetune\data\manifest.json` before training: if any domain's
`mean_output_chars` is below 40, the targets are too short and the resulting
adapter will answer in a few characters. Raw LegalBench answers average 3.8
characters, which is why `--legal-format irac` is the default. Also confirm the
three `train_examples` counts are near-identical — an uneven split is the same
consistency problem as uneven hyperparameters.

## Step 2 — smoke run (measure per-run time here)

```powershell
.\run_experiment.ps1 -Smoke
```

This runs 1 query x 3 topologies x 1 repeat per arm into
`benchmark_runs_smoke.jsonl` (kept separate from real data on purpose).

**Now inspect the answers — this is the most important check in the process:**

```powershell
python -c "import json; [print(r['arm'].ljust(5), r['mode'].ljust(10), '|', repr(r.get('response',''))[:150]) for r in map(json.loads, open('benchmark_runs_smoke.jsonl', encoding='utf-8')) if r.get('success')]"
```

Decide:

- **Real, differing answers across topologies and arms** -> proceed. Note the
  `elapsed_s` values and project the full run.
- **Most or all return `Insufficient authoritative support -- recommend expert
  review.`** -> STOP. Retrieval is returning nothing usable. Check `chunks` and
  that Ollama is up. Do not proceed.
- **`peft` and `base` answers identical** -> STOP. Either the adapters are inert
  or the arm switch is not taking effect. Re-run `validate_adapters.py`.
- **All topologies identical within an arm** -> STOP. Sampling is not happening;
  `LEGALAI_DETERMINISTIC` must be `0` during the run.
- **Any answer looks like a canned marketing-style template about GPAI provider
  tiers** -> STOP. That is a demo feature that must stay disabled
  (`LEGALAI_ENABLE_COMPL_AI=0`); it bypasses the agent graph entirely.

## Step 3 — the full run

```powershell
.\run_experiment.ps1 -BenchmarkOnly
```

Runs both arms in sequence, `peft` then `base`, appending into one
`benchmark_runs.jsonl` with an `arm` column. `-BenchmarkOnly` stops before
analysis because the reference answers still need human review.

Notes:

- Leave the machine on and awake. Disable sleep.
- **Do not run anything else GPU-heavy at the same time.** With only ~180 MiB of
  spare VRAM, another CUDA process will cause an out-of-memory failure, not just
  noisy latency. Closing browser windows frees VRAM if it is tight.
- `ABSTAINED` on some runs is normal and is recorded as data, not an error.
- **If it dies partway**, resume without losing work:
  `.\run_experiment.ps1 -Resume -BenchmarkOnly`
- To run only one arm: `-Arms peft` or `-Arms base`. Running only one drops RQ2
  from the paper; it does not silently degrade.

Outputs: `benchmark_runs.jsonl`, `run_meta.json`, and `run_meta_peft.json` /
`run_meta_base.json` (per-arm, because arm 2 overwrites `run_meta.json`).

When it finishes, report total duration, completed vs failed counts, and the
abstention rate per arm and topology:

```powershell
python -c "import json,collections; rows=[json.loads(l) for l in open('benchmark_runs.jsonl',encoding='utf-8') if l.strip()]; c=collections.Counter((r['arm'], r['mode'], bool(r.get('abstained'))) for r in rows if r.get('success')); [print(a.ljust(5), m.ljust(10), 'abstained' if ab else 'answered', n) for (a,m,ab),n in sorted(c.items())]"
```

## Step 4 — analysis (only after the human steps below)

Two things must happen first, and **only Charbel can do them** — do not attempt
them yourself and do not skip them:

1. **Human review of the 30 reference answers** in `eval_dataset.json`. They were
   drafted by a model and are all flagged `needs_review: true`. Every quality
   metric is measured against them.
2. **Relevance annotation** (`python scripts\annotate_relevance.py`), or accept
   that retrieval metrics stay suppressed. Do not re-enable them by any other
   means: the previous labels were copies of the retriever's own output, which
   made precision@5 equal to 1.0 by construction.

The judge is configured in `legalai\.env`. Current default: DeepSeek V4 Flash
(`JUDGE_PROVIDER=deepseek`, `JUDGE_MODEL=deepseek-v4-flash`, ~$1-2 for 540 runs,
capped by `JUDGE_BUDGET_USD`, cached so a re-run never pays twice).

**Do not set `GENERATION_PROVIDER=deepseek` while the judge is DeepSeek.** That
would be a model grading its own answers, which is precisely the limitation this
pivot removes.

Validate the judge with one call before spending on 540:

```powershell
python llm_judge.py --check
.\run_experiment.ps1 -SkipBenchmark
```

## What to report back

1. The full output of `python scripts\print_summary.py`.
2. From `results.json`, the `provenance` block — including `arms`,
   `runs_per_arm`, `expert_models` and `ablation_tested`.
3. Any warning lines printed during analysis, verbatim — especially about judge
   failures, gold answers needing review, truncation, or a missing ablation.
4. Confirmation that these files exist: `analysis_summary.csv`,
   `by_query_type.csv`, `significance.csv`, `results.json`, `metrics_table.tex`,
   `metrics_table_ablation.tex`, and 6 PNGs in `paper_figures\`.
5. For each (topology, arm): mean end-to-end latency, mean judge average and
   abstention rate — copied from `analysis_summary.csv`, not retyped from memory.
6. From `finetune\vram_report.json`: the concurrency speedup. The paper reports
   it, so it must come from a measurement, not from this document.

## Things that would invalidate the experiment — do not do them

- Setting `LEGALAI_ENABLE_COMPL_AI=1` (canned answers bypass the graph and report
  fake 1 ms telemetry).
- Setting `LEGALAI_DETERMINISTIC=1` for a multi-repeat run (greedy decoding makes
  all repeats identical and collapses every confidence interval to zero).
- Setting `LEGALAI_ENABLE_EURLEX_LIVE=1` during a run. A live legal lookup makes
  the legal expert's context depend on what the web returned that minute, so two
  runs of the same query stop being comparable.
- Setting `LEGALAI_USE_ADAPTERS` by hand for a benchmark. `benchmark.py --arm`
  sets it and cross-checks it against what the server reports; overriding it
  manually is how 270 rows get labelled with the wrong arm.
- Adding topologies back into `benchmark.py MODES`. The paper describes three.
- Editing `analyze_results.py` to make a test "work", loosening the abstention
  rule in `agents/aggregator.py`, or pooling the two arms into one mean.
- Mixing smoke-run rows into `benchmark_runs.jsonl`.
- Filling any number into `main.tex` yourself. Every `\TBD{}` gets filled from
  `results.json` afterwards, deliberately.
- Deleting `legalai\archive\pre_fix_20260725\` or
  `legalai\chroma_storage_stale_news_backup\` — both are preserved provenance.

## Background you may need

An earlier run of this experiment was invalid for five independent reasons, all
now fixed and all covered by tests: an aggregator rule where one abstaining expert
silenced the whole ensemble; a canned-answer path that hijacked 10% of queries; a
`single` mode that actually ran 2-3 agents; statistics that paired on repeats
instead of queries; and an unset context window that silently truncated retrieved
text. `FIXES_APPLIED.md` documents all of it. If you find yourself about to undo
any of those fixes to make something pass, that is the signal to stop and report
instead.

Two defects were found while building the PEFT pipeline and are worth knowing
about, because both were silent:

- **Phi-3.5-mini could not co-reside** with the two 3B experts, not because of
  parameter count but because it has no grouped-query attention (32 KV heads),
  making its KV cache 384 KiB/token against Llama's 112. All three models fitted
  in VRAM; only the cache did not. A residency check alone would have passed it.
- **Raw LegalBench targets average 3.8 characters.** Training the legal expert on
  them directly produces an adapter that answers in one word, which the judge
  scores as incomplete — indistinguishable in the results from "specialisation
  made quality worse". `prepare_datasets.py` now warns when any domain's mean
  target falls below 40 characters.
