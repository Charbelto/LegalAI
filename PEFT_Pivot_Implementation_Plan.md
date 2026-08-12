# Implementation Plan: PEFT/LoRA Multi-Agent Specialization Pivot
### "When Is One Agent Enough?" → "Which Topology Best Combines Specialized Agents?"

**Project:** Legal AI — EU AI Act Compliance QA, Multi-Agent Routing Topologies
**Authors:** Charbel Toumieh (WMG, University of Warwick), Dr. Omid Chatrabgoun (Coventry University)
**Hardware target:** Laptop, RTX 4070 Laptop GPU (8GB VRAM), 32GB system RAM — fully local, no third-party LLM API calls for generation
**Status:** Planning document — several parameters below are proposed defaults, explicitly flagged for confirmation, not settled fact.

This document is written to be handed to anyone — a person or an AI assistant — with no other context, and be enough on its own to execute the work, explain why each decision was made, and answer follow-up questions about it.

---

## 1. Executive Summary

The project's original design compared a single-agent baseline against seven multi-agent topologies, all using one shared, unmodified language model (DeepSeek V4 Flash), to answer: **does multi-agent coordination beat a well-configured single agent on EU AI Act compliance QA?** That experiment is complete, its data is collected, and its finding was clear: no, it doesn't, at least not in this domain.

The advisor has directed a pivot. The new design keeps the same underlying system (retrieval, agent graph, evaluation methodology, metrics) but changes three fundamental things:

1. **Generation moves from one shared hosted model (DeepSeek) to three separate, small, locally-run open-weight models — one per domain expert — each specialized via PEFT/LoRA fine-tuning on its own dataset.** This makes the system fully local (no data leaves the laptop for generation) and turns each expert into an actual domain specialist rather than a generalist model with a domain-specific prompt.
2. **The topology set shrinks from 8 down to 3: fully sequential (ALL), fully parallel (PARALLEL), and DAG.** The single-agent baseline (SINGLE) and the remaining topologies (LEGAL-FIRST, PLANNER-BASED, VERIFY-ONLY, LEGAL-NEWS) are *not deleted from the codebase* — they stay implemented and usable in the running system — but they are excluded from the benchmark and from the paper entirely.
3. **The research question changes.** Because SINGLE is no longer part of the comparison, the paper can no longer claim anything about single-agent-vs-multi-agent. The new question is: *given three domain-specialized (PEFT fine-tuned) expert agents, which coordination topology combines them best, and why?* This moves the paper from the "single-vs-multi" literature (Jwalapuram, Tran & Kiela, Cemri) into the "topology design" literature (GPTSwarm, MacNet, AgentPrune) that the paper already reviews as related work — it just becomes the paper's primary lane instead of a secondary one.

Everything else — the 5 metric families, the statistical protocol (paired Wilcoxon, Holm correction, Cliff's delta), the abstention-handling rule, the 30-query evaluation set, the figures-and-table structure, and the judge (DeepSeek, used only for scoring, never for generation) — stays as-is. This is a re-pointing of the experiment's independent variable, not a rebuild of the whole methodology.

---

## 2. Why This Pivot Makes Sense (and Where It's in Tension)

**Why it's a legitimate direction:** specializing each expert agent via lightweight fine-tuning on its own domain data is a real, well-supported technique, and arguably more realistic than the original design's single shared corpus feeding three "specialist" prompts. It also removes a genuine weakness of the current paper — the judge and the system-under-test being the same model (DeepSeek), which biases the judge scores. Once generation is fully local and the judge stays a separate model (DeepSeek), that self-preference-bias caveat disappears cleanly.

**Where it's in real tension with the current paper, and why that's OK as long as it's acknowledged:** the entire current paper's methodology is built on holding the language model fixed and varying only topology, so that any measured difference is attributable to structure alone. Once each agent becomes a separately fine-tuned model on separate data, topology is no longer the only thing varying — specialization varies too. This is fine, but it means the paper must be honest that it answers a **different question** than before, not a refined version of the same one. Section 6 below (paper rewrite) handles this explicitly.

**The gap worth closing before finalizing:** as designed so far, there is no comparison showing whether the PEFT fine-tuning itself did anything. The paper would show "DAG beats PARALLEL beats ALL" (or whichever wins) using three fine-tuned agents, but never demonstrate that fine-tuning made those agents better than the same models un-fine-tuned. **Recommendation (confirm with advisor): keep a lightweight internal control** — run each of the 3 topologies once more with the same three base models but *without* the LoRA adapters loaded (prompt-only, same as the old design). This roughly doubles the benchmark (270 → 540 runs) but lets the paper make a much stronger claim: not just "topology X wins" but "specialization via PEFT measurably improves quality, and topology X is the best way to combine specialized agents." This is optional — the project works without it, but it closes an obvious reviewer question.

---

## 3. What Changes vs. What Stays the Same

| Component | Old (current, complete) | New (this pivot) |
|---|---|---|
| Generation model | One shared model: DeepSeek V4 Flash (hosted API) | Three separate local models, one per expert (see Section 4) |
| Fine-tuning | None — prompting only | QLoRA fine-tuning per agent, on a domain-specific dataset |
| Judge model | DeepSeek V4 Flash (same as generation — self-preference-bias caveat) | DeepSeek V4 Flash (now genuinely distinct from all 3 generation models — caveat resolved) |
| Topologies compared | 8 (SINGLE + 7 multi-agent) | 3 (ALL, PARALLEL, DAG) — rest stay implemented, excluded from benchmark/paper |
| Research question | Does multi-agent beat single-agent? | Which topology best combines 3 specialized agents? |
| Corpus (legal) | EU AI Act text only, static, 351 chunks | Same corpus retained, **plus** live search of EUR-Lex / EU Commission pages for recent legal updates |
| Corpus (news) | Live web search (already built, disabled during benchmark) | Same live web search, now paired with a fine-tuned news-specialist model |
| Corpus (general) | Shared corpus, generalist prompt | Fine-tuned generalist model, no additional retrieval |
| Evaluation set | 30 queries (simple/decomposable/routing), gold answers drafted by Llama 3.1 8B, unreviewed | Same 30 queries and gold answers, **user is manually reviewing and adjusting them** |
| Metrics | 5 families: lexical, judged quality, retrieval (suppressed), operational, structural | Unchanged |
| Statistical protocol | Each multi-agent topology vs. SINGLE, Holm-corrected | 3 pairwise comparisons among ALL/PARALLEL/DAG (or vs. their non-fine-tuned controls, if the ablation above is adopted), same Wilcoxon + Holm + Cliff's delta approach |
| Benchmark size | 30 × 8 × 3 = ~630-720 runs (7 benchmarked topologies were actually used) | 30 × 3 × 3 = 270 runs (540 if the ablation control is included) |
| Paper structure | Abstract → Intro → Related Work → Methodology → Results → Discussion → Threats → Future Work → Conclusion | Same section structure, same metric families, same figure types — content and framing updated throughout |

---

## 4. New Architecture: Three Specialized Agents

| Agent | Base model | Why this model | Fine-tuning method | Fine-tuning dataset | Why this dataset | Extra capability |
|---|---|---|---|---|---|---|
| **Legal expert** | Llama 3.2 3B | Small enough to run concurrently with the other two in 8GB VRAM; same model family already used elsewhere in the project (gold-answer drafting used Llama 3.1 8B) | QLoRA (4-bit base + LoRA adapter) | **LegalBench** (`nguha/legalbench`, Hugging Face) | Already cited in this paper's own Related Work section — no new citation-verification burden; English-language; well-established academic benchmark | Retains existing RAG against the 351-chunk EU AI Act ChromaDB corpus; **new**: live search against EUR-Lex and EU Commission digital-strategy pages for recently published legal material |
| **News expert** | Qwen2.5 3B-Instruct | Same size class as the other two, for a fair concurrent-parallel comparison | QLoRA | **NewsQA** (`Maluuba/newsqa`, Hugging Face) | 100,000+ human-written QA pairs over real news articles — the standard, well-documented choice for this exact task | Retains existing live web/news search (`auto_fetcher.py` / `scraper.py`) — already built, no new engineering needed here |
| **General Q&A expert** | Ministral 3B | Mistral's own compact edge-deployment model — real, current release, not a substitute; same size class as the other two | QLoRA | **Dolly-15k** (Databricks) | Human-written (not another AI's output, unlike Alpaca), which avoids a provenance/licensing footnote about training on another company's model outputs; permissively licensed | No additional retrieval — this agent is the "everything else" fallback, consistent with its original design role |

**Judge (unchanged in role, now cleanly independent):** DeepSeek V4 Flash, used only to score answers 1-5 on accuracy / completeness / groundedness against the gold reference. Never used for generation in this pivot, which is what resolves the self-preference-bias caveat.

**Why three different base models instead of one model with three adapters:** the advisor specifically asked for three different underlying models (Llama, Qwen, Mistral family), not one shared base with swapped adapters. This is a legitimate, common design choice (different model "personalities" per role) — just worth knowing it costs more VRAM than the alternative (one base + three adapters), which is exactly why all three were downsized to the 3B class rather than staying at 7-8B.

**Hardware reality, stated plainly:** three ~3-4B models at 4-bit quantization run roughly 2-2.5GB each for weights alone, so approximately 7-8GB combined before context/activation overhead — tight but plausible on an 8GB card, unlike the original 7-8B choices which definitively would not have fit three-at-once. **This has not been empirically verified yet.** First concrete implementation step (Section 5, Step 0) is loading all three quantized models simultaneously and confirming they actually run concurrently before anything else is built on top of that assumption.

---

## 5. Step-by-Step Implementation Plan

### Step 0 — Hardware feasibility check (do this first, before any fine-tuning work)
Load Llama 3.2 3B, Qwen2.5 3B-Instruct, and Ministral 3B simultaneously, each 4-bit quantized, on the RTX 4070 Laptop GPU, and confirm all three fit in 8GB VRAM and can run inference concurrently. If they don't fit, the fallback options, in order of preference, are: (a) more aggressive quantization (e.g., int4 with a shorter context window), (b) drop to even smaller variants if available, (c) as an absolute last resort, accept sequential loading for PARALLEL mode and disclose that plainly in the paper's Threats to Validity — the advisor was explicit that true concurrency is preferred, so this is the option to avoid if at all possible.

### Step 1 — Dataset acquisition and preparation
- Download LegalBench, NewsQA, and Dolly-15k from Hugging Face.
- Each dataset needs reformatting into a consistent instruction/Q&A format (question + answer, or instruction + response) suitable for supervised fine-tuning.
- **Open decision, propose a default:** none of these datasets need to be used in full — LegalBench has 162 tasks, NewsQA has 100k+ pairs, Dolly-15k has 15k examples. For a laptop-scale QLoRA run, a practical starting point is **2,000-3,000 examples per domain**, enough to see a real specialization effect without turning fine-tuning into a multi-day job. Adjust upward if training time allows.
- Filter/dedupe: make sure no query or source text in the legal dataset overlaps with the news dataset (the advisor specifically asked for this), since both eventually touch on AI-regulation-adjacent current events.

### Step 2 — QLoRA fine-tuning (three separate runs, one per model/dataset pair)
Standard, well-documented stack: Hugging Face `transformers` + `peft` + `trl` (SFTTrainer) + `bitsandbytes` for 4-bit quantization. Proposed starting hyperparameters (adjust empirically):
- LoRA rank (r): 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules: attention projection layers (standard choice for LoRA on decoder-only transformers)
- Quantization: 4-bit NF4 (QLoRA)
- Learning rate: ~2e-4
- Epochs: 3 (watch for overfitting given small dataset size)
- Batch size: 4-8 with gradient accumulation, tuned to fit in remaining VRAM after the frozen base model is loaded

Run this three times, once per (model, dataset) pair. Each run should be validated by generating a handful of test answers before moving on — checking a fine-tuned legal model actually writes more legally-grounded, article-citing answers than the un-tuned base model, for example — rather than assuming the fine-tune worked.

### Step 3 — Integrate fine-tuned models into the existing multi-agent graph
The system's architecture (planner, router, memory, retrieval, three domain experts, aggregator, validator, response — all built in LangGraph) does not need to be rebuilt. Concrete file-level changes:
- **`config.py`**: add configuration for the three new local model paths/names (fine-tuned checkpoint locations) and remove/deprecate the DeepSeek-as-generation path (keep DeepSeek config for judging only).
- **`agents/base.py`**: `build_chat_llm()` currently branches on `GENERATION_PROVIDER` (ollama vs. deepseek). This needs a third mode — loading a local fine-tuned QLoRA checkpoint per agent role, rather than a single shared model for all agents. This is the biggest code change: today, one model serves every expert; going forward, each expert node needs to load its *own* specific fine-tuned model.
- **`agents/legal.py`, `agents/news.py`, `agents/general_qa.py`** (or wherever each domain expert's node logic lives): update each to point at its own fine-tuned model, and add the legal agent's new live-search step (EUR-Lex / EU Commission pages) alongside its existing retrieval step.
- **`graph/workflow.py`**: no structural changes needed for ALL/PARALLEL/DAG — those routing paths already exist. Just confirm they still work correctly once each expert node is backed by a different underlying model rather than one shared one.
- **`embed.py`**: no changes needed for the legal corpus itself (still the same 351-chunk EU AI Act text) — this file is unaffected by the model swap.

### Step 4 — Reduce the benchmarked topology set
- **`benchmark.py`**: change the `MODES` list from the current 7 entries down to exactly 3: `["all", "parallel", "dag"]`. Leave every other topology's implementation untouched in `graph/workflow.py` — they remain fully functional and selectable via the API/UI, they're just not included in the benchmark sweep or the paper.
- **`run_experiment.ps1`**: update run totals in comments/documentation (30 × 3 × 3 = 270, or 540 if the fine-tuned-vs-base ablation from Section 2 is adopted).

### Step 5 — Re-run the benchmark
Same process as the original run (see `RUN_HANDOFF.md` for the exact sequence: sanity checks → smoke test → full run), just against the new 3-topology, 3-model setup. Confirm the smoke test produces genuinely different, sensible answers per topology before committing to the full run.

### Step 6 — Re-run analysis and regenerate figures/tables
- **`analyze_results.py`**: statistical comparisons change from "each topology vs. SINGLE" to pairwise comparisons among ALL, PARALLEL, and DAG (3 pairwise tests, Holm-corrected across that family of 3, same Wilcoxon signed-rank + Cliff's delta approach as before).
- **`make_paper_figures.py`**: figure generation logic mostly carries over unchanged (same 4 figure types), just with 3 bars/points per chart instead of 7 — actually simpler to read than before.
- **`metrics_table.tex`**: regenerate — with only 3 topology columns instead of 7, the table overflow problem from the previous version disappears entirely; it should fit in the normal (non-rotated) table layout now.

### Step 7 — Rewrite the paper
See Section 6 below for the specific section-by-section changes needed in `main.tex` and `references.bib`.

---

## 6. Paper Rewrite Plan (`main.tex` and `references.bib`)

**Title:** needs to change — "When Is One Agent Enough?" no longer describes the paper once SINGLE is removed. Proposed direction: something like *"Specialized but Structured: A Controlled Comparison of Coordination Topologies for PEFT-Tuned Multi-Agent Compliance Question Answering."* (Advisor should sign off on the exact title.)

**Abstract:** rewrite entirely. New framing: three domain-specialized agents (Llama 3.2 3B / legal, Qwen2.5 3B / news, Ministral 3B / general), each PEFT/LoRA fine-tuned on a domain-specific dataset, compared across three coordination topologies (sequential, parallel, DAG) on the same EU AI Act compliance QA task. Report whichever topology wins, by how much, and the cost tradeoffs — same style of quantitative claim as before, new numbers.

**Introduction — "What is already known" subsection:** the framing built around the single-vs-multi literature (Jwalapuram, Tran & Kiela, Cemri, Xu et al.) needs to shift from being the primary contrast to a brief acknowledgment ("prior work has shown single agents can match multi-agent systems when the model is held fixed and un-specialized; this paper asks a different question — once agents are genuinely specialized, does topology still not matter, or does it start to?"). This reframes those citations as motivation rather than the direct target of the paper's claim.

**Introduction — "What is left" / contributions / RQs:** this needs the most substantive rewrite. Proposed new RQs (confirm with advisor):
- **RQ1:** Given three domain-specialized (PEFT fine-tuned) expert agents, which coordination topology — sequential, parallel, or DAG — produces the best judged answer quality, and at what operational cost?
- **RQ2:** Does PEFT specialization itself measurably improve answer quality over the same base models un-tuned? *(Only answerable if the Section 2 ablation is adopted — otherwise drop this RQ.)*
- **RQ3:** How does topology affect operational cost (latency, tokens) once each expert is a distinct local model rather than one shared hosted model?

**Related Work:** Section 2.2 (multi-agent topology and orchestration — GPTSwarm, MacNet, DyLAN, AgentPrune, G-Designer, ARG-Designer) becomes the paper's primary point of contrast instead of secondary. Section 2.1 (single-vs-multi evidence) becomes brief motivating context rather than the main target. New citations needed for LegalBench (already have it — `guha2023legalbench` is already in `references.bib`), NewsQA, Dolly-15k, Llama 3.2, Qwen2.5, Ministral, and LoRA/QLoRA/PEFT methodology papers (e.g., the original LoRA paper, Hu et al. 2021, and the QLoRA paper, Dettmers et al. 2023 — neither is currently in `references.bib` and both need adding and verifying, the same way all 31 existing citations were verified earlier).

**Methodology:** System-under-test subsection needs to describe three separate fine-tuned models instead of one shared model. Topologies subsection: drop SINGLE, LEGAL-NEWS, LEGAL-FIRST, PLANNER-BASED, VERIFY-ONLY from the *compared* set (a sentence noting they remain implemented but out of scope is worth keeping, for transparency). New subsection needed: **fine-tuning protocol** (datasets, QLoRA hyperparameters, hardware, training details) — this is new content that doesn't exist in the current paper at all.

**Results:** Table 1 shrinks from 7 columns to 3 — should fit normally now, no rotation/resize needed. All four figures get regenerated with 3 series instead of 7. RQ-by-RQ results subsections get rewritten around the new RQs above.

**Discussion, Threats to Validity, Future Work, Conclusion:** all need rewriting to match the new finding once you have it — can't be drafted meaningfully until Step 5-6 actually produce results. Threats to Validity should keep the existing self-preference-bias paragraph (now stating the caveat no longer applies, since judge and system-under-test are finally distinct) and add a new limitation specific to this pivot: three different base models means any topology difference is entangled with whichever model happens to sit in which graph position, which is a real interpretive limit worth stating plainly rather than glossing over.

**`references.bib`:** add and independently verify (same rigor as the existing 31 entries): NewsQA, Dolly-15k, Llama 3.2, Qwen2.5, Ministral/Mistral, the LoRA paper, and the QLoRA paper. LegalBench is already present and verified.

---

## 7. Open Decisions Still Needing Sign-Off

These are flagged rather than silently assumed, per the instruction that this document should be usable by anyone without needing to guess:

1. **The fine-tuned-vs-base ablation (Section 2)** — recommended, not yet confirmed. Doubles benchmark size if adopted (270 → 540 runs).
2. **Exact QLoRA hyperparameters and per-domain training-set size** — proposed defaults given in Section 5, Step 1-2; these are starting points, not tuned values, and should be adjusted based on how training actually goes.
3. **Whether the paper's title and RQ wording (Section 6) match what the advisor actually wants** — proposed here, not yet his sign-off.
4. **The 8GB VRAM concurrent-loading assumption (Section 4)** — proposed and reasoned through, but not yet empirically tested. This is Step 0 for a reason: everything downstream depends on it.
5. **Judge model** — user has explicitly deferred this decision ("I will think about that later"); DeepSeek is the placeholder for now.

---

## 8. Anticipated Questions and Answers (for a meeting, defense, or handoff)

**"Why did you drop the single-agent baseline?"** Because the advisor's direction was to specialize each agent via fine-tuning and compare coordination topologies among themselves — once agents are no longer generalist and identical, "single vs. multi" stops being the relevant question. This paper now answers a different, related question: how best to combine specialized agents, not whether to have more than one.

**"Doesn't that contradict your first paper's finding?"** No — it answers a different question about a different system. The first paper's finding (a well-configured single agent matches or beats multi-agent coordination when every expert shares one identical, unspecialized model) still stands on its own terms. This pivot asks what happens once the agents are actually different, specialized models — a scenario the first paper's own Future Work section explicitly flagged as worth testing.

**"Why only three topologies now, not seven?"** Practicality and clarity: with three genuinely different specialized models instead of one shared model, the topology×model design space grows quickly. Sequential, parallel, and DAG were chosen because they represent three structurally distinct coordination patterns (strict chain, full concurrency, and converging dependency) without multiplying the experiment further. The other topologies remain fully implemented and usable — they're excluded from the paper for scope, not because they don't work.

**"Why these three models specifically?"** They needed to be open-weight (so they can be fine-tuned and run entirely locally, satisfying the offline/no-third-party-API requirement), small enough that all three can plausibly run concurrently on an 8GB VRAM laptop GPU (ruling out the originally-considered 7-8B versions and models like Kimi K2, which needs 350GB+ even quantized), and from three different model families as specifically requested, rather than one shared base with different adapters.

**"Why keep DeepSeek at all?"** Only as the judge, never for generation. Since it no longer produces any of the answers being scored, it's now a genuinely independent evaluator — which actually fixes a real weakness the first paper had to disclose as a limitation.

**"Isn't sending data to DeepSeek for judging a privacy problem?"** The data sent to the judge is the (already public) EU AI Act–derived questions and the system's own generated answers — not private or proprietary data. The offline requirement is specifically about generation (not sending your own queries/data to a third party to get answers), which local generation satisfies regardless of which model does the judging.

**"How do you know the fine-tuning actually did anything?"** This is exactly the gap flagged in Section 2 — without the recommended base-vs-fine-tuned ablation, the paper cannot make this claim directly, only that "these fine-tuned, specialized agents perform this way under these topologies." If that claim matters to the project, the ablation should be adopted.

**"What's the sample size, statistically?"** Same 30 queries as before (three query types: simple, decomposable, routing — chosen because they map naturally onto the three specialized domains), three repeats per cell, now against 3 topologies instead of 8 — 270 runs (or 540 with the ablation). Same caveat as the original paper: 30 queries reliably detects large effects, not small ones; non-significant results should be read as inconclusive, not as proof of equivalence.
