# Citation Notes

Verification date: **2026-07-26**. Every claim below is tied to a paper confirmed to exist via live web search on this date (title, author surnames, year, and arXiv ID / venue all checked against arXiv, publisher, or official project pages). Context: a paper arguing "(a) a single LLM agent sometimes beats a multi-agent system, and (b) when multi-agent does help, communication *topology* matters more than raw agent count," evaluated on EU AI Act compliance QA.

---

## A. Single-agent vs. multi-agent evidence

**`jwalapuram2026illusion` — The Illusion of Multi-Agent Advantage (2026)**
Systematically compares automatically-generated multi-agent systems (MAS) against a Chain-of-Thought + Self-Consistency single-agent baseline and argues that most of the reported "MAS advantage" in prior literature is really just an artifact of MAS spending more aggregate inference compute (redundant sampling), not expert collaboration per se.
**PRE-EMPTS the core claim.** This is the single closest paper to "single agent is sometimes better than multi-agent" — it argues the *entire premise* of MAS superiority is largely a compute-accounting illusion. Any paper making a similar claim must explicitly differentiate itself from this one (e.g., by controlling for compute differently, using a different task domain — legal/compliance QA rather than generic reasoning — or focusing on topology rather than agent count).

**`cemri2025mast` — Why Do Multi-Agent LLM Systems Fail? (MAST) (2025)**
Empirically catalogs *why* MAS underperform: 14 failure modes across 3 categories (system design, inter-agent misalignment, task verification) from 1,600+ annotated traces across 7 MAS frameworks. Does not claim single-agent superiority directly, but documents the failure mechanisms that would produce it.
**Partially pre-empts / strongly motivates.** Doesn't make the head-to-head claim itself but supplies the mechanistic "why" that a paper like ours would otherwise have to establish from scratch. Any novel contribution should cite this and either (a) map its own observed failures onto the MAST taxonomy, or (b) show a failure mode MAST doesn't cover.

**`tran2026singleagent` — Single-Agent LLMs Outperform Multi-Agent Systems on Multi-Hop Reasoning Under Equal Thinking Token Budgets (2026)**
Under equal reasoning-token budgets, single agents match or beat five multi-agent variants on multi-hop QA (FRAMES, MuSiQue), with an information-theoretic (Data Processing Inequality) argument that inter-agent handoffs can only lose information. Explicitly diagnoses that most published MAS advantages come from letting MAS spend 2–4x more tokens than the single-agent baseline.
**PRE-EMPTS the core claim almost exactly**, including the "the field wasn't controlling for compute" diagnosis. A novel paper needs a different angle: different domain (legal/EU AI Act QA vs. generic multi-hop reasoning), different independent variable (topology, not just agent-count/compute), or a boundary condition this paper doesn't test (e.g., regulatory/compliance tasks requiring retrieval + multi-step statutory interpretation).

**`xu2026oneflow` — Rethinking the Value of Multi-Agent Workflow: A Strong Single Agent Baseline (2026)** (proposes the "OneFlow" algorithm)
Across seven benchmarks (coding, math, QA, domain reasoning, planning/tool-use), shows a well-tuned single agent matches homogeneous multi-agent workflows and, via KV-cache reuse, is cheaper — and proposes OneFlow to auto-compile workflows into efficient single-agent execution.
**PRE-EMPTS the claim from an efficiency/systems angle.** Combined with items 1 and 3, there are now (as of mid-2026) at least three independent papers showing single-agent parity or superiority. The "single agent can win" finding is no longer novel by itself — the differentiator has to be the *topology* argument (see Section B) or the *domain* (EU AI Act compliance QA specifically).

**`li2024moreagents` — More Agents Is All You Need (2024)**
Shows that simple sampling-and-voting ("Agent Forest") scales LLM performance with the *number* of agent instances, largely orthogonal to more complex multi-agent orchestration schemes.
**Sets up the "more agents/calls help" side of the debate** the paper must reconcile with the single-agent-wins papers above — useful as the "one side of the pre-2026 consensus" this paper pushes back on, alongside item 6.

**`chen2024morellmcalls` — Are More LLM Calls All You Need? Towards Scaling Laws of Compound Inference Systems (2024)**
Derives scaling laws for compound inference systems (Vote, Filter-Vote) showing performance from more LM calls is *non-monotonic* — more calls can hurt past a point, depending on task difficulty distribution.
**Important nuance, not pre-emption.** This already shows "more calls/agents" isn't simply better — i.e., it already gestures at "count isn't the only variable," which is adjacent to (but distinct from) a topology-centric argument. Should be cited as prior evidence that naive agent-count scaling is not the right lens, motivating a shift to studying structure/topology instead.

---

## B. Multi-agent topology / structure design

**Bottom line: this is an active, crowded subfield.** GPTSwarm, MacNet, DyLAN, AgentPrune, G-Designer, ARG-Designer, and AgentDropout are all 2023–2026 papers whose *entire contribution* is designing, pruning, or learning communication topology for multi-agent LLM systems. A 2026 survey (`zhu2026masorchestrationsurvey`) already taxonomizes topology choices (centralized / decentralized / hierarchical, static vs. dynamic-adaptive) across major frameworks.

- **`zhuge2024gptswarm`** (GPTSwarm, ICML 2024): represents agents as optimizable computation graphs; learns both node prompts and edge connectivity.
- **`qian2024macnet`** (MacNet, ICLR 2025): DAG-organized collaboration scaling to 1,000+ agents; finds *irregular* topologies outperform regular ones and a "collaborative scaling law."
- **`liu2023dylan`** (DyLAN, 2023): dynamic agent network with an Agent Importance Score for automatic team/topology selection.
- **`zhang2024agentprune`** (AgentPrune / "Cut the Crap," ICLR 2025): one-shot pruning of the spatial-temporal communication graph for token efficiency and adversarial robustness.
- **`zhang2024gdesigner`** (G-Designer, 2024): variational graph auto-encoder that generates task-adaptive communication topologies per query.
- **`li2025argdesigner`** (ARG-Designer / "Assemble Your Crew," AAAI 2026 Oral): autoregressive graph generation that jointly designs *both* team composition (which roles/how many agents) and communication edges from a natural-language task query.
- **`wang2025agentdropout`** (AgentDropout, ACL 2025): drops redundant agent nodes/edges across communication rounds for token efficiency and accuracy gains.
- **`zhu2026masorchestrationsurvey`** (Future Internet, 2026): survey proposing a three-topology + adaptivity taxonomy across six major frameworks (LangGraph, CrewAI, AutoGen, OpenAI Agents SDK, MetaGPT, DSPy).

**This directly undercuts a "nobody studies agent structure/topology" gap claim.** Topology design — automated, learned, pruned, or graph-generative — is a well-established 2024–2026 line of work with its own survey already published. A paper cannot claim novelty merely for "considering topology." Where novelty could remain: (a) applying/benchmarking these *existing* topology-design ideas specifically against single-agent baselines in a **legal/regulatory compliance QA** setting (which none of B1–B8 study), or (b) showing that topology choice matters *less* than expected once compute/tokens are equalized (extending the equal-budget methodology of `tran2026singleagent` to topology comparisons, which none of the topology papers do — they compare topologies to each other or to naive MAS baselines, not to a token-matched single agent).

---

## C. Frameworks

- **`wu2023autogen`** (AutoGen): general conversable multi-agent framework, human-in-the-loop.
- **`hong2023metagpt`** (MetaGPT, ICLR 2024): encodes Standard Operating Procedures into agent prompts/roles for software-engineering-style pipelines.
- **`li2023camel`** (CAMEL, NeurIPS 2023): role-playing communicative agents, foundational "society of agents" framework.
- **`langchain2024langgraph`** (LangGraph): graph/state-machine orchestration library; no companion peer-reviewed paper exists — cited as software (GitHub repo/docs).

These are implementation substrates, not evidence for or against the paper's thesis; cite them when describing what topology/framework was used to implement experimental conditions.

---

## D. Evaluation methodology

**`zheng2023mtbench`** establishes LLM-as-a-judge as a scalable human-preference proxy (~85% agreement with humans, exceeding human-human agreement) but explicitly flags position, verbosity, and **self-enhancement bias** as limitations.

**`wataoka2024selfpreference`** directly documents self-preference bias: LLM judges assign higher scores to lower-perplexity text, which correlates with a judge favoring outputs stylistically similar to its own (i.e., generations from itself or closely related models). **This is directly load-bearing for methodology**: if the paper's single-agent condition and multi-agent condition are judged by an LLM, and one condition's outputs are more "judge-like" in style, results could be confounded by self-preference bias rather than genuine quality differences. The paper must either use a judge model disjoint from all generator models, use human evaluation, or explicitly test/control for this bias — and cite this paper when doing so.

**`papineni2002bleu`, `lin2004rouge`, `wilcoxon1945individual`, `holm1979simple`, `cliff1993dominance`** — standard methodology citations (automatic metrics, non-parametric significance testing, multiple-comparison correction, effect size). No novelty claim rides on these; they are just correctly-cited tools.

---

## E. Legal / compliance domain

- **`guha2023legalbench`** (LegalBench, NeurIPS 2023 D&B): 162-task collaboratively-built benchmark for legal reasoning (issue-spotting, rule-recall, interpretation, application, conclusion, rhetoric) — general legal reasoning, not EU AI Act specific.
- **`pipitone2024legalbenchrag`** (LegalBench-RAG, 2024): benchmark for the *retrieval* step of legal RAG pipelines (contracts, privacy policies); narrow document-type scope.
- **`cui2023chatlaw`** (ChatLaw, 2023): Chinese-legal-domain LLM; later version reframed as a multi-agent (Role-Aligned Mixture-of-Experts, SOP-style law-firm roles) legal assistant.
- **`guldimann2024complai`** (COMPL-AI, 2024): first technical interpretation of the EU AI Act into 18 measurable requirements, with a 12-model benchmarking suite. **This is the closest existing work to "EU AI Act compliance evaluation,"** but it evaluates *models* against Act-derived technical requirements (robustness, safety, fairness, etc.) — it does **not** evaluate multi-agent vs. single-agent *systems* answering compliance *questions*. That gap (agent-topology comparison specifically on EU AI Act QA) appears open.
- **`li2025lexrag`** (LexRAG, SIGIR 2025): multi-turn legal-consultation RAG benchmark (1,013 dialogues, 17,228 candidate articles) — general legal consultation, not EU AI Act, not agent-topology-focused.
- **`raptopoulos2025pakton`** (PAKTON, EMNLP 2025 Oral): multi-agent (questioner/researcher) framework for QA over long legal agreements (contracts), with a RAG component. **This is a working legal multi-agent system**, but targets contract QA, not regulatory-compliance QA, and does not compare against a single-agent baseline or study topology as an independent variable.

None of E23–E28 directly studies single-vs-multi-agent topology on EU AI Act compliance QA. This combination — EU AI Act compliance QA **as the task**, and agent-topology-vs-agent-count **as the independent variable** — is the most plausible remaining gap, provided the paper also engages honestly with Section A/B above.

---

## UNVERIFIED — DO NOT CITE

- **HERA ("an agentic orchestration system").** Web search surfaced *multiple* distinct real papers using the name "HERA," but none matches "an agentic orchestration system for legal/compliance work" with confidence:
  - arXiv 2604.00901, "Experience as a Compass: Multi-agent RAG with Evolving Orchestration and Agent Prompts" (Li & Ramakrishnan, 2026) — a generic multi-agent RAG orchestration framework called HERA, not legal-domain-specific.
  - arXiv 2605.24598, "Hera: Learning Long-Horizon Coordination for Device-Cloud Collaborative LLM Agents" (Zhang et al., 2026) — a device/cloud LLM-agent coordination system, unrelated to legal or compliance work.
  - Neither is confirmed to be the "legal/compliance HERA" implied by the prompt. Rather than guess which one (or invent a third), this item is left unverified. If you have a specific source (e.g., an internal tool, a preprint from a specific lab, or a conference poster) for "HERA" as a legal agentic system, supply the exact title/venue and it can be re-verified.

No other items from the requested list were left unverified — all other 28 requested papers/standards were confirmed to exist with correct titles, author surnames, years, and arXiv IDs/venues as of 2026-07-26.

---

## What is left that is actually novel

Being blunt: **the "single agent can beat multi-agent" finding, by itself, is not novel in mid-2026.** At least three papers (`jwalapuram2026illusion`, `tran2026singleagent`, `xu2026oneflow`) have already published essentially this result in 2026 alone, on top of the 2024 scaling-law caveats (`chen2024morellmcalls`). And **"multi-agent topology matters" is also not a novel observation** — it is an entire established subfield with dedicated methods (GPTSwarm, MacNet, DyLAN, AgentPrune, G-Designer, ARG-Designer, AgentDropout) and its own 2026 survey.

What *does* appear to still be open, based on this search:

1. **The specific task domain.** Nobody in this list evaluates single-vs-multi-agent topology comparisons on **EU AI Act compliance QA** specifically. COMPL-AI benchmarks *models* against the Act's requirements; PAKTON and ChatLaw build legal multi-agent systems for contracts/general law, not regulatory compliance; LegalBench/LegalBench-RAG/LexRAG are general legal reasoning/RAG benchmarks. A rigorous single-vs-multi-agent, topology-vs-agent-count comparison *on EU AI Act QA specifically* has not been done by any paper found here.
2. **Connecting the "single-agent-can-win" literature to the "topology-design" literature.** The topology papers (Section B) mostly compare various multi-agent topologies to each other or to naive/undertuned multi-agent baselines — none of them benchmark their learned topologies against a properly token/compute-matched *single-agent* baseline in the style of `tran2026singleagent`. Showing whether sophisticated topology design (AgentPrune, G-Designer, ARG-Designer) still beats a strong equal-budget single agent, specifically in a domain like legal compliance QA, would be a genuine, currently-missing data point.
3. **A domain-specific failure-mode account.** MAST (`cemri2025mast`) taxonomizes MAS failures in general; whether those same failure modes (or different ones specific to statutory interpretation / cross-referencing regulatory articles) dominate in EU AI Act QA is untested.

What is **not** available as a novel contribution without a stronger angle: simply claiming "we show single agent can beat multi-agent" or "we show topology matters more than agent count" as a headline finding — both are already published. The paper's contribution needs to be framed as the **domain-specific empirical test** (EU AI Act compliance QA) of claims that are already established in general-purpose settings, ideally paired with a mechanistic account (tying results back to MAST failure categories or the Data-Processing-Inequality argument) of *why* the domain does or doesn't behave like the general-purpose benchmarks already studied.
