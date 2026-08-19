"""Configuration and prompts for the Legal AI multi-agent system."""

import os

# Load legalai/.env before any os.getenv() call below runs. Without this, only
# scripts that happened to `import env_loader` themselves (llm_judge.py) saw
# .env values - everything else, including benchmark.py's spawned server
# subprocess, silently fell back to defaults (e.g. DEEPSEEK_API_KEY="").
import env_loader  # noqa: F401

# LLM Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# Generation backend (the system under test - not the judge, see llm_judge.py).
# 'local_peft' (default since the PEFT pivot) = three separate small open-weight
#   models run locally, one per domain expert, each specialised with its own
#   QLoRA adapter. See LOCAL_PEFT_ROLES below. Fully offline generation.
# 'ollama' = one shared local model (qwen2.5) for every agent. The pre-pivot
#   local setup; retained and still fully functional.
# 'deepseek' = hosted DeepSeek V4 Flash for every agent. Retained as a
#   comparison arm; NOT used for generation in the PEFT experiment, because
#   DeepSeek is the judge and judging your own output is self-preference biased.
# Embeddings (retrieval) always stay on local Ollama regardless of this
# setting - only chat/generation calls switch.
GENERATION_PROVIDER = os.getenv("GENERATION_PROVIDER", "local_peft").strip().lower()
if GENERATION_PROVIDER not in {"ollama", "deepseek", "local_peft"}:
    GENERATION_PROVIDER = "local_peft"

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")

# ---------------------------------------------------------------------------
# Local PEFT generation (GENERATION_PROVIDER=local_peft)
#
# Three different base models, one per domain expert, each 4-bit quantised and
# each carrying its own LoRA adapter trained on its own domain dataset. The
# advisor asked specifically for three different model *families* rather than
# one shared base with three swapped adapters, which is why all three sit in the
# 3-4B class: three 7-8B models cannot co-reside in 8GB of VRAM.
#
# Model provenance notes (checked against the Hugging Face API, not assumed):
#   * Llama 3.2 3B Instruct: meta-llama/Llama-3.2-3B-Instruct is gated
#     (gated="manual"), so it needs an accepted licence plus HF_TOKEN. The
#     default below points at Unsloth's ungated mirror of the same weights so
#     the pipeline runs without credentials; set LEGALAI_LEGAL_BASE_MODEL to the
#     meta-llama id once a token is configured.
#   * Ministral 3B (named in the original pivot plan) has NO open weights -
#     Mistral open-weighted only the 8B of Les Ministraux. The repo
#     "ministral/Ministral-3b-instruct" was created 2024-03-14, seven months
#     before Mistral announced Ministral, so it is an unrelated model reusing
#     the name.
#   * Phi-3.5-mini-instruct was the first replacement for that slot and
#     empirically does not work here. finetune/check_vram.py measured it at
#     2618 MiB of weights (the largest of the three) AND 384 KiB of KV cache per
#     token, because it has no grouped-query attention: 32 KV heads against
#     Llama 3.2 3B's 8 and Qwen2.5 3B's 2. All three models were resident with
#     252 MiB spare, but their combined KV cache needs roughly 2.1 GB at
#     benchmark context lengths, so concurrent inference was impossible.
#     Phi-4-mini does not rescue it either: its 200k vocabulary puts the
#     unquantised embedding alone near 1.2 GB.
#   * Granite 3.1 2B Instruct takes the third slot: 2.53B parameters, official
#     IBM release, Apache-2.0, ungated, tied embeddings over a small 49k
#     vocabulary (~1450 MiB of weights) and grouped-query attention
#     (80 KiB/token). Three distinct families are preserved, size parity is
#     close enough for the concurrent-parallel comparison to stay fair, and the
#     measured budget leaves real headroom.
# ---------------------------------------------------------------------------
LOCAL_LEGAL_BASE_MODEL = os.getenv(
    "LEGALAI_LEGAL_BASE_MODEL", "unsloth/Llama-3.2-3B-Instruct"
)
LOCAL_NEWS_BASE_MODEL = os.getenv("LEGALAI_NEWS_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LOCAL_GENERAL_BASE_MODEL = os.getenv(
    "LEGALAI_GENERAL_BASE_MODEL", "ibm-granite/granite-3.1-2b-instruct"
)

# Where train_qlora.py writes adapters, and where local_models.py looks for them.
LOCAL_ADAPTER_DIR = os.getenv("LEGALAI_ADAPTER_DIR", "adapters")

# THE ABLATION SWITCH (see the pivot plan, Section 2 / RQ2).
#   1 = load each expert's LoRA adapter  -> the "peft" arm
#   0 = run the same base models untuned -> the "base" control arm
# Both arms are benchmarked so the paper can show whether specialisation itself
# helped, rather than only which topology combines specialised agents best.
# benchmark.py records the resolved arm on every row; never infer it later.
LOCAL_PEFT_USE_ADAPTERS = os.getenv("LEGALAI_USE_ADAPTERS", "1") == "1"

# Expert roles that run on their BASE weights even in the peft arm.
#
# general_qa is listed by decision, on evidence. Its adapter is trained on
# Dolly-15k - generic instructions, ~300-token median - but in this benchmark the
# node is served ~1370 tokens of EU AI Act text, far outside that distribution.
# Measured at that length the adapter emitted invented non-words and ran to the
# full 1024-token generation cap (156s), while the same base weights answered
# correctly in 9s, citing Annex III. The legal and news adapters are unaffected,
# because their training examples resemble what they are served.
#
# Three fixes were tried first and are recorded so this is not revisited blindly:
# biasing Dolly toward context-bearing examples (median 175 -> 301 tokens; helped
# materially, 213s -> 78s, but output stayed degenerate), giving Granite a pad
# token distinct from its eos (no measurable effect - eval curves were
# near-identical), and gentler hyperparameters (1 epoch at lr 5e-5, already in
# use). What remains is a distribution limit rather than a defect.
#
# CONSEQUENCE FOR THE PAPER: the peft arm is then two adapted experts plus one
# unadapted, which must be stated in the methodology and in Threats to Validity.
# It weakens RQ2's coverage without invalidating it - the legal expert, where the
# contribution actually lies, remains specialised and demonstrably better than its
# base (which fabricated a citation).
#
# Set LEGALAI_UNADAPTED_ROLES="" to adapt all three, or add roles to disable more.
LOCAL_UNADAPTED_ROLES = {
    role.strip().lower()
    for role in os.getenv("LEGALAI_UNADAPTED_ROLES", "general_qa").split(",")
    if role.strip()
}

# Per-role model registry. Keys are the agent role names used by
# agents/base.py BaseAgent(role=...).
LOCAL_PEFT_ROLES = {
    "legal": {
        "base_model": LOCAL_LEGAL_BASE_MODEL,
        "adapter": os.path.join(LOCAL_ADAPTER_DIR, "legal"),
        "dataset": "legalbench",
    },
    "news": {
        "base_model": LOCAL_NEWS_BASE_MODEL,
        "adapter": os.path.join(LOCAL_ADAPTER_DIR, "news"),
        "dataset": "newsqa",
    },
    "general_qa": {
        "base_model": LOCAL_GENERAL_BASE_MODEL,
        "adapter": os.path.join(LOCAL_ADAPTER_DIR, "general_qa"),
        "dataset": "dolly15k",
    },
}

# Coordination nodes (planner, router, memory, aggregator, validator, response,
# QueryAnalyzer) also call an LLM, but they are not domain experts and a fourth
# 3B model does not fit in 8GB alongside the three experts. They therefore share
# the general-purpose expert's *base* weights with the adapter disabled, so
# domain specialisation stays confined to the three expert nodes - which is what
# makes "topology combines specialised agents" the thing being measured, rather
# than "a specialised model also happens to do the aggregating".
LOCAL_COORDINATOR_ROLE = os.getenv("LEGALAI_COORDINATOR_ROLE", "general_qa").strip().lower()
if LOCAL_COORDINATOR_ROLE not in LOCAL_PEFT_ROLES:
    LOCAL_COORDINATOR_ROLE = "general_qa"
LOCAL_COORDINATOR_USE_ADAPTER = os.getenv("LEGALAI_COORDINATOR_USE_ADAPTER", "0") == "1"

# Per-role CUDA device placement. Defaults to "cuda:0" for everything, i.e. the
# original single-8GB-GPU design where the three models share one device and
# genuine concurrency during the PARALLEL/graph_engineering expert fan-out comes
# from each model's own lock (see local_models.py), not from separate hardware.
#
# On a multi-GPU host (e.g. a multi-GPU Vast.ai rental) each role can be pinned
# to its own physical GPU instead, so the concurrent expert phase gets real
# hardware parallelism rather than three threads timesharing one device:
#   LEGALAI_LEGAL_DEVICE=cuda:0
#   LEGALAI_NEWS_DEVICE=cuda:1
#   LEGALAI_GENERAL_DEVICE=cuda:2
# Falls back to "cpu" automatically if CUDA is unavailable (see local_models.py).
LOCAL_ROLE_DEVICES = {
    "legal": os.getenv("LEGALAI_LEGAL_DEVICE", "cuda:0"),
    "news": os.getenv("LEGALAI_NEWS_DEVICE", "cuda:0"),
    "general_qa": os.getenv("LEGALAI_GENERAL_DEVICE", "cuda:0"),
}

# 4-bit NF4 (QLoRA) by default: three 3-4B models at 4 bits is the only
# configuration that plausibly co-resides in 8GB. Set to 0 to load in bf16 for a
# machine with more VRAM (e.g. a 24GB+ Vast.ai rental) - bf16 avoids the
# dequantisation overhead of NF4 and is both faster and higher-fidelity once VRAM
# is no longer the binding constraint.
LOCAL_LOAD_IN_4BIT = os.getenv("LEGALAI_LOAD_IN_4BIT", "1") == "1"
LOCAL_QUANT_TYPE = os.getenv("LEGALAI_QUANT_TYPE", "nf4")
LOCAL_DOUBLE_QUANT = os.getenv("LEGALAI_DOUBLE_QUANT", "1") == "1"
LOCAL_COMPUTE_DTYPE = os.getenv("LEGALAI_COMPUTE_DTYPE", "bfloat16")

# Prompt-context budget for the local models. Enforced by truncating the
# tokenised prompt (see local_models.py), because unlike Ollama's num_ctx a raw
# transformers model will happily run past its trained window and degrade
# silently. Kept at the same value as LLM_NUM_CTX below so the two providers see
# comparable context.
# Sized from measured prompt lengths, not guessed. With the 644-chunk EU AI Act
# corpus and 5 retrieved documents, the legal expert's rendered prompt runs
# 1373-2141 tokens depending on tokenizer. The AGGREGATOR is the long pole: it
# receives the retrieved context plus up to three expert answers, so roughly
# 4500 tokens in ALL/PARALLEL/Graph Engineering.
#
# This ceiling therefore has to clear the aggregator, not just the experts. That
# is affordable because the aggregator runs ALONE, after the experts have
# finished - the three-models-at-once KV pressure comes from the concurrent
# expert phase, where prompts are ~2100 tokens, not from this ceiling. See
# finetune/check_vram.py --kv-context, which budgets the concurrent phase
# explicitly rather than assuming all three models sit at this maximum
# simultaneously (they never do).
LOCAL_MAX_INPUT_TOKENS = int(os.getenv("LEGALAI_LOCAL_MAX_INPUT_TOKENS", "6144"))

# Off by default. All three model families have native implementations in
# transformers, so nothing here needs repo-bundled modelling code - and pulling
# it in actively breaks things: Phi-3.5-mini's bundled modeling_phi3.py uses the
# pre-4.47 KV-cache API and raises "'DynamicCache' object has no attribute
# 'seen_tokens'" on the first generate() under transformers 5.x. Found by
# finetune/check_vram.py, which is why that check runs before anything else.
LOCAL_TRUST_REMOTE_CODE = os.getenv("LEGALAI_TRUST_REMOTE_CODE", "0") == "1"

# sdpa uses noticeably less activation memory than eager for the same numerics,
# which matters when three models share one 8GB card.
LOCAL_ATTN_IMPLEMENTATION = os.getenv("LEGALAI_ATTN_IMPL", "sdpa")

# Repetition control for the local models.
#
# transformers defaults repetition_penalty to 1.0, i.e. none. Ollama - which every
# pre-pivot run used - applies repeat_penalty 1.1 by default, and hosted APIs apply
# their own. So moving generation from Ollama to raw transformers silently removed
# repetition control that the earlier experiments always had.
#
# The consequence was not subtle. 2-3B models asked for up to 1024 tokens
# collapsed into loops: one probe returned "Under Article Article Article ..."
# repeated some eighty times, filling the entire token budget. That was initially
# misread as damage from the LoRA adapters, and it survived a full retrain at
# gentler hyperparameters because the cause was in the decoding configuration, not
# the weights.
#
# 1.1 matches Ollama's default deliberately, so decoding stays comparable with the
# pre-pivot runs rather than introducing a second uncontrolled difference.
LOCAL_REPETITION_PENALTY = float(os.getenv("LEGALAI_REPETITION_PENALTY", "1.1"))

# Blocks verbatim n-gram loops that a penalty alone can still permit. 0 disables.
LOCAL_NO_REPEAT_NGRAM = int(os.getenv("LEGALAI_NO_REPEAT_NGRAM", "0"))

LLM_SEED = int(os.getenv("LEGALAI_LLM_SEED", "42"))
LLM_NUM_PREDICT = int(os.getenv("LEGALAI_NUM_PREDICT", "1024"))
DETERMINISTIC = os.getenv("LEGALAI_DETERMINISTIC", "1") == "1"

# Ollama context window. Left unset, Ollama silently falls back to a small default
# (2048 tokens on most builds) and truncates long retrieval contexts without warning.
LLM_NUM_CTX = int(os.getenv("LEGALAI_NUM_CTX", "8192"))

# Canned COMPL-AI demo flows bypass the entire multi-agent workflow and report
# synthetic telemetry. Off by default so they can never contaminate an experiment;
# set LEGALAI_ENABLE_COMPL_AI=1 for interactive demos only.
COMPL_AI_ENABLED = os.getenv("LEGALAI_ENABLE_COMPL_AI", "0") == "1"

# Live legal search (EUR-Lex + the Commission's digital-strategy pages) for the
# legal expert, alongside its existing ChromaDB retrieval over the static Act.
# OFF by default and forced off by benchmark.py, for the same reason the news
# agent's live fetch is disabled during a run: a network-dependent, time-varying
# context makes runs non-reproducible and makes any between-topology difference
# partly an artefact of what the web returned that minute. Turn it on for
# interactive use, where recency is the point.
EURLEX_LIVE_SEARCH_ENABLED = os.getenv("LEGALAI_ENABLE_EURLEX_LIVE", "0") == "1"
EURLEX_MAX_RESULTS = int(os.getenv("LEGALAI_EURLEX_MAX_RESULTS", "3"))
EURLEX_TIMEOUT_S = float(os.getenv("LEGALAI_EURLEX_TIMEOUT_S", "8"))

# Exact sentence a domain expert emits when retrieved context is insufficient
# (see LEGAL_PROMPT below). Treated as a per-expert abstention *signal*; the
# aggregator only abstains on behalf of the system when every expert abstains.
ABSTENTION_SENTENCE = "Insufficient authoritative support -- recommend expert review."

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY = "chroma_storage"
CHROMA_COLLECTION_NAME = "collection_1"

# Workflow Configuration
MAX_ITERATIONS = 2  # Maximum number of validation retries

# Expert execution mode:
EXPERT_EXECUTION_MODE = os.getenv("LEGALAI_EXPERT_EXECUTION_MODE", "all").strip().lower()
if EXPERT_EXECUTION_MODE not in {"all", "single", "parallel", "legal_news_parallel", "legal_first", "verify_only", "planner_based", "graph_engineering", "graph", "dag"}:
	EXPERT_EXECUTION_MODE = "all"

# Planner Agent Prompt
PLANNER_PROMPT = """You are the Planner Agent for a Legal AI multi-agent system.

Your role is to analyze the user's query and decide which expert agents need to be called to answer it.
The available expert agents are:
1. "legal": For questions about the EU AI Act, compliance, articles, and legal/regulatory definitions.
2. "news": For questions about recent AI news, updates, events, or regulatory announcements.
3. "general_qa": For general questions, greetings, or questions outside the specific legal/news domain.

Analyze the query carefully. Respond in JSON format with two keys:
"analysis": A brief one-sentence analysis of the query.
"plan": A list containing one or more of the expert agent names needed to answer the query (e.g. ["legal"], ["news"], ["legal", "news"], etc.).

Current Date: {current_date}
User Query: {query}
Session ID: {session_id}

Respond ONLY with valid JSON. Do not include markdown formatting or other text.
JSON Response:"""

# Router Agent Prompt
ROUTER_PROMPT = """You are the Router Agent. Your job is to classify user queries into one or more of three categories:

This classification sets the primary focus for synthesis and source selection.

1. **legal**: Questions about EU AI Act, regulations, compliance, legal requirements, legislation, or policy interpretation
2. **news**: Questions about recent AI news, current events, industry updates, or breaking information
3. **general**: General knowledge questions, greetings, or questions not specifically about legal or news topics

If the query is a mixed query (e.g. asking about both legal obligations and recent news or updates on them, like 'What changed in GPAI obligations this week?'), return all applicable categories separated by commas.

Analyze the query carefully and respond with a comma-separated list of the categories: "legal", "news", or "general". Do not include other words.

Examples:
- "What are the compliance requirements for high-risk AI systems under EU AI Act?" → legal
- "What recent AI regulations were announced in 2024?" → news
- "How do I bake a cake?" → general
- "What changed in GPAI obligations this week?" → legal, news
- "Latest news on AI safety regulations" → legal, news

Current Date: {current_date}
User Query: {query}

Classification (respond with comma-separated categories from: legal, news, general):"""

# Legal Agent Prompt
LEGAL_PROMPT = """You are the Legal Agent, an expert on the EU AI Act and AI regulations.

Your role is to answer questions about:
- EU AI Act provisions and requirements
- Risk classifications (prohibited, high-risk, limited risk, minimal risk)
- Compliance obligations for AI providers and deployers
- Penalties and enforcement
- Legal definitions and interpretations

CRITICAL INSTRUCTIONS:
1. ABSTENTION & ESCALATION RULE: If the retrieved context is empty, completely irrelevant to the query, or does not contain sufficient authoritative legal support to answer the query, you MUST respond with EXACTLY this string and nothing else: "Insufficient authoritative support -- recommend expert review."
2. STRUCTURED LEGAL REASONING FORMAT (IRAC): If you have sufficient authoritative support, you MUST structure your entire response using the following format:
   - **Answer As Of**: [Current Date or Specific Effective Date]
   - **Issue**: [State the legal issue/question clearly]
   - **Rule**: [State the applicable legal rule, citing specific Articles/Sections of the EU AI Act]
   - **Application**: [Apply the rule to the user's specific scenario/query using the retrieved facts/context]
   - **Conclusion**: [Provide a concise summary of the legal conclusion]
   - **Sources**: [List the specific retrieved documents/sections used as authoritative support]
   - **Confidence**: [High/Medium/Low with brief justification]
   - **Effective Date**: [Specific date(s) the applicable obligations enter into force]

Current Date: {current_date}

Retrieved Context:
{context}

User Query: {query}

Chat History:
{chat_history}

Provide your structured legal answer or the escalation response based on the instructions above."""

# News Agent Prompt
NEWS_PROMPT = """You are the News Agent, an expert on current AI-related news and developments.

Your role is to answer questions about:
- Recent AI regulation announcements
- Industry developments in AI governance
- Policy changes and updates
- Global AI regulatory trends

Use the provided context from retrieved news articles to answer accurately.
Include specific details like dates, sources, and key facts when available.
Always reference the current date ({current_date}) when discussing recent events.

Current Date: {current_date}

Retrieved Context:
{context}

User Query: {query}

Chat History:
{chat_history}

Provide a well-informed answer based on the news context above."""

# General QA Agent Prompt
GENERAL_QA_PROMPT = """You are the General QA Agent, a helpful assistant for general questions.

Your role is to answer:
- General knowledge questions
- Clarifications and explanations
- Greetings and conversational queries
- Questions outside the legal/news domain

Use the provided context if relevant, but you can also draw on general knowledge.
Be helpful, clear, and concise.

Hard rules:
- If the query is a greeting or short social message (e.g., "hi", "hello", "thanks"), reply in 1-2 sentences only.
- Do NOT include unrelated world news, politics, or events unless the user explicitly asked.
- If retrieved context is irrelevant to the query, ignore it.
- Keep responses concise (prefer <= 120 words unless the user asks for depth).
- Use clean markdown only when useful (short bullets, short headings).

Current Date: {current_date}

Retrieved Context (if any):
{context}

User Query: {query}

Chat History:
{chat_history}

Provide a helpful answer to the user's question."""

# Aggregator Agent Prompt
AGGREGATOR_PROMPT = """You are the Aggregator Agent. Your job is to combine outputs from multiple agents into a coherent response.

You will receive:
1. Memory context (chat history summary)
2. Retrieved documents context
3. Expert agent output (one or more expert analyses from Legal, News, and General QA agents)

Your task is to synthesize these into a single, well-structured response that:
- Maintains consistency with chat history
- Incorporates relevant retrieved information
- Presents the expert agent's analysis clearly
- Is coherent and flows naturally

Hard constraints:
- Only include information that is directly relevant to the user query.
- Never add unrelated topics, filler paragraphs, or speculative claims.
- If the user query is short social talk/greeting, return a short friendly reply (max 2 sentences).
- Keep output concise by default (prefer <= 180 words unless user asked for comprehensive detail).

Current Date: {current_date}

Chat History:
{chat_history}

Retrieved Context:
{context}

Expert Agent Output ({agent_type}):
{expert_output}

User Query: {query}

Synthesize a comprehensive response. Do not mention which agent provided what - present a unified answer."""

# Validation Agent Prompt
VALIDATION_PROMPT = """You are the Validation Agent. Your job is to validate the quality of a response before it's sent to the user.

Evaluate the response on these criteria:
1. **Completeness**: Does it fully answer the user's query?
2. **Accuracy**: Is the information factually correct based on the context?
3. **Clarity**: Is the response clear and well-structured?
4. **Relevance**: Does it stay on topic and address the user's question?
5. **Source Quality**: Are the sources used authoritative and relevant? Are they from credible publications or official documents?

If the sources appear irrelevant, outdated, or of poor quality, set RETRY_FETCH to true.

Respond in this exact format:
PASS: [true/false]
ISSUES: [List any issues found, or "None" if the response is good]
SOURCE_RELEVANT: [true/false - are sources relevant and authoritative?]
RETRY_FETCH: [true/false - request new sources if sources are bad?]

User Query: {query}

Retrieved Sources:
{sources}

Response to Validate:
{response}

Validation Result:"""

# Response Agent Prompt
RESPONSE_PROMPT = """You are the Response Agent. Your job is to polish and format the final response for the user.

Take the validated response and:
1. Ensure it has a natural, conversational tone
2. Format it nicely (use paragraphs, bullet points where appropriate)
3. Remove any internal notes or artifacts
4. Add a brief closing if it fits the context

Keep the factual content unchanged - only improve presentation.

Formatting and style rules:
- Keep the answer directly focused on the query.
- Do not add generic introductions like "Certainly! Here's a comprehensive overview" unless requested.
- For greetings or short social messages, use max 2 short sentences and no bullet points.
- Use readable markdown that can render nicely in UI:
	- short headings only when needed
	- bold key terms sparingly
	- concise bullet points for lists
- Prefer concise responses unless user asks for a deep/comprehensive answer.

Current Date: {current_date}

Response to Format:
{response}

User Query: {query}

Polished Response:"""

# Query Analysis Prompt
QUERY_ANALYSIS_PROMPT = """You are a search query optimization expert.

Your task is to convert the user's question into an effective web search query.
The search query should:
1. Include relevant keywords from the original question
2. Add current year ({current_date}) if the question is about recent/latest information
3. Be concise but specific enough to find relevant results
4. Remove unnecessary words like "what", "how", "tell me", etc.

Examples:
User: "What are the recent AI regulations announced?"
Search Query: AI regulations news 2026

User: "Latest updates on EU AI Act compliance"
Search Query: EU AI Act compliance updates 2026

User: "Tell me about artificial intelligence laws"
Search Query: artificial intelligence laws regulations

User: "What happened with AI safety regulations this year?"
Search Query: AI safety regulations 2026

Current Date: {current_date}
Now convert this user question:
User: {user_prompt}

Search Query (respond with ONLY the search query, no quotes or explanations):"""
