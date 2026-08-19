# Legal AI

Legal AI is a thesis-grade multi-agent RAG system focused on:

- EU AI Act legal/compliance analysis
- Current AI governance news retrieval
- FastAPI streaming backend + React frontend
- Chroma vector retrieval with Ollama models

This repository now includes a full end-to-end workflow: development, production mode, session persistence, deployment artifacts, and operational diagnostics.

## Core capabilities

- Multi-agent orchestration (planner, router, retrieval, sequential legal/news/general experts, aggregator, validator, response formatter)
- SSE chat streaming with live workflow traces
- Automatic query-aware source fetching and embedding
- Persistent conversation sessions saved to local JSON files
- Frontend session manager with load/delete/clear controls
- Chat utility controls:
	- retry prompt
	- regenerate answer
	- copy last answer
	- export JSON/Markdown
	- import previous transcript
	- request cancellation button
- Runtime and readiness diagnostics endpoints
- Production mode where backend serves built frontend assets

## Architecture

- backend/main.py: FastAPI application routes and SSE streaming endpoint
- backend/service.py: orchestration service, readiness checks, session APIs, workflow execution
- backend/session_store.py: persistent session storage and retrieval
- graph/workflow.py + agents/: LangGraph workflow and specialized agents
- frontend/: Vite React UI with operational dashboard controls
- app.py: one-command launcher for dev and production modes

## Prerequisites

- Python 3.10+
- Node.js 18+
- Ollama running and reachable
- Required Ollama models pulled:
	- qwen2.5 (chat)
	- nomic-embed-text (embedding)

Example:

```bash
ollama pull qwen2.5
ollama pull nomic-embed-text
```

## Quick start (development)

From the legalai folder:

```bash
pip install -r requirements.txt
python app.py
```

Starts:

- Frontend: http://127.0.0.1:5173
- Backend: http://127.0.0.1:8000

Useful launcher flags:

- python app.py --backend-only
- python app.py --frontend-only
- python app.py --no-reload
- python app.py --backend-port 8001 --frontend-port 5174
- python app.py --model qwen2.5:3b
- python app.py --embedding-model nomic-embed-text

## Production mode (single host)

Run backend serving built frontend static assets:

```bash
python app.py --production --build-frontend --no-reload --workers 2
```

Then open:

- http://127.0.0.1:8000

Production-mode notes:

- LEGALAI_SERVE_STATIC is set automatically by launcher in production mode
- frontend/dist must exist (launcher builds it when --build-frontend is set)

## Environment variables

Copy from .env.example and adjust:

- OLLAMA_BASE_URL
- OLLAMA_MODEL
- OLLAMA_EMBEDDING_MODEL
- LEGALAI_EXPERT_EXECUTION_MODE (all|single, default: all)
- LEGALAI_ALLOWED_ORIGINS
- LEGALAI_SERVE_STATIC

Expert mode notes:

- all: runs legal, news, and general QA experts sequentially, then aggregates
- single: runs only the router-selected expert before aggregation (lower latency)

Per-request override:

- The chat API accepts expert_execution_mode in request payload (all|single)
- The frontend Fetch Settings panel exposes this as a manual switch

Frontend env example is in frontend/.env.example:

- VITE_API_BASE_URL=http://127.0.0.1:8000

## API endpoints

Core:

- GET /
- GET /health
- GET /runtime
- GET /readiness
- GET /sources?limit=10
- POST /chat
- POST /chat/stream
- POST /admin/clear

Timing diagnostics:

- /runtime includes expert_execution_mode
- /chat and final /chat/stream payload include:
	- agent_timings_ms (per workflow node)
	- workflow_elapsed_ms (end-to-end graph runtime)

Session persistence:

- GET /sessions?limit=20
- GET /sessions/{session_id}
- DELETE /sessions/{session_id}
- DELETE /sessions

Interactive docs:

- GET /docs

## Deployment with Docker

### Option A: Single container (backend + built frontend)

```bash
docker build -t legalai:latest .
docker run -p 8000:8000 \
	-e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
	-e OLLAMA_MODEL=qwen2.5 \
	-e OLLAMA_EMBEDDING_MODEL=nomic-embed-text \
	legalai:latest
```

### Option B: Docker Compose

```bash
docker compose up --build
```

Persisted volumes:

- ./articles -> /app/articles
- ./chroma_storage -> /app/chroma_storage
- ./sessions -> /app/sessions
- ./articles.json -> /app/articles.json

## Operational caveats

1. First run latency: If vector storage is empty, EU AI Act ingestion/embedding happens lazily on first request.
2. Ollama dependency: If Ollama is offline or models are missing, readiness becomes degraded and chat can fail.
3. News scraping variability: Some sites block scraping or return thin HTML; fetch results are best-effort.
4. Network constraints: Corporate proxies/firewalls can block model pulls or source fetching.
5. Disk locks on Windows: Data clear operations use retry logic to handle transient file locking.
6. Large prompts: Message length is validated (max 8000 chars) to prevent runaway payloads.

## Thesis completeness checklist

- Multi-agent workflow present and instrumented
- Retrieval + source ingestion pipeline present
- Streaming and non-streaming chat endpoints present
- Session persistence and management present
- Operational health/readiness/runtime diagnostics present
- Frontend controls for export/import/trace/session/source management present
- Production launcher path present
- Container deployment path present

## Suggested next additions (optional)

1. Add automated tests for service/session endpoints.
2. Add authentication/authorization before public deployment.
3. Add citation confidence scoring and source ranking analytics.

## Coordination-Topology Experiment (PEFT-specialised experts)

This repository contains a full experiment framework comparing **three
coordination topologies** — `all` (sequential), `parallel` (concurrent) and
`graph_engineering` (converging dependency with terminal loop engineering verification) — over **three separately fine-tuned expert models**:

| role | base model | LoRA adapter trained on |
|---|---|---|
| legal | Llama 3.2 3B Instruct | LegalBench |
| news | Qwen2.5 3B Instruct | NewsQA |
| general_qa | Granite 3.1 2B Instruct | Dolly-15k |

Generation is fully local (`GENERATION_PROVIDER=local_peft`): all three models are
4-bit quantised and co-resident on one 8 GB GPU. Every topology is run twice — once
with the adapters loaded (`peft`) and once on the identical untuned base models
(`base`) — so the effect of specialisation can be separated from the effect of
structure. That is 30 queries x 3 topologies x 3 repeats x 2 arms = **540 runs**.

The other topologies (`single`, `legal_first`, `planner_based`, `verify_only`,
`legal_news_parallel`) remain fully implemented and selectable through the API and
UI, but are excluded from the benchmark and the paper.

**To run the full benchmark on a rented cloud GPU (recommended over a laptop
8GB GPU for turnaround time), see [`VASTAI_DEPLOY.md`](VASTAI_DEPLOY.md)** —
GPU sizing guidance, then three commands: `python setup.py` (once),
`python test.py` (a few minutes, sanity check + a timing estimate),
`python run.py` (the full run). On Windows, `run_experiment.ps1` runs the
same pipeline with more manual control (arm selection, judge provider
overrides, etc.); `run_experiment.sh` is its Linux equivalent. Local 3B
generation on an 8GB laptop GPU is far slower than a hosted API — budget on
the order of a day for the full 540 runs there.

### Fine-tuning pipeline

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # CUDA build first
pip install -r requirements-finetune.txt

python finetune/check_vram.py --concurrent      # can all three models co-reside?
python finetune/prepare_datasets.py             # build the three SFT sets
python finetune/train_qlora.py --role all       # one QLoRA adapter per expert
python finetune/validate_adapters.py            # confirm the adapters changed behaviour
```

`check_vram.py` is the gating check and reports both weight residency and KV-cache
budget. The KV cache is the binding constraint: a model without grouped-query
attention can be resident and still make concurrent inference impossible.

### Stable Document ID Scheme
Each document retrieved from the vector database is assigned a stable ID in `doc.metadata["id"]`:
- **ChromaDB Chunk IDs**: Chunks derived from the initial vector store ingest are assigned their native database index IDs (e.g. `"0"`, `"1"`, ... `"N"`).
- **Fallback / Ingested Chunks**: For new news articles or chunks that do not map to the database indices, a stable fallback ID is generated using the formula: `"{document_name}_{content_hash}"`, where `content_hash` is the first 8 hex characters of the MD5 hash of the chunk's content.

### Local Execution Instructions
To run the experiment and generate analysis:

```bash
# 1. Setup virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies (includes scipy, statsmodels, pytest, pandas, matplotlib, seaborn)
pip install -r requirements.txt

# 3. Local generation + fine-tuning stack (see requirements-finetune.txt)
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-finetune.txt

# 4. Ollama is still needed for EMBEDDINGS even though generation is local
ollama pull nomic-embed-text

# 5. Ingest the EU AI Act (Official Journal text, CELEX:32024R1689)
python embed.py --replace

# 6. Generate gold answers (N >= 30 queries)
python scripts/build_gold.py

# 7. Train the three expert adapters (see the fine-tuning pipeline above)
python finetune/prepare_datasets.py
python finetune/train_qlora.py --role all

# 8. Run both arms, 3 topologies, 3 repeats (540 runs). Use -Smoke first.
./run_experiment.ps1 -Smoke
./run_experiment.ps1 -BenchmarkOnly

# 9. Aggregate results and run statistical tests
python analyze_results.py

# 10. Generate publication-grade plots and LaTeX tables
python evaluate_workflows.py
python make_paper_figures.py
```

