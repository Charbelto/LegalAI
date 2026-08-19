# Running the full benchmark on Vast.ai

This is the practical guide for renting a GPU on [Vast.ai](https://vast.ai) and
running the 540-run PEFT-vs-base / topology benchmark there instead of on an
8GB laptop GPU.

## 1. Which GPU to rent

The system under test is three separately fine-tuned **2-3B parameter** models
(Llama 3.2 3B, Qwen2.5 3B, Granite 3.1 2B), not one large model. That changes
the sizing question from "does it fit" (the constraint the code was originally
built around, see `finetune/check_vram.py`) to "how fast can it decode", because
single-request decoding of a small model is memory-bandwidth-bound, not
compute-bound - extra tensor cores beyond a certain point buy little.

| GPU | VRAM | Why | Est. cost on Vast.ai |
|---|---|---|---|
| **RTX 4090 (recommended)** | 24 GB | ~3x the bf16 footprint of all three models combined (~18 GB), so `LEGALAI_LOAD_IN_4BIT=0` (bf16, no quant overhead) fits comfortably with room for KV caches and concurrent generation. Ada-generation memory bandwidth (~1 TB/s) is close to a datacenter A100 for small-batch decode, at a fraction of the hourly rate. Widely available on Vast.ai. | ~$0.20-0.45/hr |
| A100 40GB PCIe/SXM | 40 GB | More bandwidth (~1.9 TB/s) and headroom if you want to raise context length or add a 4th coordinator-only model; diminishing returns for single-stream small-model decode relative to cost. Use if 4090 availability is poor or you want extra margin. | ~$0.80-1.50/hr |
| L40S | 48 GB | Similar reasoning to A100, Ada-generation, good if you also want to keep the local Ollama embedding model resident on GPU without any contention. | ~$0.60-1.00/hr |

**Recommendation: rent one RTX 4090 (24GB).** It is the cost-effective choice
for this workload. Also request:
- **8+ vCPUs, 32+ GB system RAM** (ChromaDB, tokenization, the FastAPI server, and Ollama's embedding model all run on CPU/host RAM alongside the GPU work).
- **80+ GB disk** (base model weights + adapters + Chroma store + judge cache + benchmark logs add up).
- A template with **CUDA 12.4+** pre-installed (Vast.ai's "PyTorch (cuDNN Runtime)" template, or plain Ubuntu 22.04 if you'd rather install everything yourself).

Once VRAM is no longer the binding constraint, set `LEGALAI_LOAD_IN_4BIT=0` in
`.env` so the three experts load in bf16 instead of 4-bit NF4 - this removes
the bitsandbytes dequantisation overhead on every forward pass and is both
faster and higher-fidelity than the laptop's 4-bit configuration.

### Using the spare VRAM: replica pools + benchmark concurrency

A 24GB card holds all three models (bf16, ~18GB) with room to spare, which
this codebase can now use for genuine intra-role concurrency instead of
sitting idle. Two independent knobs, both safe to combine:

**Tier 1 - free, zero extra VRAM.** Raise `benchmark.py --concurrency` (or
`CONCURRENCY=2` for `run_experiment.sh`) to 2 or 3 even with every pool size
left at 1. Legal and News are already different loaded models with their own
lock, so two concurrent benchmark requests that land on Legal and News at the
same time now overlap for real - each gets its own CUDA stream (see
`local_models.py`'s `_LoadedModel.stream`), so their kernels genuinely
interleave on the GPU instead of implicitly serialising on one shared default
stream. This is the same mechanism that already lets `parallel`/
`graph_engineering` fan Legal and News out *within* one request; raising
`--concurrency` extends it *across* requests too, for free.

**Tier 2 - costs VRAM, targets the real bottleneck.** `general_qa` is the
busiest role by a wide margin: besides the General QA expert itself, every
coordination node (planner, router, aggregator, validator, response) also
resolves to it (see `resolve_role()` / `LOCAL_COORDINATOR_ROLE`), so it is
what most concurrent requests end up queueing behind. `LEGALAI_GENERAL_POOL_SIZE=2`
loads a second independent copy of Granite (~5GB more) so two of those calls
can run at once instead of one waiting on the other's lock:

```bash
export LEGALAI_GENERAL_POOL_SIZE=2   # ~5GB extra VRAM; legal/news stay at 1
```

VRAM math to check before raising further (bf16, per replica): Legal
(Llama 3.2 3B) ~6.4GB, News (Qwen2.5 3B) ~6.2GB, General QA (Granite 3.1 2B)
~5.1GB; KV cache per replica at benchmark context lengths is only a few hundred
MB (see `finetune/check_vram.py`'s KV budget), so weights dominate the total.
`legal=1, news=1, general_qa=2` is ~23GB of weights alone on a 24GB card -
workable but with little headroom for activations/CUDA context, so **run
`finetune/check_vram.py --concurrent` after changing a pool size and confirm
it reports `PASS` before trusting it**, rather than assuming the arithmetic
above holds exactly on your specific rental. If it doesn't fit, either drop
back to `general_qa=1` (raising benchmark `--concurrency` alone still helps
Legal/News) or rent a 40GB+ card.

**What this costs you:** recorded `elapsed_s` at concurrency > 1 includes
queueing/contention time, not just the topology's own work, so it is no longer
an isolated per-request latency measurement. Apply the *same* concurrency
level to every topology and both arms in one benchmark pass (which
`run_experiment.sh`/`.ps1` already do - concurrency is a single flag for the
whole run) so the relative latency comparison between ALL/PARALLEL/
graph_engineering - what the paper's significance tests actually use - stays
valid, and disclose the concurrency level used alongside any absolute latency
figure in the paper.

Multi-GPU device pinning (below) is the other lever if you rent more than one
GPU: it gives Legal/News/General QA their own physical device each, so the
concurrent expert phase gets real hardware parallelism instead of sharing one
device's compute via streams.

### Optional: multi-GPU device pinning

If you rent a multi-GPU instance, each expert can be pinned to its own physical
device so the concurrent expert phase (`parallel`, `graph_engineering`) gets
real hardware parallelism instead of three threads timesharing one GPU:

```bash
export LEGALAI_LEGAL_DEVICE=cuda:0
export LEGALAI_NEWS_DEVICE=cuda:1
export LEGALAI_GENERAL_DEVICE=cuda:2   # falls back to cuda:0 if only 2 GPUs exist
```

Leave these unset (they default to `cuda:0`) on a single-GPU rental - three
small models comfortably share one 24GB card.

## 2. Rent and connect

1. Create a Vast.ai account and add credit.
2. Search: GPU = RTX 4090, Disk >= 80 GB, a CUDA 12.4+/PyTorch template.
3. Rent the instance, then connect with the SSH command Vast.ai gives you on the instance's page:
   ```bash
   ssh -p <port> root@<host>
   ```

## 3. Set up the box

```bash
apt-get update && apt-get install -y git curl python3.11 python3.11-venv

git clone https://github.com/Charbelto/LegalAI.git
cd LegalAI/legalai

python3.11 -m venv .venv
source .venv/bin/activate

# torch FIRST, from PyTorch's own CUDA index, or pip silently grabs a CPU wheel
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -r requirements-finetune.txt

# Ollama, for embeddings only (generation is local_peft, not Ollama)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull nomic-embed-text
```

Create `.env` (copy `.env.example` if present, otherwise create it) with at least:

```bash
GENERATION_PROVIDER=local_peft
LEGALAI_LOAD_IN_4BIT=0          # bf16 - see the GPU sizing note above
LEGALAI_GENERAL_POOL_SIZE=2     # optional Tier 2 - only after check_vram.py --concurrent confirms it fits
DEEPSEEK_API_KEY=sk-...         # for the LLM judge only; never used for generation
JUDGE_PROVIDER=deepseek
JUDGE_MODEL=deepseek-v4-flash
```

The three LoRA adapters (`adapters/legal`, `adapters/news`, `adapters/general_qa`)
are already committed to the repo, so no fine-tuning step is needed unless you
are retraining them.

## 4. Validate before spending GPU-hours on a full run

```bash
python finetune/check_vram.py --concurrent    # confirms residency + KV budget on THIS GPU
python -m pytest tests -q                     # 49 tests, ~20s, no GPU needed
```

## 5. Smoke test, then the full run

```bash
chmod +x run_experiment.sh
./run_experiment.sh --smoke --yes             # 1 query x 3 modes x 1 repeat per arm
```

Time the smoke run and extrapolate before committing to the full run:

```bash
python -c "import json,statistics; rows=[json.loads(l) for l in open('benchmark_runs_smoke.jsonl')]; times=[r['elapsed_s'] for r in rows if r.get('success')]; print(f'{statistics.mean(times):.1f}s avg over {len(times)} runs -> full 540-run estimate: {statistics.mean(times)*540/3600:.1f} hours')"
```

Inspect `benchmark_runs_smoke.jsonl` per the checklist in `run_experiment.ps1`
(no abstention-sentence-everywhere, peft/base answers actually differ, modes
differ from each other within an arm). If it looks right, run the full
benchmark - `CONCURRENCY` applies the Tier 1/2 settings above uniformly across
the whole run:

```bash
CONCURRENCY=2 ./run_experiment.sh --yes       # full 540-run benchmark + analysis + figures
```

Rough expectation: bf16 + per-replica CUDA streams alone (no concurrency
change) should already beat the 8-13 hour 4-bit/single-stream estimate;
`CONCURRENCY=2`-`3` on top of that is the difference between "several hours"
and "closer to one" - but treat both as directional. The smoke-run
extrapolation above, measured on your actual rental, is the number to trust.

`--yes` skips the interactive confirmations (judge-preflight cost check,
DeepSeek self-judging warning) so the run can proceed unattended over SSH -
review those warnings in the log afterward rather than blindly trusting them
on a first run.

## 6. Get the results back

The run produces `benchmark_runs.jsonl`, `run_meta*.json`, `analysis_summary.csv`,
`significance.csv`, `by_query_type.csv`, `metrics_table.tex`,
`metrics_table_ablation.tex`, and `paper_figures/*.png`. Either:

```bash
# from your local machine
scp -P <port> -r root@<host>:~/LegalAI/legalai/paper_figures ./paper_figures
scp -P <port> root@<host>:~/LegalAI/legalai/{benchmark_runs.jsonl,analysis_summary.csv,significance.csv,by_query_type.csv,metrics_table.tex,metrics_table_ablation.tex} ./
```

or commit them from the instance and push (see the main README for the
git workflow) - either way, **destroy the Vast.ai instance once the run
finishes** so it stops billing.

## 7. Before trusting the numbers

`graph/workflow.py` was changed so the terminal validator (Loop Engineering) now
runs **only** for the `graph_engineering` topology - `all` and `parallel`
terminate immediately after aggregation. Any `metrics_table.tex` /
`metrics_table_ablation.tex` produced by a run against the current code
reflects that; numbers already in the repo from before this change do not
(they show identical 20-step graphs for every topology, which was the bug).

Also note for the write-up: `local_models.py` now seeds each generation call
with its own `torch.Generator` instead of the old `torch.manual_seed()` /
`torch.cuda.manual_seed_all()` pair, which reseeded PyTorch's global RNG and
could be silently clobbered by a concurrently-running call (already possible
pre-pooling, since Legal and News already ran concurrently in `parallel`/
`graph_engineering`, and more likely now with replica pools and raised
benchmark concurrency). If you ever re-verify "same seed -> identical output"
as part of validating a run, that check is now actually sound under
concurrency; it was a latent gap before.
