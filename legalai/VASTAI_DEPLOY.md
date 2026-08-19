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
faster and higher-fidelity than the laptop's 4-bit configuration. (There's
more you can do with the spare VRAM - replica pools, concurrency - covered in
step 3 below, after setup.)

## 2. Rent and connect

1. Create a Vast.ai account and add credit.
2. Search: GPU = RTX 4090, Disk >= 80 GB, a CUDA 12.4+/PyTorch template.
3. Rent the instance, then connect with the SSH command Vast.ai gives you on the instance's page:
   ```bash
   ssh -p <port> root@<host>
   ```

## 3. Get the code, then three commands

```bash
apt-get update && apt-get install -y git python3 python3-pip
git clone https://github.com/Charbelto/LegalAI.git
cd LegalAI/legalai
```

(If you picked a "PyTorch"/CUDA template, Python and pip are already there and
this line is a harmless no-op. If you picked a bare Ubuntu image, this is what
gets you both.)

That's the only "infrastructure" step. Everything else is three Python
commands, run in this order:

```bash
python setup.py    # once: installs torch/deps, sets up Ollama, creates .env, runs the tests
python test.py      # a few minutes: sanity check + a time estimate for the full run
python run.py       # the full 540-run benchmark, analysis, and figures
```

`setup.py` installs the CUDA build of torch and everything in
`requirements.txt`/`requirements-finetune.txt`, installs and starts Ollama
(needed for embeddings - see the note below), and copies `.env.example` to
`.env` if you don't have one yet - **also writing `LEGALAI_LOAD_IN_4BIT=0`
and `LEGALAI_GENERAL_POOL_SIZE=2`**, since `test.py`/`run.py` both default to
`--concurrency 4` (see "Using the spare VRAM" below for what that pairing
means and its VRAM trade-off). **After it finishes, open `.env` and set
`DEEPSEEK_API_KEY`** (used only by the LLM judge that scores answers after
generation, never for generation itself). The three LoRA adapters
(`adapters/legal`, `adapters/news`, `adapters/general_qa`) are already
committed to the repo, so no fine-tuning step is needed.

`test.py` runs the unit tests, then a tiny 6-request smoke benchmark **at the
same `--concurrency 4` default `run.py` will use**, and prints an estimate
for how long the full run will take **on your actual rental** - trust that
number over anything estimated in this doc. It also flags the obvious red
flags (abstention sentence everywhere, peft/base answers being identical,
etc.) - fix those before moving on. Both scripts run a VRAM pre-flight check
automatically before touching concurrency > 1 (see below) - it fails in
seconds with a clear message if the pool doesn't fit, instead of failing
hours into a run.

`run.py` runs both experimental arms, then the statistics/judge/charts
pipeline, and leaves `metrics_table.tex`, `metrics_table_ablation.tex`, and
`paper_figures/*.png` ready to copy out. It can take hours - run it inside
`tmux` or `screen` so it survives your SSH connection dropping:

```bash
tmux new -s run
python run.py
# Ctrl+B then D to detach; `tmux attach -t run` later to check on it
```

**On Ollama specifically:** yes, it installs fine on Vast.ai - it's a normal
Linux box with internet access. `setup.py`/`test.py`/`run.py` all call a
shared helper (`ollama_setup.py`) that installs it if missing, starts it if
it's not running, and pulls the embedding model - you shouldn't need to touch
it directly.

### Using the spare VRAM (this is now the default - read this before your first real run)

A 24GB card holds all three models (bf16, ~18GB) with room to spare, so
`test.py`/`run.py` default to **`--concurrency 4` paired with
`LEGALAI_GENERAL_POOL_SIZE=2`** (`setup.py` writes that setting into `.env`
for you). What each half does:

**Concurrency 4 - free, just runs more requests at once.** Legal and News are
already different models with their own GPU stream, so requests that need
different experts at the same time genuinely overlap instead of queueing.

**`LEGALAI_GENERAL_POOL_SIZE=2` - costs ~5GB, targets the real bottleneck.**
`general_qa` is the busiest role: besides the General QA expert, every
coordination node (planner, router, aggregator, validator, response) shares
its weights, so it's what concurrent requests queue behind most. This loads a
second copy of Granite so two of those calls can run at once.

**Be honest about the VRAM math, though: this pairing is tight, not
comfortable.** Per replica in bf16: Legal ~6.4GB, News ~6.2GB, General QA
~5.1GB. `legal=1, news=1, general_qa=2` is already **~23GB of weights alone
on a 24GB card**, before KV caches, activations, or CUDA overhead. It's
plausible this doesn't fit on your specific rental - that's exactly why
`test.py`/`run.py` both run `finetune/check_vram.py --concurrent` automatically
before doing anything expensive, and refuse to proceed with a clear message
if it doesn't report a fit, rather than letting you find out three hours into
`run.py`. If it fails:

```bash
# in .env, back off the pool:
LEGALAI_GENERAL_POOL_SIZE=1
# or run at the safe default:
python run.py --concurrency 1
# or rent a 40GB+ card and keep general_qa=2
```

`--skip-vram-check` exists on both scripts if you've already confirmed it
fits and don't want to re-check every time, but there's little reason to use
it - the check takes well under a minute.

**The trade-off:** once you raise concurrency above 1, the recorded latency
for each run includes queueing time, not just the topology's own work - so
it's no longer an isolated per-request measurement. This is still fine for
the paper as long as the same concurrency level applies to every topology and
arm in one run (which it does, since it's one setting for the whole
`run.py` pass) - the relative comparison between ALL/PARALLEL/graph_engineering
stays valid, only the absolute latency numbers reflect "under this much
concurrency" rather than "in isolation." Mention the concurrency level used
alongside any latency figure.

Multi-GPU device pinning is the other lever if you rent more than one GPU -
gives Legal/News/General QA their own physical device each:

```bash
export LEGALAI_LEGAL_DEVICE=cuda:0
export LEGALAI_NEWS_DEVICE=cuda:1
export LEGALAI_GENERAL_DEVICE=cuda:2   # falls back to cuda:0 if only 2 GPUs exist
```

Leave these unset on a single-GPU rental.

## 4. Get the results back

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

## 5. Before trusting the numbers

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
