#!/usr/bin/env bash
# =============================================================================
# One-command clean experiment run (Linux/Vast.ai port of run_experiment.ps1).
#
#   ./run_experiment.sh                       full run: 30 queries x 3 topologies
#                                              x 3 repeats x 2 arms = 540 runs
#   ARMS=peft ./run_experiment.sh             PEFT arm only (270 runs, drops RQ2)
#   ./run_experiment.sh --smoke               3 runs per arm, separate output files
#   ./run_experiment.sh --skip-benchmark      re-analyse the existing runs file only
#   GENERATION_PROVIDER=ollama ./run_experiment.sh   pre-pivot shared-model arm
#
# Kept behaviourally identical to run_experiment.ps1 - see that file for the
# reasoning behind each guard. This script exists because Vast.ai instances are
# Linux and PowerShell is not available there.
#
# Flags: --smoke --skip-benchmark --benchmark-only --resume --yes
# Env:   JUDGE_PROVIDER JUDGE_MODEL GENERATION_PROVIDER ARMS(both|peft|base)
#        REPEATS(3) CONCURRENCY(0=auto)
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SMOKE=0
SKIP_BENCHMARK=0
BENCHMARK_ONLY=0
RESUME=0
ASSUME_YES=0
REPEATS="${REPEATS:-3}"
CONCURRENCY="${CONCURRENCY:-0}"
ARMS="${ARMS:-both}"

for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    --skip-benchmark) SKIP_BENCHMARK=1 ;;
    --benchmark-only) BENCHMARK_ONLY=1 ;;
    --resume) RESUME=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    *) echo "Unknown flag: $arg" >&2; exit 1 ;;
  esac
done

confirm() {
  # Non-interactive by default on a cloud box: --yes (or a non-tty stdin, e.g. a
  # detached/nohup run) proceeds automatically rather than blocking forever on a
  # prompt nobody can answer.
  local prompt="$1"
  if [[ "$ASSUME_YES" == "1" || ! -t 0 ]]; then
    echo "$prompt (auto-yes: --yes or non-interactive shell)"
    return 0
  fi
  read -r -p "$prompt (y/N) " answer
  [[ "$answer" == "y" || "$answer" == "Y" ]]
}

echo "=== Legal AI experiment run ==="

# --- Interpreter -------------------------------------------------------------
PY="./.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "WARNING: no venv at $PY; falling back to 'python3' on PATH." >&2
  echo "WARNING: if the local PEFT stack is installed in a venv, this run will fail to load models." >&2
  PY="python3"
fi
echo "interpreter: $PY"

EFFECTIVE_PROVIDER_CHECK="${GENERATION_PROVIDER:-}"
if [[ "$EFFECTIVE_PROVIDER_CHECK" != "ollama" && "$EFFECTIVE_PROVIDER_CHECK" != "deepseek" ]]; then
  MISSING=$("$PY" -c "import importlib.util as u; print(','.join(m for m in ('torch','transformers','peft','accelerate','bitsandbytes') if u.find_spec(m) is None))")
  if [[ -n "$MISSING" ]]; then
    echo "ERROR: interpreter '$PY' is missing required packages for local generation: $MISSING" >&2
    echo "Install them:" >&2
    echo "    $PY -m pip install torch --index-url https://download.pytorch.org/whl/cu126" >&2
    echo "    $PY -m pip install -r requirements-finetune.txt" >&2
    exit 1
  fi
  echo "      local generation stack present (torch, transformers, peft, accelerate, bitsandbytes)"
fi

# --- Environment ---------------------------------------------------------------
export LEGALAI_DETERMINISTIC=0
export LEGALAI_NUM_CTX=8192
export LEGALAI_ENABLE_COMPL_AI=0
export LEGALAI_ENABLE_EURLEX_LIVE=0
export BENCH_REPEATS="$REPEATS"

if [[ -z "${GENERATION_PROVIDER:-}" && -f .env ]]; then
  GENERATION_PROVIDER="$(grep -E '^\s*GENERATION_PROVIDER\s*=' .env | head -1 | cut -d= -f2- | tr -d ' "'"'"'')"
fi
GENERATION_PROVIDER="${GENERATION_PROVIDER:-local_peft}"
export GENERATION_PROVIDER

: "${OLLAMA_NUM_PARALLEL:=4}"
export OLLAMA_NUM_PARALLEL

EFFECTIVE_CONCURRENCY="$CONCURRENCY"
if [[ "$EFFECTIVE_CONCURRENCY" -le 0 ]]; then
  if [[ "$GENERATION_PROVIDER" == "deepseek" ]]; then EFFECTIVE_CONCURRENCY=8; else EFFECTIVE_CONCURRENCY=1; fi
fi
echo "generation_provider=$GENERATION_PROVIDER deterministic=$LEGALAI_DETERMINISTIC num_ctx=$LEGALAI_NUM_CTX OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL concurrency=$EFFECTIVE_CONCURRENCY"

if [[ "$GENERATION_PROVIDER" != "local_peft" ]]; then
  echo "      (generation_provider=$GENERATION_PROVIDER has no PEFT adapters; single pass, arm label 'peft' is not applicable)"
  ARM_LIST=(peft)
elif [[ "$ARMS" == "both" ]]; then
  ARM_LIST=(peft base)
else
  ARM_LIST=("$ARMS")
fi
echo "arms=${ARM_LIST[*]} repeats=$REPEATS"

if [[ "$GENERATION_PROVIDER" == "deepseek" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
  if ! confirm "GENERATION_PROVIDER=deepseek but DEEPSEEK_API_KEY is not set. Continue anyway?"; then exit 1; fi
fi
if [[ "$GENERATION_PROVIDER" == "deepseek" && ( -z "${JUDGE_PROVIDER:-}" || "${JUDGE_PROVIDER:-}" == "deepseek" ) ]]; then
  if ! confirm "GENERATION_PROVIDER=deepseek with a DeepSeek judge means the judge grades its own output. Continue anyway?"; then exit 1; fi
fi

if [[ "$GENERATION_PROVIDER" == "local_peft" && " ${ARM_LIST[*]} " == *" peft "* && "$SKIP_BENCHMARK" == "0" ]]; then
  MISSING_ROLES=()
  for role in legal news general_qa; do
    [[ -f "adapters/$role/adapter_config.json" ]] || MISSING_ROLES+=("$role")
  done
  if [[ ${#MISSING_ROLES[@]} -gt 0 ]]; then
    echo "ERROR: no LoRA adapter for: ${MISSING_ROLES[*]}. Train them first or run ARMS=base." >&2
    exit 1
  fi
  echo "      adapters present for legal, news, general_qa"
fi

# --- Preconditions -------------------------------------------------------------
echo
echo "[1/6] Checking Ollama..."
if ! curl -sf http://localhost:11434/api/tags >/dev/null; then
  echo "ERROR: Ollama is not reachable at localhost:11434. Start it first (see VASTAI_DEPLOY.md)." >&2
  exit 1
fi
echo "      Ollama reachable"

echo
echo "[2/6] Running validity tests..."
"$PY" -m pytest tests -q

# --- Benchmark -------------------------------------------------------------
if [[ "$SKIP_BENCHMARK" == "0" ]]; then
  echo
  echo "[3/6] Benchmark (${#ARM_LIST[@]} arm(s))..."
  STARTED=$(date +%s)
  ARM_INDEX=0
  for arm in "${ARM_LIST[@]}"; do
    echo
    echo "      --- arm: $arm ---"
    BENCH_ARGS=(--arm "$arm")
    [[ "$SMOKE" == "1" ]] && BENCH_ARGS+=(--smoke)
    [[ "$RESUME" == "1" ]] && BENCH_ARGS+=(--resume)
    if [[ "$ARM_INDEX" -gt 0 && "$RESUME" == "0" ]]; then BENCH_ARGS+=(--append); fi
    if [[ "$EFFECTIVE_CONCURRENCY" -gt 1 ]]; then BENCH_ARGS+=(--concurrency "$EFFECTIVE_CONCURRENCY"); fi

    "$PY" -u benchmark.py "${BENCH_ARGS[@]}"
    ARM_INDEX=$((ARM_INDEX + 1))
  done
  echo "      benchmark took $(($(date +%s) - STARTED))s in total"
else
  echo
  echo "[3/6] Benchmark skipped (--skip-benchmark)"
fi

if [[ "$SMOKE" == "1" ]]; then
  echo
  echo "Smoke run complete -> benchmark_runs_smoke.jsonl"
  echo "Inspect the answers before committing to the full run, then re-run without --smoke."
  exit 0
fi

if [[ "$BENCHMARK_ONLY" == "1" ]]; then
  echo
  echo "Answers generated. Analysis skipped (--benchmark-only)."
  echo "Review the gold answers, then run: ./run_experiment.sh --skip-benchmark"
  exit 0
fi

# --- Judge preflight -----------------------------------------------------------
echo
echo "[3.5/6] Judge preflight..."
"$PY" llm_judge.py --check
if ! confirm "Proceed with the full judging pass?"; then
  echo "Stopped before judging. Re-run with --skip-benchmark when ready."
  exit 0
fi

# --- Analysis --------------------------------------------------------------
echo
echo "[4/6] Analysis (LLM judge runs here; this is slow)..."
"$PY" analyze_results.py

echo
echo "[5/6] Charts and LaTeX table..."
"$PY" evaluate_workflows.py
"$PY" make_paper_figures.py
"$PY" make_topology_figure.py

echo
echo "[6/6] Summary"
"$PY" scripts/print_summary.py || true
"$PY" llm_judge.py --spend || true

PROVIDER="${JUDGE_PROVIDER:-ollama}"
echo
echo "[7/7] Snapshotting run..."
"$PY" scripts/snapshot_run.py --label "gen-${GENERATION_PROVIDER}_judge-${PROVIDER}" || true

echo
echo "Done. metrics_table.tex, metrics_table_ablation.tex, and paper_figures/*.png are ready for Overleaf."
