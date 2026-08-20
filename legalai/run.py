"""The full experiment: both arms of the 540-run benchmark, then analysis,
statistics, charts, and LaTeX tables.

    python run.py
    python run.py --concurrency 1   # fall back to fully sequential/isolated latency

Defaults to --concurrency 4, paired with LEGALAI_GENERAL_POOL_SIZE=2 in .env
(see VASTAI_DEPLOY.md) - a reasonable pairing for a 24GB GPU, but a genuinely
tight one on VRAM (see the pre-flight check below), not a "safe for any card"
default. Pass --concurrency 1 to disable it entirely.

Takes hours, not minutes - run python test.py first and use its time estimate
before starting this unattended. Safe to leave running over SSH/screen/tmux;
each step prints its own progress.

Produces: benchmark_runs.jsonl, run_meta*.json, analysis_summary.csv,
significance.csv, by_query_type.csv, metrics_table.tex,
metrics_table_ablation.tex, paper_figures/*.png.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ollama_setup import ensure_ollama_ready

ROOT = Path(__file__).resolve().parent
DEFAULT_CONCURRENCY = 4


def run(cmd: list[str], required: bool = True) -> None:
    print(f"\n[run] $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0 and required:
        sys.exit(f"[run] '{' '.join(cmd)}' failed (exit {result.returncode}). Stopping.")


def check_vram_fits() -> None:
    """Refuse to start a multi-hour run on a config that won't fit in VRAM.

    LEGALAI_GENERAL_POOL_SIZE=2 (this script's paired default with
    --concurrency 4) is genuinely tight on a 24GB card - see VASTAI_DEPLOY.md's
    VRAM math. Better to fail in the ~30s this takes than 3 hours into the run.
    """
    print("[run] Pre-flight: confirming the model pool fits in VRAM (finetune/check_vram.py)...")
    result = subprocess.run(
        [sys.executable, "finetune/check_vram.py", "--concurrent", "--no-generate"],
        cwd=ROOT,
    )
    if result.returncode != 0:
        sys.exit(
            "[run] VRAM check failed - the configured model pool does not fit on this "
            "GPU. Lower LEGALAI_GENERAL_POOL_SIZE in .env (try 1) and/or pass "
            "--concurrency 1, or rent a bigger card. See VASTAI_DEPLOY.md."
        )
    print("[run] VRAM check passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Max in-flight benchmark requests. Defaults to {DEFAULT_CONCURRENCY}, "
        "paired with LEGALAI_GENERAL_POOL_SIZE=2 in .env. Pass 1 for the safest, "
        "most isolated latency measurement (no pooling/concurrency benefit). See "
        "VASTAI_DEPLOY.md's 'use the spare VRAM' section for the VRAM trade-off.",
    )
    parser.add_argument(
        "--skip-vram-check", action="store_true",
        help="Skip the pre-flight VRAM check. Not recommended - see check_vram_fits().",
    )
    args = parser.parse_args()
    concurrency_flags = ["--concurrency", str(args.concurrency)] if args.concurrency > 1 else []

    ensure_ollama_ready()

    if args.concurrency > 1 and not args.skip_vram_check:
        check_vram_fits()

    print("[run] [1/6] Validity tests...")
    run([sys.executable, "-m", "pytest", "tests", "-q"])

    print("\n[run] [2/6] Benchmark - arm 1/2: peft (270 runs)...")
    run([sys.executable, "-u", "benchmark.py", "--arm", "peft", *concurrency_flags])

    print("\n[run] [3/6] Benchmark - arm 2/2: base (270 runs)...")
    run([sys.executable, "-u", "benchmark.py", "--arm", "base", "--append", *concurrency_flags])

    print("\n[run] [4/6] Judge preflight (one live call, confirms cost/config)...")
    run([sys.executable, "llm_judge.py", "--check"])

    print("\n[run] [5/6] Analysis - the LLM judge scores every run here; this is the slow step...")
    run([sys.executable, "analyze_results.py"])

    print("\n[run] [6/6] Charts and LaTeX tables...")
    run([sys.executable, "evaluate_workflows.py"])
    run([sys.executable, "make_paper_figures.py"])
    run([sys.executable, "make_topology_figure.py"])

    # Non-critical wrap-up - don't fail the whole run over a summary script.
    run([sys.executable, "scripts/print_summary.py"], required=False)
    run([sys.executable, "llm_judge.py", "--spend"], required=False)

    print("\n=== Done ===")
    print("Ready to copy out - see VASTAI_DEPLOY.md 'Get the results back':")
    print("  evaluation_assets/*.png     <- what overleaf_paper.tex actually includes (as figures/)")
    print("  metrics_table.tex, metrics_table_ablation.tex")
    print("  benchmark_runs.jsonl, run_meta*.json, analysis_summary.csv, significance.csv,")
    print("  by_query_type.csv, results.json")
    print("  paper_figures/*.png         <- extra figure set, not yet wired into the paper")


if __name__ == "__main__":
    main()
