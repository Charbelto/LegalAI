"""The full experiment: both arms of the 540-run benchmark, then analysis,
statistics, charts, and LaTeX tables.

    python run.py
    python run.py --concurrency 2   # use spare VRAM - see VASTAI_DEPLOY.md before raising this

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


def run(cmd: list[str], required: bool = True) -> None:
    print(f"\n[run] $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0 and required:
        sys.exit(f"[run] '{' '.join(cmd)}' failed (exit {result.returncode}). Stopping.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Max in-flight benchmark requests. 1 (default) is the safest, most "
        "isolated latency measurement. Raising this trades per-request latency "
        "purity for wall-clock speed on a VRAM-rich GPU - see VASTAI_DEPLOY.md's "
        "'use the spare VRAM' section before raising it.",
    )
    args = parser.parse_args()
    concurrency_flags = ["--concurrency", str(args.concurrency)] if args.concurrency > 1 else []

    ensure_ollama_ready()

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
    print("metrics_table.tex, metrics_table_ablation.tex, and paper_figures/*.png are ready.")
    print("Copy them back to your local machine (scp) or commit + push - see VASTAI_DEPLOY.md.")


if __name__ == "__main__":
    main()
