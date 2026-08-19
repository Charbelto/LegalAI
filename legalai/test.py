"""Quick sanity check before committing to the full run.

    python test.py
    python test.py --concurrency 2   # match whatever you plan to pass to run.py

Runs 1 query x 3 topologies x 1 repeat, for both the peft and base arms (6
requests total) - a few minutes, not hours - then prints a time estimate for
the full 540-run benchmark extrapolated from what it just measured on THIS
machine. Run this before python run.py, not instead of it: the numbers this
produces are too few to trust as results, only as a sanity/timing check. Pass
the same --concurrency you intend to use for run.py so the timing estimate
reflects the actual setting, not the sequential default.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ollama_setup import ensure_ollama_ready

ROOT = Path(__file__).resolve().parent
SMOKE_FILE = ROOT / "benchmark_runs_smoke.jsonl"
FULL_RUN_COUNT = 540  # 30 queries x 3 topologies x 3 repeats x 2 arms


def run(cmd: list[str]) -> None:
    print(f"[test] $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    concurrency_flags = ["--concurrency", str(args.concurrency)] if args.concurrency > 1 else []

    ensure_ollama_ready()

    print("\n[test] Validity tests...")
    run([sys.executable, "-m", "pytest", "tests", "-q"])

    print("\n[test] Smoke benchmark (peft arm)...")
    run([sys.executable, "benchmark.py", "--smoke", "--arm", "peft", *concurrency_flags])
    print("\n[test] Smoke benchmark (base arm)...")
    run([sys.executable, "benchmark.py", "--smoke", "--arm", "base", "--append", *concurrency_flags])

    rows = [
        json.loads(line)
        for line in SMOKE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    succeeded = [r for r in rows if r.get("success")]
    failed = [r for r in rows if not r.get("success")]

    print(f"\n[test] {len(succeeded)} succeeded, {len(failed)} failed")
    for r in failed:
        print(f"[test]   FAILED {r.get('mode')}/{r.get('arm')}: {r.get('error')}")

    if succeeded:
        times = [r["elapsed_s"] for r in succeeded]
        avg = sum(times) / len(times)
        hours = avg * FULL_RUN_COUNT / 3600
        print(f"\n[test] Average {avg:.1f}s per run over {len(times)} runs")
        print(f"[test] -> estimated full benchmark ({FULL_RUN_COUNT} runs): {hours:.1f} hours")

    if failed:
        print("\n[test] Fix the failures above before running python run.py.")
        sys.exit(1)

    print("\n[test] Inspect a few answers before trusting this:")
    print(
        "  python -c \"import json; [print(r['arm'], r['mode'], round(r.get('elapsed_s',0),1), "
        f"repr(r.get('response',''))[:150]) for r in map(json.loads, open('{SMOKE_FILE.name}', "
        "encoding='utf-8'))]\""
    )
    print("\n[test] Red flags to check for (see run_experiment.ps1's checklist):")
    print("  - the abstention sentence in every response -> retrieval is empty")
    print("  - peft and base answers identical -> adapters are inert")
    print("  - all 3 topologies identical within an arm -> sampling is off")
    print("\n[test] Looks right? Run: python run.py")


if __name__ == "__main__":
    main()
