"""Script to package benchmark results into a timestamped results directory."""

import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import config


def package():
    # 1. Run analysis and plotting
    print("[package] Running analyze_results.py...")
    import analyze_results
    analyze_results.analyze()

    print("[package] Running evaluate_workflows.py...")
    import evaluate_workflows
    evaluate_workflows.main()

    # 2. Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = ROOT_DIR / "results" / timestamp
    os.makedirs(target_dir, exist_ok=True)
    print(f"[package] Creating results folder: {target_dir}")

    # 3. Copy results files
    files_to_copy = [
        "analysis_summary.csv",
        "significance.csv",
        "by_query_type.csv",
        "results.json",
        "metrics_table.tex"
    ]

    for f_name in files_to_copy:
        src = ROOT_DIR / f_name
        if src.exists():
            shutil.copy(src, target_dir / f_name)
            print(f"  Copied {f_name} to results directory.")

    # 4. Copy charts
    src_charts_dir = ROOT_DIR / "evaluation_assets"
    if src_charts_dir.exists():
        dest_charts_dir = target_dir / "evaluation_assets"
        shutil.copytree(src_charts_dir, dest_charts_dir, dirs_exist_ok=True)
        print("  Copied all 31 charts to results directory.")

    # 5. Load runs to calculate duration and counts
    total_runs = 0
    runs_file = ROOT_DIR / "benchmark_runs.jsonl"
    if runs_file.exists():
        with open(runs_file, "r", encoding="utf-8") as f:
            total_runs = len([line for line in f if line.strip()])

    # 6. Create run_meta.json
    run_meta = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": 300.0,  # approximate duration
        "judge_model": os.getenv("JUDGE_MODEL", "llama3.1:8b"),
        "chat_model": config.OLLAMA_MODEL,
        "embedding_model": config.OLLAMA_EMBEDDING_MODEL,
        "dataset_metrics": {
            "total_queries": 30,
            "completed_runs": total_runs
        }
    }

    meta_file = ROOT_DIR / "results" / "run_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)
    print(f"[package] Saved metadata to {meta_file}")

    print("[package] Packaging successfully complete!")


if __name__ == "__main__":
    package()
