"""Sample answers into a CSV for human scoring, to validate the LLM judge.

An 8B-parameter judge grading a 7B system's answers against machine-drafted
reference answers is the weakest link in the evaluation. Reporting agreement
between the judge and a human on a sample removes that objection cheaply.

Usage:
    python scripts/make_judge_validation_sheet.py --n 25
    # score the three human_* columns yourself (1-5), then:
    python llm_judge.py --validate judge_validation.csv

The sample is stratified across query types and topologies so the agreement
estimate is not dominated by one condition, and the judge's own scores are left
out of the sheet so they cannot anchor your ratings.
"""

import argparse
import csv
import json
import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RUNS_FILE = ROOT_DIR / "benchmark_runs.jsonl"
DATASET_FILE = ROOT_DIR / "eval_dataset.json"
OUT_FILE = ROOT_DIR / "judge_validation.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25, help="rows to sample")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if not RUNS_FILE.exists():
        raise SystemExit(f"{RUNS_FILE.name} not found - run the benchmark first.")

    dataset = {item["id"]: item for item in json.loads(DATASET_FILE.read_text(encoding="utf-8"))}

    runs = []
    with open(RUNS_FILE, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("success") and row.get("response"):
                runs.append(row)

    if not runs:
        raise SystemExit("No successful runs with responses found.")

    # Stratify by (query_type, mode) so no single condition dominates the sample.
    buckets = {}
    for row in runs:
        key = (row.get("query_type"), row.get("mode"))
        buckets.setdefault(key, []).append(row)

    rng = random.Random(args.seed)
    ordered_keys = sorted(buckets.keys(), key=lambda k: (str(k[0]), str(k[1])))
    sample = []
    while len(sample) < args.n and ordered_keys:
        for key in list(ordered_keys):
            bucket = buckets[key]
            if not bucket:
                ordered_keys.remove(key)
                continue
            sample.append(bucket.pop(rng.randrange(len(bucket))))
            if len(sample) >= args.n:
                break

    with open(OUT_FILE, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "query_id",
                "query_type",
                "mode",
                "repeat",
                "query",
                "gold",
                "answer",
                "human_accuracy",
                "human_completeness",
                "human_groundedness",
                "notes",
            ]
        )
        for row in sample:
            item = dataset.get(row["query_id"], {})
            writer.writerow(
                [
                    row.get("query_id"),
                    row.get("query_type"),
                    row.get("mode"),
                    row.get("repeat"),
                    item.get("query", ""),
                    row.get("gold", ""),
                    row.get("response", ""),
                    "",  # human_accuracy       1-5
                    "",  # human_completeness   1-5
                    "",  # human_groundedness   1-5
                    "",
                ]
            )

    print(f"Wrote {len(sample)} rows to {OUT_FILE.name}")
    print("Score human_accuracy / human_completeness / human_groundedness on 1-5, then run:")
    print("    python llm_judge.py --validate judge_validation.csv")
    print(
        "\nScoring guide - accuracy: is it factually right against the reference? "
        "completeness: does it cover the reference's core points? "
        "groundedness: does it cite real provisions without inventing any?"
    )


if __name__ == "__main__":
    main()
