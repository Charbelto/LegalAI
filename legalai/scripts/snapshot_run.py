"""Archive a completed run so a later pass cannot overwrite it.

The expensive part of this experiment is generating the answers (~12 GPU hours);
judging them is cheap and repeatable. That makes a two-pass workflow natural:

    pass 1: local judge, free            -> snapshot
    pass 2: hosted judge, a few dollars  -> snapshot, then compare

Every analysis run overwrites analysis_summary.csv, significance.csv, results.json
and the figures, so snapshot before re-judging or the first pass is gone.

Usage:
    python scripts/snapshot_run.py --label local-judge
    python scripts/snapshot_run.py --list
    python scripts/snapshot_run.py --compare 20260727_local-judge 20260728_openai-judge
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT_DIR / "runs"

# (filename, required) - a snapshot without the required files is not a run.
ARTIFACTS = [
    ("benchmark_runs.jsonl", True),
    ("run_meta.json", False),
    ("analysis_summary.csv", True),
    ("by_query_type.csv", False),
    ("significance.csv", False),
    ("results.json", True),
    ("metrics_table.tex", False),
    ("judge_cache.json", False),
    ("judge_spend.json", False),
    ("eval_dataset.json", False),
    ("judge_validation.csv", False),
]

FIGURE_DIR = "paper_figures"


def _code_fingerprint() -> str:
    """A short hash of the source files that produce a run's numbers.

    Replaces a `git rev-parse HEAD` call: this project is kept fully local with no
    git repository, so a commit id is not available and shelling out to git would
    be a dependency on tooling that is deliberately absent. Hashing the files
    themselves is arguably better provenance anyway - it reflects the code that
    actually ran, including uncommitted edits, which a commit id does not.

    Returns "unknown" rather than raising if anything is unreadable; provenance is
    worth recording but never worth failing a snapshot over.
    """
    import hashlib

    tracked = [
        "config.py", "local_models.py", "benchmark.py", "analyze_results.py",
        "evaluate_workflows.py", "make_paper_figures.py", "llm_judge.py",
        "state.py", "graph/workflow.py", "agents/base.py", "agents/aggregator.py",
    ]
    digest = hashlib.sha1()
    seen = 0
    for relative in sorted(tracked):
        # ROOT_DIR is legalai/, where these sources live. The old git call used
        # ROOT_DIR.parent because the repository sat one level up; that offset does
        # not apply to source paths.
        path = ROOT_DIR / relative
        try:
            digest.update(relative.encode("utf-8"))
            digest.update(path.read_bytes())
            seen += 1
        except Exception:
            continue
    if not seen:
        return "unknown"
    return f"src-{digest.hexdigest()[:10]} ({seen} files)"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def snapshot(label: str) -> Path:
    missing = [name for name, required in ARTIFACTS if required and not (ROOT_DIR / name).exists()]
    if missing:
        raise SystemExit(
            "Cannot snapshot - these files are missing: "
            + ", ".join(missing)
            + "\nRun the benchmark and analyze_results.py first."
        )

    safe_label = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)
    target = RUNS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M')}_{safe_label}"
    target.mkdir(parents=True, exist_ok=False)

    copied = []
    for name, _required in ARTIFACTS:
        source = ROOT_DIR / name
        if source.exists():
            shutil.copy2(source, target / name)
            copied.append(name)

    figures = ROOT_DIR / FIGURE_DIR
    if figures.is_dir():
        shutil.copytree(figures, target / FIGURE_DIR)
        copied.append(f"{FIGURE_DIR}/ ({len(list(figures.glob('*.png')))} png)")

    results = _read_json(ROOT_DIR / "results.json")
    provenance = results.get("provenance", {})
    spend = _read_json(ROOT_DIR / "judge_spend.json")
    run_meta = _read_json(ROOT_DIR / "run_meta.json")

    # Generation (the system under test) is recorded separately from the judge,
    # so an ollama-generation snapshot and a deepseek-generation snapshot stay
    # distinguishable even when judged by the same model - the whole point of
    # running both is to compare them later.
    #
    # Preference order, most to least authoritative:
    #  1. results.json provenance.generation_provider (set by analyze_results.py
    #     directly from config.GENERATION_PROVIDER - correct once you've re-run
    #     analysis after this field was added).
    #  2. run_meta.json server_config (recorded by the actual backend process
    #     that generated the answers - authoritative even without re-running
    #     analysis, but only has chat_model, not a raw provider/model split for
    #     ollama vs deepseek beyond what's already resolved).
    #  3. run_meta.json's env block - benchmark.py's OWN process environment,
    #     which only reflects PowerShell-level env vars (like -GenerationProvider)
    #     and misses anything that only ever lived in .env inside the server
    #     subprocess. Weakest source; kept only for older runs with neither of
    #     the above.
    server_config = run_meta.get("server_config") or {}
    gen_env = run_meta.get("env") or {}

    generation_provider = (
        provenance.get("generation_provider")
        or server_config.get("generation_provider")
        or gen_env.get("GENERATION_PROVIDER")
        or "ollama"
    )
    if server_config.get("chat_model"):
        generation_model = server_config.get("chat_model")
    else:
        generation_model = (
            gen_env.get("DEEPSEEK_MODEL") if generation_provider == "deepseek" else gen_env.get("OLLAMA_MODEL")
        )

    manifest = {
        "label": label,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        # Was git_commit. This project is kept fully local with no repository, so
        # provenance is a hash of the source files that actually ran.
        "code_fingerprint": _code_fingerprint(),
        "generation": {
            "provider": generation_provider,
            "model": generation_model,
            # Under local_peft there is no single generation model, so record the
            # per-expert breakdown and which arm(s) the snapshot covers. Without
            # this a local_peft snapshot's provenance would say only
            # "local_peft", which is not enough to reproduce it.
            "expert_models": provenance.get("expert_models"),
            "coordinator_role": provenance.get("coordinator_role"),
            "arms": provenance.get("arms"),
            "runs_per_arm": provenance.get("runs_per_arm"),
            "load_in_4bit": provenance.get("local_load_in_4bit"),
            "quant_type": provenance.get("local_quant_type"),
        },
        "judge": {
            # provenance.judge_provider (from results.json, set by
            # analyze_results.py) is authoritative and survives running this
            # script later/elsewhere. os.getenv is only a fallback for older
            # results.json files predating that field, and only correct if this
            # script happens to run in the same shell session as the analysis did.
            "provider": provenance.get("judge_provider") or os.getenv("JUDGE_PROVIDER", "ollama"),
            "model": provenance.get("judge_model") or os.getenv("JUDGE_MODEL"),
            "gold_model": provenance.get("gold_model"),
            "system_model": provenance.get("system_model"),
            "failures_excluded": provenance.get("judge_failures_excluded"),
            "spend_usd": spend.get("total_usd"),
            "billed_calls": spend.get("calls"),
        },
        "experiment": {
            "queries": provenance.get("queries"),
            "modes": provenance.get("modes"),
            "query_types": provenance.get("query_types"),
            "repeats_per_cell": provenance.get("repeats_per_cell"),
            "runs_analyzed": provenance.get("runs_analyzed"),
            "retrieval_metrics_reportable": provenance.get("retrieval_metrics_reportable"),
            "queries_with_relevance_annotation": provenance.get("queries_with_relevance_annotation"),
            # Whether the peft-vs-base control actually ran. A snapshot without
            # it cannot support any RQ2 claim, and that should be visible in the
            # manifest rather than inferred from row counts later.
            "ablation_tested": provenance.get("ablation_tested"),
        },
        "run_meta": provenance.get("run_meta", {}),
        "files": copied,
    }
    (target / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Snapshot written to runs/{target.name}")
    print(f"  generation : {manifest['generation']['provider']}/{manifest['generation']['model']}")
    print(f"  judge   : {manifest['judge']['provider']}/{manifest['judge']['model']}")
    print(f"  queries : {manifest['experiment']['queries']}  "
          f"modes: {len(manifest['experiment']['modes'] or [])}  "
          f"repeats: {manifest['experiment']['repeats_per_cell']}")
    if manifest["judge"]["spend_usd"]:
        print(f"  spend   : ${manifest['judge']['spend_usd']:.4f}")
    print(f"  files   : {len(copied)}")
    return target


def list_snapshots():
    if not RUNS_DIR.is_dir():
        print("No snapshots yet.")
        return
    entries = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir())
    if not entries:
        print("No snapshots yet.")
        return
    print(f"{'snapshot':38s} {'generation':22s} {'judge':28s} {'queries':>7} {'spend':>8}")
    for entry in entries:
        manifest = _read_json(entry / "MANIFEST.json")
        generation = manifest.get("generation", {})
        judge = manifest.get("judge", {})
        experiment = manifest.get("experiment", {})
        generation_label = f"{generation.get('provider', 'ollama')}/{generation.get('model', '?')}"
        judge_label = f"{judge.get('provider')}/{judge.get('model')}"
        spend = judge.get("spend_usd")
        spend_text = f"${spend:.4f}" if isinstance(spend, (int, float)) else "-"
        print(f"{entry.name:38s} {generation_label:22s} {judge_label:28s} {str(experiment.get('queries')):>7} {spend_text:>8}")


def compare(name_a: str, name_b: str):
    """Side-by-side judge scores per topology from two snapshots."""
    import csv

    def load_summary(name):
        """Index a snapshot's summary by (mode, arm).

        Keying on mode alone was wrong once the PEFT pivot made
        analysis_summary.csv one row per (mode, arm): six rows collapsed into
        three and whichever arm happened to come last in the file silently
        overwrote the other, so this comparison reported one arm's numbers under
        a label that named only the topology.
        """
        path = RUNS_DIR / name / "analysis_summary.csv"
        if not path.exists():
            raise SystemExit(f"No analysis_summary.csv in snapshot '{name}'")
        with open(path, "r", encoding="utf-8") as handle:
            rows = {}
            for row in csv.DictReader(handle):
                # Pre-pivot snapshots have no arm column; label them so they
                # never silently pair with a specific arm of a newer snapshot.
                rows[(row["mode"], row.get("arm") or "n/a")] = row
            return rows

    summary_a = load_summary(name_a)
    summary_b = load_summary(name_b)
    full_a = _read_json(RUNS_DIR / name_a / "MANIFEST.json")
    full_b = _read_json(RUNS_DIR / name_b / "MANIFEST.json")
    manifest_a = full_a.get("judge", {})
    manifest_b = full_b.get("judge", {})
    gen_a = full_a.get("generation", {})
    gen_b = full_b.get("generation", {})

    print(f"A = {name_a}  generation={gen_a.get('provider', 'ollama')}/{gen_a.get('model', '?')}  judge={manifest_a.get('provider')}/{manifest_a.get('model')}")
    print(f"B = {name_b}  generation={gen_b.get('provider', 'ollama')}/{gen_b.get('model', '?')}  judge={manifest_b.get('provider')}/{manifest_b.get('model')}")
    if gen_a.get("provider") != gen_b.get("provider"):
        print(
            "NOTE: A and B used different generation providers - this compares "
            "topology performance across two different systems under test, not "
            "just two judges scoring the same answers."
        )
    same_generation = gen_a.get("provider", "ollama") == gen_b.get("provider", "ollama")

    print()
    label_width = 30
    if same_generation:
        print(f"{'topology / arm':{label_width}s} {'judge A':>9} {'judge B':>9} {'delta':>8}   {'rougeL':>8}")
    else:
        print(f"{'topology / arm':{label_width}s} {'judge A':>9} {'judge B':>9} {'delta':>8}   {'rougeL A':>9} {'rougeL B':>9}")
    print("-" * (label_width + 42))

    def get_float(row, key):
        try:
            return float(row.get(key, "") or "nan")
        except ValueError:
            return float("nan")

    for cell in sorted(set(summary_a) | set(summary_b)):
        mode, arm = cell
        row_a = summary_a.get(cell, {})
        row_b = summary_b.get(cell, {})
        # A cell present in only one snapshot is reported as such, not silently
        # differenced against a missing row.
        if not row_a or not row_b:
            only_in = "A" if row_a else "B"
            print(f"{mode + ' / ' + arm:{label_width}s} present only in snapshot {only_in}")
            continue
        judge_a = get_float(row_a, "judge_average_mean")
        judge_b = get_float(row_b, "judge_average_mean")
        delta = judge_b - judge_a
        rouge_a = get_float(row_a, "rouge_l_f_mean")
        label = f"{mode} / {arm}"
        if same_generation:
            print(f"{label:{label_width}s} {judge_a:9.3f} {judge_b:9.3f} {delta:+8.3f}   {rouge_a:8.4f}")
        else:
            rouge_b = get_float(row_b, "rouge_l_f_mean")
            print(f"{label:{label_width}s} {judge_a:9.3f} {judge_b:9.3f} {delta:+8.3f}   {rouge_a:9.4f} {rouge_b:9.4f}")

    if same_generation:
        print(
            "\nROUGE-L is identical across snapshots by construction (same answers), so it is "
            "shown once as a sanity check: if it differs, the snapshots are from different runs.\n"
            "If the two judges rank the topologies the same way, your conclusion is robust to "
            "judge choice - which is worth a sentence in the paper."
        )
    else:
        print(
            "\nA and B generated different answers (different GENERATION_PROVIDER), so ROUGE-L "
            "legitimately differs too - both are shown, not just one as a sanity check.\n"
            "If the two providers rank the topologies the same way, your conclusion about which "
            "topology helps is robust to the choice of generation model - worth a sentence in the paper."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, help="short name for this snapshot, e.g. local-judge")
    parser.add_argument("--list", action="store_true", help="list existing snapshots")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two snapshots")
    args = parser.parse_args()

    if args.list:
        list_snapshots()
    elif args.compare:
        compare(*args.compare)
    elif args.label:
        snapshot(args.label)
    else:
        parser.print_help()
        print("\nExisting snapshots:")
        list_snapshots()


if __name__ == "__main__":
    main()
