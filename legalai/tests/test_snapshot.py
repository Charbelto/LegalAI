"""Tests for run snapshotting: the safeguard that stops a second judging pass
from destroying the first one's results."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_snapshot_module():
    spec = importlib.util.spec_from_file_location(
        "snapshot_run", ROOT / "scripts" / "snapshot_run.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["snapshot_run"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def snapshot_module(tmp_path):
    module = _load_snapshot_module()
    module.ROOT_DIR = tmp_path
    module.RUNS_DIR = tmp_path / "runs"
    yield module
    sys.modules.pop("snapshot_run", None)


def _write_run(
    root: Path,
    judge_model="gemma4:12b",
    judge_average=3.5,
    provider="ollama",
    spend=None,
    generation_provider=None,
    generation_model=None,
    rouge=0.1739,
):
    (root / "benchmark_runs.jsonl").write_text('{"query_id": "q01", "success": true}\n', encoding="utf-8")
    if generation_provider is not None:
        env = {"GENERATION_PROVIDER": generation_provider}
        if generation_provider == "deepseek":
            env["DEEPSEEK_MODEL"] = generation_model or "deepseek-v4-flash"
        else:
            env["OLLAMA_MODEL"] = generation_model or "qwen2.5"
        (root / "run_meta.json").write_text(json.dumps({"env": env}), encoding="utf-8")
    (root / "results.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "judge_model": judge_model,
                    "gold_model": "llama3.1:8b",
                    "system_model": "qwen2.5",
                    "queries": 30,
                    "modes": ["single", "all"],
                    "repeats_per_cell": 5,
                    "runs_analyzed": 1200,
                    "judge_failures_excluded": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    with open(root / "analysis_summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "judge_average_mean", "rouge_l_f_mean"])
        writer.writerow(["single", judge_average, rouge])
        writer.writerow(["all", judge_average - 0.4, round(rouge - 0.0072, 4)])
    if spend is not None:
        (root / "judge_spend.json").write_text(
            json.dumps({"total_usd": spend, "calls": 1200}), encoding="utf-8"
        )
    figures = root / "paper_figures"
    figures.mkdir(exist_ok=True)
    (figures / "fig1_operational.png").write_bytes(b"not-a-real-png")


def test_snapshot_copies_artifacts_and_writes_manifest(snapshot_module, tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_PROVIDER", "ollama")
    _write_run(tmp_path)

    target = snapshot_module.snapshot("local-judge")

    assert (target / "benchmark_runs.jsonl").exists()
    assert (target / "analysis_summary.csv").exists()
    assert (target / "paper_figures" / "fig1_operational.png").exists()

    manifest = json.loads((target / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["label"] == "local-judge"
    assert manifest["judge"]["model"] == "gemma4:12b"
    assert manifest["experiment"]["queries"] == 30
    assert "local-judge" in target.name


def test_snapshot_refuses_incomplete_run(snapshot_module, tmp_path):
    (tmp_path / "benchmark_runs.jsonl").write_text("{}\n", encoding="utf-8")
    # No results.json / analysis_summary.csv.

    with pytest.raises(SystemExit, match="missing"):
        snapshot_module.snapshot("incomplete")


def test_second_snapshot_does_not_overwrite_the_first(snapshot_module, tmp_path, monkeypatch):
    """The whole point: a hosted-judge pass must not destroy the local results."""
    _write_run(tmp_path, judge_model="gemma4:12b", judge_average=3.5)
    first = snapshot_module.snapshot("local-judge")

    # Simulate re-judging with a hosted model overwriting the working files.
    _write_run(tmp_path, judge_model="hosted-model", judge_average=4.1, spend=1.87)
    second = snapshot_module.snapshot("openai-judge")

    assert first != second
    first_manifest = json.loads((first / "MANIFEST.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "MANIFEST.json").read_text(encoding="utf-8"))

    assert first_manifest["judge"]["model"] == "gemma4:12b"
    assert second_manifest["judge"]["model"] == "hosted-model"
    assert second_manifest["judge"]["spend_usd"] == pytest.approx(1.87)

    with open(first / "analysis_summary.csv", encoding="utf-8") as handle:
        rows = {row["mode"]: row for row in csv.DictReader(handle)}
    assert float(rows["single"]["judge_average_mean"]) == pytest.approx(3.5)


def test_snapshot_records_generation_provider(snapshot_module, tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_PROVIDER", "deepseek")
    _write_run(tmp_path, generation_provider="deepseek", generation_model="deepseek-v4-flash")

    target = snapshot_module.snapshot("gen-deepseek_judge-deepseek")

    manifest = json.loads((target / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["generation"]["provider"] == "deepseek"
    assert manifest["generation"]["model"] == "deepseek-v4-flash"


def test_snapshot_defaults_generation_to_ollama_when_run_meta_missing(snapshot_module, tmp_path):
    """Older snapshots (taken before this feature existed) have no run_meta.json
    env block - must not crash, and must assume the ollama default rather than
    silently reporting nothing."""
    _write_run(tmp_path)  # generation_provider=None -> no run_meta.json written

    target = snapshot_module.snapshot("no-run-meta")

    manifest = json.loads((target / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["generation"]["provider"] == "ollama"


def test_compare_flags_differing_generation_providers(snapshot_module, tmp_path, capsys):
    """The real point of the toggle: running local vs hosted generation and
    being able to tell, from the comparison output, that they actually differ."""
    _write_run(
        tmp_path,
        judge_model="deepseek-v4-flash",
        judge_average=3.5,
        generation_provider="ollama",
        generation_model="qwen2.5",
        rouge=0.20,
    )
    first = snapshot_module.snapshot("gen-ollama")

    _write_run(
        tmp_path,
        judge_model="deepseek-v4-flash",
        judge_average=4.0,
        generation_provider="deepseek",
        generation_model="deepseek-v4-flash",
        rouge=0.31,
    )
    second = snapshot_module.snapshot("gen-deepseek")

    snapshot_module.compare(first.name, second.name)

    output = capsys.readouterr().out
    assert "generation=ollama/qwen2.5" in output
    assert "generation=deepseek/deepseek-v4-flash" in output
    assert "different generation providers" in output.lower()
    assert "0.2000" in output and "0.3100" in output  # both rougeL columns shown


def test_compare_reports_both_judges(snapshot_module, tmp_path, capsys):
    _write_run(tmp_path, judge_model="gemma4:12b", judge_average=3.5)
    first = snapshot_module.snapshot("local-judge")
    _write_run(tmp_path, judge_model="hosted-model", judge_average=4.1, spend=2.0)
    second = snapshot_module.snapshot("openai-judge")

    snapshot_module.compare(first.name, second.name)

    output = capsys.readouterr().out
    assert "gemma4:12b" in output
    assert "hosted-model" in output
    assert "+0.600" in output  # 4.1 - 3.5 for single
