"""Print a short, honest summary of the most recent analysis run."""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def main():
    results = ROOT_DIR / "results.json"
    if not results.exists():
        print("results.json missing -- run analyze_results.py first")
        return

    bundle = json.loads(results.read_text(encoding="utf-8"))
    prov = bundle.get("provenance", {})

    arms = prov.get("arms", [])
    print(
        f"  queries={prov.get('queries')}  topologies={len(prov.get('modes', []))}  "
        f"arms={arms or 'n/a'}  repeats={prov.get('repeats_per_cell')}  "
        f"runs={prov.get('runs_analyzed')}"
    )
    if prov.get("runs_per_arm"):
        print(f"  runs per arm: {prov['runs_per_arm']}")
    print(
        f"  judge={prov.get('judge_model')}  gold={prov.get('gold_model')}  "
        f"system={prov.get('system_model')}"
    )
    for expert in prov.get("expert_models", []):
        print(
            f"    expert {expert['role']:11s} {expert['base_model']:42s} "
            f"tuned on {expert['finetune_dataset']}"
        )
    if prov.get("judge_failures_excluded"):
        print(f"  WARNING {prov['judge_failures_excluded']} judge calls excluded")
    if prov.get("gold_needs_review"):
        print(
            "  WARNING reference answers are still flagged needs_review -- "
            "do not describe them as expert-curated"
        )
    if not prov.get("ablation_tested"):
        print(
            "  WARNING no peft-vs-base ablation in this run -- RQ2 is unanswerable "
            "and must be dropped from the paper rather than asserted"
        )

    significance = bundle.get("significance", [])

    def _print_rows(rows, heading):
        print(f"\n  {heading}")
        for row in rows:
            p = row.get("judge_average_p_holm")
            delta = row.get("judge_average_cliffs_delta")
            label = row.get("comparison", "?")
            if p is None or (isinstance(p, float) and p != p):
                print(f"    {label:34s} not tested (underpowered)")
            else:
                verdict = "SIGNIFICANT" if p < 0.05 else "n.s."
                delta_txt = f"{delta:+.2f}" if isinstance(delta, (int, float)) else "n/a"
                print(f"    {label:34s} p_holm={p:.4f} delta={delta_txt}  {verdict}")

    # RQ1/RQ3: pairwise topology comparisons, within each arm.
    for arm in arms or [None]:
        for scope_label in [
            "overall",
            "query_type=simple",
            "query_type=decomposable",
            "query_type=routing",
        ]:
            rows = [
                r
                for r in significance
                if r.get("scope") == scope_label and (arm is None or r.get("arm") == arm)
            ]
            if rows:
                _print_rows(
                    rows,
                    f"topology pairs [arm={arm} scope={scope_label}] (judge average):",
                )

    # RQ2: the specialisation ablation.
    ablation = [r for r in significance if r.get("scope") == "arm_ablation"]
    if ablation:
        _print_rows(ablation, "RQ2 ablation, peft vs base per topology (judge average):")

    print(
        "\n  Reminder: a positive Cliff's delta means the FIRST side named in the "
        "comparison scored higher (so for 'peft_vs_base', positive favours the "
        "fine-tuned arm). A non-significant result is inconclusive, not proof of "
        "equivalence."
    )


if __name__ == "__main__":
    main()
