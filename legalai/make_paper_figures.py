"""Build the five composite figures the paper embeds.

Why composites: Overleaf was hitting its compile timeout because the draft pulled
in ~15 separate PNGs through nested `subfigure` environments. Each panel is now
drawn by matplotlib into a single moderate-resolution PNG, so LaTeX loads a
handful of images instead of fifteen and the document compiles in a fraction of
the time.

Changed by the PEFT pivot
-------------------------
Three topologies instead of seven, so every panel is far less crowded. More
importantly, there is no longer a single-agent baseline to highlight in red and
measure everything against: SINGLE is out of the compared set, and the question
is which of three structures combines three specialised agents best. So:

* each topology gets its own colour rather than baseline-vs-rest;
* the trade-off panel shades the region *dominated* by the best-quality
  topology (slower and worse) instead of the region worse than SINGLE;
* figures 1-4 are drawn for one arm at a time (default: peft) because mixing a
  fine-tuned system with its own untuned control into one bar chart would be
  meaningless;
* figure 5 is new and is the RQ2 ablation: peft vs base, side by side.

Inputs (produced by analyze_results.py):
    analysis_summary.csv   per-(mode, arm) means, standard deviations, 95% CI margins
    by_query_type.csv      per-(query_type, mode, arm) means, for the H2 interaction

Outputs (paper_figures/):
    fig1_operational.png       latency, node breakdown, steps, token cost
    fig2_quality.png           BLEU, ROUGE, judge scores, abstention
    fig3_tradeoff.png          quality vs latency with error bars
    fig4_query_type.png        quality by topology x query type
    fig5_ablation.png          PEFT vs untuned base, per topology (RQ2)

Usage:
    python make_paper_figures.py
    python make_paper_figures.py --arm base      # draw 1-4 for the control arm
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
SUMMARY_FILE = ROOT_DIR / "analysis_summary.csv"
BY_TYPE_FILE = ROOT_DIR / "by_query_type.csv"
EVAL_DATASET_FILE = ROOT_DIR / "eval_dataset.json"
OUT_DIR = ROOT_DIR / "paper_figures"

# Figures are sized for a two-column IEEE page at \textwidth and saved at a
# resolution that stays legible without bloating the PDF.
DPI = 150

# The three compared topologies, in structural order: strict chain, full
# concurrency, converging dependency.
MODE_ORDER = ["all", "parallel", "dag"]
MODE_LABELS = {
    "all": "ALL\n(sequential)",
    "parallel": "PARALLEL\n(concurrent)",
    "dag": "DAG\n(converging)",
    # Kept so a run that included the still-implemented topologies still plots.
    "single": "SINGLE\n(1 agent)",
    "legal_first": "LEGAL-FIRST\n(conditional)",
    "planner_based": "PLANNER",
    "verify_only": "VERIFY-ONLY\n(bypass)",
    "legal_news_parallel": "LEGAL-NEWS",
}

# One colour per topology. No baseline colour any more - nothing in the compared
# set is a control, so colouring one bar red would imply a reference that the
# design no longer has.
MODE_COLORS = {
    "all": "#2c3e50",
    "parallel": "#2980b9",
    "dag": "#16a085",
}
FALLBACK_COLOR = "#7f8c8d"
ACCENT = "#2980b9"
# Arm colours for the ablation figure.
ARM_COLORS = {"peft": "#8e44ad", "base": "#95a5a6"}
ARM_LABELS = {"peft": "PEFT (LoRA-specialised)", "base": "Base (untuned control)"}

NODE_ORDER = [
    "planner",
    "router",
    "memory",
    "retrieval",
    "legal",
    "news",
    "general_qa",
    "aggregator",
    "validator",
    "response",
]


def _query_type_counts():
    """How many distinct queries fall in each query_type, for figure x-axis labels.

    Was previously hardcoded as the literal string "(n queries)" in every label,
    which is a placeholder, not data - fixed to read the real per-type count from
    the dataset the run actually used.
    """
    try:
        data = json.loads(EVAL_DATASET_FILE.read_text(encoding="utf-8"))
        items = data.get("queries") or data.get("items") if isinstance(data, dict) else data
        if isinstance(data, dict) and items is None:
            items = next(iter(data.values()))
        counts = {}
        for item in items:
            qtype = item.get("query_type") or item.get("type")
            if qtype:
                counts[qtype] = counts.get(qtype, 0) + 1
        return counts
    except Exception:
        return {}


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Missing {path.name}. Run the benchmark and then analyze_results.py first:\n"
            "    python benchmark.py\n"
            "    python analyze_results.py"
        )
    frame = pd.read_csv(path)
    # Pre-pivot outputs have no arm column; treat them as a single unnamed arm
    # rather than crashing, so an old results directory still re-plots.
    if "arm" not in frame.columns:
        frame["arm"] = "unknown"
    return frame


def _select_arm(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Restrict to one arm, falling back with a clear message if it is absent."""
    available = sorted(frame["arm"].dropna().unique())
    if arm in available:
        return frame[frame["arm"] == arm].copy()
    fallback = available[0] if available else None
    print(
        f"[figures] arm '{arm}' not in {available}; using '{fallback}' instead. "
        "Panels are labelled with the arm actually plotted."
    )
    return frame[frame["arm"] == fallback].copy() if fallback else frame.copy()


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    """Order rows by MODE_ORDER, keeping only modes actually present."""
    present = [m for m in MODE_ORDER if m in set(df["mode"])]
    extra = sorted(set(df["mode"]) - set(MODE_ORDER))
    ordered = present + extra
    return df.set_index("mode").loc[ordered].reset_index()


def _labels(modes) -> list:
    return [MODE_LABELS.get(m, m.replace("_", "\n")) for m in modes]


def _colors(modes) -> list:
    return [MODE_COLORS.get(m, FALLBACK_COLOR) for m in modes]


def _col(df: pd.DataFrame, metric: str, suffix: str = "mean"):
    """Return a metric column, or None if this run did not produce it."""
    name = f"{metric}_{suffix}"
    if name not in df.columns:
        print(f"[figures] note: {name} not in summary; panel will be skipped")
        return None
    return pd.to_numeric(df[name], errors="coerce")


def _bar(ax, df, metric, title, ylabel, scale=1.0, annotate=True):
    values = _col(df, metric)
    if values is None:
        ax.set_axis_off()
        ax.set_title(f"{title}\n(not available)", fontsize=9)
        return
    errors = _col(df, metric, "ci")
    values = values * scale
    errors = errors * scale if errors is not None else None

    positions = np.arange(len(df))
    ax.bar(
        positions,
        values,
        yerr=errors,
        capsize=3,
        color=_colors(df["mode"]),
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(_labels(df["mode"]), fontsize=7.5, rotation=0)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    if annotate:
        finite = values.dropna()
        if not finite.empty:
            headroom = 1.18 if finite.max() > 0 else 1.0
            ax.set_ylim(0, finite.max() * headroom)
            for x, value in zip(positions, values):
                if pd.notna(value):
                    ax.text(x, value, f"{value:,.2f}".rstrip("0").rstrip("."),
                            ha="center", va="bottom", fontsize=7)


def figure_operational(summary: pd.DataFrame, arm: str):
    """Latency, where the latency goes, execution steps, and real token cost."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    _bar(axes[0, 0], summary, "elapsed_s", "End-to-End Latency", "seconds")

    # Node-level stacked breakdown: shows *which* stage each topology pays for.
    ax = axes[0, 1]
    bottoms = np.zeros(len(summary))
    palette = plt.cm.tab20(np.linspace(0, 1, len(NODE_ORDER)))
    positions = np.arange(len(summary))
    plotted = False
    for color, node in zip(palette, NODE_ORDER):
        values = _col(summary, f"timing_{node}")
        if values is None:
            continue
        values = (values / 1000.0).fillna(0.0)
        if values.sum() == 0:
            continue
        ax.bar(positions, values, bottom=bottoms, label=node, color=color,
               edgecolor="white", linewidth=0.3)
        bottoms += values.to_numpy()
        plotted = True
    ax.set_xticks(positions)
    ax.set_xticklabels(_labels(summary["mode"]), fontsize=7.5)
    ax.set_title("Latency Decomposition by Node", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("seconds", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    if plotted:
        ax.legend(fontsize=6, ncol=2, loc="upper right", framealpha=0.85)

    _bar(axes[1, 0], summary, "steps", "Graph Execution Steps", "state transitions")
    _bar(axes[1, 1], summary, "cost", "Cost per Query (measured tokens)", "USD")

    fig.suptitle(
        f"Operational profile: coordination cost by topology ({ARM_LABELS.get(arm, arm)} arm)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def figure_quality(summary: pd.DataFrame, arm: str):
    """Reference alignment, judged quality, and abstention behaviour."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    # BLEU-1 and BLEU-4 side by side.
    ax = axes[0, 0]
    positions = np.arange(len(summary))
    width = 0.38
    for offset, metric, label, color in (
        (-width / 2, "bleu_1", "BLEU-1", ACCENT),
        (width / 2, "bleu_4", "BLEU-4", "#8e44ad"),
    ):
        values = _col(summary, metric)
        if values is None:
            continue
        errors = _col(summary, metric, "ci")
        ax.bar(positions + offset, values, width, yerr=errors, capsize=2.5,
               label=label, color=color, edgecolor="black", linewidth=0.3)
    ax.set_xticks(positions)
    ax.set_xticklabels(_labels(summary["mode"]), fontsize=7.5)
    ax.set_title("BLEU vs Gold Standard", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("score", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    # ROUGE family grouped.
    ax = axes[0, 1]
    width = 0.26
    for offset, metric, label, color in (
        (-width, "rouge_1_f", "ROUGE-1", "#16a085"),
        (0.0, "rouge_2_f", "ROUGE-2", "#f39c12"),
        (width, "rouge_l_f", "ROUGE-L", "#2c3e50"),
    ):
        values = _col(summary, metric)
        if values is None:
            continue
        errors = _col(summary, metric, "ci")
        ax.bar(positions + offset, values, width, yerr=errors, capsize=2,
               label=label, color=color, edgecolor="black", linewidth=0.3)
    ax.set_xticks(positions)
    ax.set_xticklabels(_labels(summary["mode"]), fontsize=7.5)
    ax.set_title("ROUGE F1 vs Gold Standard", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("F1", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    _bar(axes[1, 0], summary, "judge_average", "LLM-Judge Average (1-5)", "score")
    _bar(axes[1, 1], summary, "abstained_flag", "Abstention Rate", "% of queries", scale=100.0)

    fig.suptitle(
        f"Answer quality by topology ({ARM_LABELS.get(arm, arm)} arm; bars are 95% CIs)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def figure_tradeoff(summary: pd.DataFrame, arm: str):
    """The central figure: does a more expensive structure buy a better answer?"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    for ax, (quality_metric, quality_label) in zip(
        axes, (("judge_average", "LLM-judge average (1-5)"), ("rouge_l_f", "ROUGE-L F1"))
    ):
        x = _col(summary, "elapsed_s")
        y = _col(summary, quality_metric)
        if x is None or y is None:
            ax.set_axis_off()
            continue
        x_err = _col(summary, "elapsed_s", "ci")
        y_err = _col(summary, quality_metric, "ci")

        for i, mode in enumerate(summary["mode"]):
            ax.errorbar(
                x.iloc[i],
                y.iloc[i],
                xerr=None if x_err is None else x_err.iloc[i],
                yerr=None if y_err is None else y_err.iloc[i],
                fmt="o",
                markersize=8,
                color=MODE_COLORS.get(mode, FALLBACK_COLOR),
                ecolor="grey",
                elinewidth=0.8,
                capsize=2.5,
                zorder=3,
            )
            ax.annotate(
                MODE_LABELS.get(mode, mode).replace("\n", " "),
                (x.iloc[i], y.iloc[i]),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=7,
            )

        ax.set_xlabel("End-to-end latency (s)  -->  more coordination", fontsize=8)
        ax.set_ylabel(quality_label, fontsize=8)
        ax.set_title(f"{quality_label} vs latency", fontsize=9.5, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

        # Shade the region DOMINATED by the best-scoring topology: slower than it
        # and no better. With SINGLE gone there is no external control to
        # measure against, so the reference is the empirical best on this metric -
        # chosen from the data, not hardcoded, and named in the annotation so the
        # reader knows which point defines the frontier.
        finite = y.notna() & x.notna()
        if finite.any():
            best_i = int(y[finite].idxmax())
            bx, by = float(x.loc[best_i]), float(y.loc[best_i])
            best_mode = summary["mode"].iloc[best_i]
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            if ylim[1] > ylim[0]:
                ax.axvspan(
                    bx,
                    xlim[1],
                    ymin=0,
                    ymax=max(0.0, min(1.0, (by - ylim[0]) / (ylim[1] - ylim[0]))),
                    color=MODE_COLORS.get(best_mode, FALLBACK_COLOR),
                    alpha=0.07,
                    zorder=0,
                )
            ax.axvline(bx, color=MODE_COLORS.get(best_mode, FALLBACK_COLOR),
                       linestyle=":", linewidth=0.9)
            ax.axhline(by, color=MODE_COLORS.get(best_mode, FALLBACK_COLOR),
                       linestyle=":", linewidth=0.9)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.annotate(
                f"dominated by {MODE_LABELS.get(best_mode, best_mode).replace(chr(10), ' ')}",
                (bx, ylim[0]),
                textcoords="offset points",
                xytext=(6, 8),
                fontsize=6.5,
                color="grey",
            )

    fig.suptitle(
        "Cost of coordination: shaded region = slower than the best topology "
        f"without being better ({ARM_LABELS.get(arm, arm)} arm)",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def figure_query_type(by_type: pd.DataFrame, arm: str):
    """H2: the interaction between task decomposability and topology."""
    metric = "judge_average" if "judge_average" in by_type.columns else "rouge_l_f"
    type_order = [t for t in ("simple", "decomposable", "routing") if t in set(by_type["query_type"])]
    type_order += sorted(set(by_type["query_type"]) - set(type_order))

    modes = [m for m in MODE_ORDER if m in set(by_type["mode"])]
    modes += sorted(set(by_type["mode"]) - set(MODE_ORDER))
    fig, ax = plt.subplots(figsize=(11, 4.4))

    positions = np.arange(len(type_order))
    width = 0.8 / max(len(modes), 1)

    for i, mode in enumerate(modes):
        subset = by_type[by_type["mode"] == mode].set_index("query_type")
        values = [float(subset.loc[t, metric]) if t in subset.index else np.nan for t in type_order]
        ax.bar(
            positions + i * width - 0.4 + width / 2,
            values,
            width,
            label=MODE_LABELS.get(mode, mode).replace("\n", " "),
            color=MODE_COLORS.get(mode, FALLBACK_COLOR),
            edgecolor="black",
            linewidth=0.3,
        )

    counts = _query_type_counts()
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{t}\n(n={counts[t]} queries)" if t in counts else str(t) for t in type_order],
        fontsize=8.5,
    )
    ax.set_ylabel(metric.replace("_", " "), fontsize=8)
    ax.set_title(
        "Where coordination structure should matter most: quality by task type "
        f"({ARM_LABELS.get(arm, arm)} arm)",
        fontsize=10,
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


def figure_ablation(summary: pd.DataFrame):
    """RQ2: did PEFT specialisation itself help, holding topology fixed?

    Returns None when the run only covered one arm - drawing a one-armed
    "ablation" would imply a control that was never run.
    """
    arms = [a for a in ("peft", "base") if a in set(summary["arm"])]
    if len(arms) < 2:
        print("[figures] only one arm in the summary; skipping fig5_ablation")
        return None

    modes = [m for m in MODE_ORDER if m in set(summary["mode"])]
    if not modes:
        print("[figures] no compared topologies in the summary; skipping fig5_ablation")
        return None

    panels = [
        ("judge_average", "LLM-Judge Average (1-5)", "score", 1.0),
        ("rouge_l_f", "ROUGE-L F1 vs Gold", "F1", 1.0),
        ("abstained_flag", "Abstention Rate", "% of queries", 100.0),
        ("elapsed_s", "End-to-End Latency", "seconds", 1.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    positions = np.arange(len(modes))
    width = 0.36

    for ax, (metric, title, ylabel, scale) in zip(axes.flat, panels):
        drew = False
        for offset, arm in zip((-width / 2, width / 2), arms):
            arm_rows = summary[summary["arm"] == arm].set_index("mode")
            mean_col, ci_col = f"{metric}_mean", f"{metric}_ci"
            if mean_col not in arm_rows.columns:
                continue
            values = [
                float(arm_rows.loc[m, mean_col]) * scale if m in arm_rows.index else np.nan
                for m in modes
            ]
            errors = (
                [
                    float(arm_rows.loc[m, ci_col]) * scale if m in arm_rows.index else np.nan
                    for m in modes
                ]
                if ci_col in arm_rows.columns
                else None
            )
            ax.bar(
                positions + offset,
                values,
                width,
                yerr=errors,
                capsize=3,
                label=ARM_LABELS.get(arm, arm),
                color=ARM_COLORS.get(arm, FALLBACK_COLOR),
                edgecolor="black",
                linewidth=0.35,
            )
            for x, value in zip(positions + offset, values):
                if pd.notna(value):
                    ax.text(x, value, f"{value:,.2f}".rstrip("0").rstrip("."),
                            ha="center", va="bottom", fontsize=6.5)
            drew = True

        if not drew:
            ax.set_axis_off()
            ax.set_title(f"{title}\n(not available)", fontsize=9)
            continue

        ax.set_xticks(positions)
        ax.set_xticklabels(_labels(modes), fontsize=7.5)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

    fig.suptitle(
        "RQ2 ablation: LoRA-specialised experts vs the identical untuned base models",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--arm",
        default="peft",
        help="Which arm figures 1-4 describe (default: peft). Figure 5 always "
        "uses both arms, and is skipped if only one was benchmarked.",
    )
    args = parser.parse_args()

    full_summary = _load(SUMMARY_FILE)
    full_by_type = _load(BY_TYPE_FILE)

    arm_summary = _ordered(_select_arm(full_summary, args.arm))
    arm_by_type = _select_arm(full_by_type, args.arm)
    plotted_arm = arm_summary["arm"].iloc[0] if not arm_summary.empty else args.arm

    OUT_DIR.mkdir(exist_ok=True)

    outputs = {
        "fig1_operational.png": figure_operational(arm_summary, plotted_arm),
        "fig2_quality.png": figure_quality(arm_summary, plotted_arm),
        "fig3_tradeoff.png": figure_tradeoff(arm_summary, plotted_arm),
        "fig4_query_type.png": figure_query_type(arm_by_type, plotted_arm),
        "fig5_ablation.png": figure_ablation(full_summary),
    }

    written = 0
    for name, fig in outputs.items():
        if fig is None:
            continue
        path = OUT_DIR / name
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        size_kb = path.stat().st_size / 1024
        print(f"[figures] wrote {path.name} ({size_kb:.0f} KB)")
        written += 1

    print(
        f"[figures] {written} figures in {OUT_DIR} (arm='{plotted_arm}' for 1-4). "
        "Upload them to the Overleaf 'figures/' folder."
    )


if __name__ == "__main__":
    main()
