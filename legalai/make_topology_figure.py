"""Draw the topology diagram (fig0) for the paper.

The paper's independent variable is structure, so the reader needs to see the
structures. This renders all eight routing topologies as small node-and-arrow
graphs in one PNG, plus a panel illustrating abstention propagation -- the failure
mode in which one abstaining expert silences an entire ensemble.

Depends on nothing but matplotlib: no run data required, so it can be built
before the benchmark finishes.

Usage:
    python make_topology_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "paper_figures"
DPI = 200

SHARED_COLOR = "#d5dbdb"      # nodes every topology runs
EXPERT_COLOR = "#aed6f1"      # domain experts
SKIPPED_COLOR = "#f2f3f4"     # nodes bypassed by this topology
# Highlights the three topologies this study compares. Was COMPARED_EDGE,
# marking the single-agent baseline; SINGLE is no longer a baseline, so the
# emphasis moves to the compared set (ALL / PARALLEL / DAG).
COMPARED_EDGE = "#c0392b"
ABSTAIN_COLOR = "#f5b7b1"

# Draw the five implemented-but-unevaluated topologies alongside the three
# compared ones. See the note in main() for why this is off.
INCLUDE_OUT_OF_SCOPE = False

# Each topology: list of (label, column, row, kind) plus the edges between them.
# Columns are laid out left to right; rows stack the parallel branches.
TOPOLOGIES = [
    (
        "SINGLE: exactly one expert (out of scope)",
        [("R", 0, 0, "shared"), ("Retr", 1, 0, "shared"), ("E1", 2, 0, "expert"),
         ("Agg", 3, 0, "shared"), ("V", 4, 0, "shared")],
        [("R", "Retr"), ("Retr", "E1"), ("E1", "Agg"), ("Agg", "V")],
        False,
    ),
    (
        "ALL (compared): fully sequential chain",
        [("R", 0, 0, "shared"), ("Retr", 1, 0, "shared"), ("Legal", 2, 0, "expert"),
         ("News", 3, 0, "expert"), ("GQA", 4, 0, "expert"), ("Agg", 5, 0, "shared")],
        [("R", "Retr"), ("Retr", "Legal"), ("Legal", "News"), ("News", "GQA"), ("GQA", "Agg")],
        True,
    ),
    (
        "PARALLEL (compared): full fan-out, then aggregate",
        [("R", 0, 0, "shared"), ("Retr", 1, 0, "shared"), ("Legal", 2, 1, "expert"),
         ("News", 2, 0, "expert"), ("GQA", 2, -1, "expert"), ("Agg", 3, 0, "shared")],
        [("R", "Retr"), ("Retr", "Legal"), ("Retr", "News"), ("Retr", "GQA"),
         ("Legal", "Agg"), ("News", "Agg"), ("GQA", "Agg")],
        True,
    ),
    (
        "LEGAL-NEWS: partial parallelism (out of scope)",
        [("R", 0, 0, "shared"), ("Retr", 1, 0, "shared"), ("Legal", 2, 0.6, "expert"),
         ("News", 2, -0.6, "expert"), ("GQA", 2, -1.8, "skipped"), ("Agg", 3, 0, "shared")],
        [("R", "Retr"), ("Retr", "Legal"), ("Retr", "News"), ("Legal", "Agg"), ("News", "Agg")],
        False,
    ),
    (
        "LEGAL-FIRST: conditional branch (out of scope)",
        [("R", 0, 0, "shared"), ("Retr", 1, 0, "shared"), ("Legal", 2, 0, "expert"),
         ("?", 3, 0, "branch"), ("News", 4, 0.8, "expert"), ("Agg", 5, 0, "shared")],
        [("R", "Retr"), ("Retr", "Legal"), ("Legal", "?"), ("?", "News"), ("?", "Agg"),
         ("News", "Agg")],
        False,
    ),
    (
        "PLANNER: LLM-chosen expert set (out of scope)",
        [("Plan", 0, 0, "branch"), ("R", 1, 0, "shared"), ("Retr", 2, 0, "shared"),
         ("E?", 3, 0.7, "expert"), ("E?", 3, -0.7, "expert"), ("Agg", 4, 0, "shared")],
        [("Plan", "R"), ("R", "Retr"), ("Retr", "E?"), ("Retr", "E?_2"),
         ("E?", "Agg"), ("E?_2", "Agg")],
        False,
    ),
    (
        "DAG (compared): parallel then convergent",
        [("R", 0, 0, "shared"), ("Retr", 1, 0, "shared"), ("Legal", 2, 0.7, "expert"),
         ("News", 2, -0.7, "expert"), ("GQA", 3, 0, "expert"), ("Agg", 4, 0, "shared")],
        [("R", "Retr"), ("Retr", "Legal"), ("Retr", "News"), ("Legal", "GQA"),
         ("News", "GQA"), ("GQA", "Agg")],
        True,
    ),
    (
        "VERIFY-ONLY: experts bypassed (out of scope)",
        [("R", 0, 0, "shared"), ("Retr", 1, 0, "shared"), ("experts", 2, -1.2, "skipped"),
         ("Agg", 3, 0, "shared"), ("V", 4, 0, "shared")],
        [("R", "Retr"), ("Retr", "Agg"), ("Agg", "V")],
        False,
    ),
]

NODE_W = 0.74
NODE_H = 0.42


def _draw_node(ax, x, y, label, kind, highlight=False):
    colors = {
        "shared": SHARED_COLOR,
        "expert": EXPERT_COLOR,
        "skipped": SKIPPED_COLOR,
        "branch": "#fdebd0",
        "abstain": ABSTAIN_COLOR,
    }
    face = colors.get(kind, SHARED_COLOR)
    style = "round,pad=0.02"
    box = patches.FancyBboxPatch(
        (x - NODE_W / 2, y - NODE_H / 2),
        NODE_W,
        NODE_H,
        boxstyle=style,
        linewidth=1.6 if highlight else 0.8,
        edgecolor=COMPARED_EDGE if highlight else "#5d6d7e",
        facecolor=face,
        linestyle="--" if kind == "skipped" else "-",
        alpha=0.55 if kind == "skipped" else 1.0,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=6.4,
        color="#7f8c8d" if kind == "skipped" else "black",
        zorder=4,
        fontstyle="italic" if kind == "skipped" else "normal",
    )


def _draw_edge(ax, x1, y1, x2, y2, color="#34495e", style="-"):
    ax.annotate(
        "",
        xy=(x2 - NODE_W / 2 - 0.03, y2),
        xytext=(x1 + NODE_W / 2 + 0.03, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            linewidth=0.9,
            color=color,
            linestyle=style,
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=0.0" if abs(y1 - y2) < 0.05 else "arc3,rad=0.12",
        ),
        zorder=2,
    )


def _panel(ax, title, nodes, edges, is_compared):
    positions = {}
    counts = {}
    for label, col, row, kind in nodes:
        key = label
        if key in positions:  # disambiguate repeated labels (e.g. two "E?")
            counts[key] = counts.get(key, 1) + 1
            key = f"{label}_{counts[key]}"
        positions[key] = (col, row, kind, label)

    for key, (col, row, kind, label) in positions.items():
        _draw_node(ax, col, row, label, kind, highlight=is_compared and kind == "expert")

    for source, target in edges:
        if source not in positions or target not in positions:
            continue
        sx, sy, _, _ = positions[source]
        tx, ty, _, _ = positions[target]
        _draw_edge(ax, sx, sy, tx, ty)

    cols = [col for col, _, _, _ in positions.values()]
    rows = [row for _, row, _, _ in positions.values()]
    ax.set_xlim(min(cols) - 0.7, max(cols) + 0.7)
    ax.set_ylim(min(rows) - 0.8, max(rows) + 0.8)
    ax.set_title(
        title,
        fontsize=7.6,
        fontweight="bold" if is_compared else "normal",
        color=COMPARED_EDGE if is_compared else "black",
        pad=4,
    )
    ax.set_axis_off()


def _abstention_panel(ax):
    """Illustrate abstention propagation: the veto rule vs the corrected rule."""
    ax.set_axis_off()
    ax.set_title(
        "Abstention propagation (structural failure mode)",
        fontsize=7.6,
        fontweight="bold",
        pad=4,
    )

    # Top row: naive veto rule.
    _draw_node(ax, 0, 0.75, "Legal\nabstains", "abstain")
    _draw_node(ax, 0, -0.05, "News\nanswers", "expert")
    _draw_node(ax, 1.15, 0.35, "Agg", "shared")
    _draw_node(ax, 2.35, 0.35, "ABSTAIN", "abstain")
    _draw_edge(ax, 0, 0.75, 1.15, 0.35, color=COMPARED_EDGE)
    _draw_edge(ax, 0, -0.05, 1.15, 0.35, color="#95a5a6", style=":")
    _draw_edge(ax, 1.15, 0.35, 2.35, 0.35, color=COMPARED_EDGE)
    ax.text(3.0, 0.35, "any-abstain veto:\nNews discarded", fontsize=6, va="center",
            color=COMPARED_EDGE)

    # Bottom row: corrected rule.
    _draw_node(ax, 0, -1.35, "Legal\nabstains", "abstain")
    _draw_node(ax, 0, -2.15, "News\nanswers", "expert")
    _draw_node(ax, 1.15, -1.75, "Agg", "shared")
    _draw_node(ax, 2.35, -1.75, "ANSWER", "expert")
    _draw_edge(ax, 0, -1.35, 1.15, -1.75, color="#95a5a6", style=":")
    _draw_edge(ax, 0, -2.15, 1.15, -1.75, color="#1e8449")
    _draw_edge(ax, 1.15, -1.75, 2.35, -1.75, color="#1e8449")
    ax.text(3.0, -1.75, "all-abstain rule:\nabstention recorded,\nanswer preserved",
            fontsize=6, va="center", color="#1e8449")

    ax.set_xlim(-0.7, 4.6)
    ax.set_ylim(-2.8, 1.3)


def main():
    # This is the paper's first figure, so it should show the experiment rather
    # than the implementation surface. Drawing all nine panels spent two thirds
    # of the area on configurations the study never evaluates and left each
    # diagram cramped; the five out-of-scope topologies are argued in prose in
    # the "Configurations implemented but out of scope" section, which makes the
    # SINGLE exclusion case better than a diagram can. Set this True to restore
    # the full nine-panel version -- the definitions above are untouched.
    if INCLUDE_OUT_OF_SCOPE:
        selected = list(TOPOLOGIES)
        fig, axes = plt.subplots(3, 3, figsize=(11, 6.4))
        flat = list(axes.flatten())
        panel_axes, abstain_ax = flat[:len(selected)], flat[len(selected)]
        for ax in flat[len(selected) + 1:]:
            ax.axis("off")
    else:
        selected = [t for t in TOPOLOGIES if t[3]]
        # 2x2 rather than one row of three: the panel titles are long enough that
        # three across collide, and a 2x2 cell is wide enough for each to sit on
        # one line. The fourth cell takes the abstention diagram.
        fig, axes = plt.subplots(2, 2, figsize=(10, 5.2))
        flat = list(axes.flatten())
        panel_axes, abstain_ax = flat[:3], flat[3]

    for ax, (title, nodes, edges, is_compared) in zip(panel_axes, selected):
        _panel(ax, title, nodes, edges, is_compared)

    _abstention_panel(abstain_ax)

    # Only advertise node kinds that the drawn panels actually contain: the three
    # compared topologies use no conditional branch and bypass no node, so those
    # two entries would be legend items with nothing to point at.
    kinds = {kind for _, nodes, _, _ in selected for *_, kind in nodes}
    legend_handles = [
        patches.Patch(facecolor=SHARED_COLOR, edgecolor="#5d6d7e", label="shared node (router R, retrieval Retr, aggregator Agg, validator V)"),
        patches.Patch(facecolor=EXPERT_COLOR, edgecolor="#5d6d7e", label="domain expert (Legal / News / General QA)"),
    ]
    if "branch" in kinds:
        legend_handles.append(patches.Patch(facecolor="#fdebd0", edgecolor="#5d6d7e", label="planner or conditional branch"))
    if "skipped" in kinds:
        legend_handles.append(patches.Patch(facecolor=SKIPPED_COLOR, edgecolor="#5d6d7e", linestyle="--", label="node bypassed by this topology"))
    legend_handles.append(patches.Patch(facecolor=ABSTAIN_COLOR, edgecolor="#5d6d7e", label="abstaining expert / abstention output"))

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )

    if INCLUDE_OUT_OF_SCOPE:
        title = (
            "Routing topologies. The three marked (compared) are evaluated in this study; the rest remain "
            "implemented but out of scope. All share one graph and one retrieval step; "
            "only the wiring between experts differs."
        )
    else:
        # Deliberately short. The LaTeX \caption carries the full explanation, and
        # a long single-line suptitle became the widest artist in the figure, so
        # bbox_inches="tight" cropped to the text rather than the diagrams and the
        # whole figure rendered as a 4:1 strip barely 1.8in tall at \textwidth.
        title = "Coordination topologies compared, and the abstention rule for merging expert output"
    fig.suptitle(title, fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / "fig0_topologies.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {path.name} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
