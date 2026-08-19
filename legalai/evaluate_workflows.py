"""Evaluation script to generate publication-grade charts with 95% CI error bars."""

import os
import re
import math
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Use a professional, clean grid theme for all figures
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.family": "sans-serif"
})

CONFIG_COLORS = {
    "all": "#1A365D",                  # Deep Navy
    "single": "#0D9488",               # Teal
    "parallel": "#7C3AED",             # Violet
    "legal_first": "#D97706",          # Amber
    "verify_only": "#E11D48",          # Crimson
    "planner_based": "#2563EB",         # Blue
    "graph_engineering": "#059669",    # Emerald Green
    "graph": "#059669",
    "dag": "#059669"
}

# --- Lexical similarity functions (reused by analysis) ---

def custom_tokenize(text):
    if not isinstance(text, str):
        text = str(text or "")
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return cleaned.split()

def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def calculate_bleu_n(ref_tokens, cand_tokens, n):
    if len(cand_tokens) < n:
        return 0.0
    ref_ngrams = get_ngrams(ref_tokens, n)
    cand_ngrams = get_ngrams(cand_tokens, n)
    if not ref_ngrams or not cand_ngrams:
        return 0.0
    ref_counts = Counter(ref_ngrams)
    cand_counts = Counter(cand_ngrams)
    overlap = sum(min(cand_counts[ng], ref_counts[ng]) for ng in cand_counts)
    return overlap / len(cand_ngrams)

def calculate_bleu(ref_text, cand_text):
    ref_tokens = custom_tokenize(ref_text)
    cand_tokens = custom_tokenize(cand_text)
    if not ref_tokens or not cand_tokens:
        return [0.0] * 4
    
    p1 = calculate_bleu_n(ref_tokens, cand_tokens, 1)
    p2 = calculate_bleu_n(ref_tokens, cand_tokens, 2)
    p3 = calculate_bleu_n(ref_tokens, cand_tokens, 3)
    p4 = calculate_bleu_n(ref_tokens, cand_tokens, 4)
    
    c = len(cand_tokens)
    r = len(ref_tokens)
    bp = 1.0 if c > r else math.exp(1 - r / c) if c > 0 else 0.0
    
    bleu1 = bp * p1
    bleu2 = bp * math.sqrt(p1 * p2) if p1 * p2 > 0 else 0.0
    bleu3 = bp * (p1 * p2 * p3) ** (1/3) if p1 * p2 * p3 > 0 else 0.0
    bleu4 = bp * (p1 * p2 * p3 * p4) ** 0.25 if p1 * p2 * p3 * p4 > 0 else 0.0
    
    return [bleu1, bleu2, bleu3, bleu4]

def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]

def calculate_rouge_l(ref_text, cand_text):
    ref_tokens = custom_tokenize(ref_text)
    cand_tokens = custom_tokenize(cand_text)
    if not ref_tokens or not cand_tokens:
        return 0.0, 0.0, 0.0
    lcs_len = lcs(ref_tokens, cand_tokens)
    r = lcs_len / len(ref_tokens)
    p = lcs_len / len(cand_tokens)
    f1 = (2 * r * p) / (r + p) if (r + p) > 0 else 0.0
    return p, r, f1

def calculate_rouge_n(ref_text, cand_text, n=1):
    ref_tokens = custom_tokenize(ref_text)
    cand_tokens = custom_tokenize(cand_text)
    if len(ref_tokens) < n or len(cand_tokens) < n:
        return 0.0, 0.0, 0.0
    ref_ngrams = get_ngrams(ref_tokens, n)
    cand_ngrams = get_ngrams(cand_tokens, n)
    ref_counts = Counter(ref_ngrams)
    cand_counts = Counter(cand_ngrams)
    overlap = sum(min(cand_counts[ng], ref_counts[ng]) for ng in cand_counts)
    r = overlap / len(ref_ngrams)
    p = overlap / len(cand_ngrams) if len(cand_ngrams) > 0 else 0.0
    f1 = (2 * r * p) / (r + p) if (r + p) > 0 else 0.0
    return p, r, f1

def word_jaccard(ref_text, cand_text):
    set1 = set(custom_tokenize(ref_text))
    set2 = set(custom_tokenize(cand_text))
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def char_jaccard(ref_text, cand_text):
    set1 = set(ref_text.lower().replace(" ", ""))
    set2 = set(cand_text.lower().replace(" ", ""))
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def cosine_similarity_tf(ref_text, cand_text):
    tokens1 = custom_tokenize(ref_text)
    tokens2 = custom_tokenize(cand_text)
    counter1 = Counter(tokens1)
    counter2 = Counter(tokens2)
    all_words = set(counter1.keys()).union(set(counter2.keys()))
    dot_product = sum(counter1[w] * counter2[w] for w in all_words)
    mag1 = math.sqrt(sum(val**2 for val in counter1.values()))
    mag2 = math.sqrt(sum(val**2 for val in counter2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)

def levenshtein_similarity(ref_text, cand_text):
    s1, s2 = ref_text.lower(), cand_text.lower()
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return 1.0 if len(s1) == 0 else 0.0
        
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    distance = previous_row[-1]
    return 1.0 - (distance / max(len(s1), len(s2)))


# --- Plotting Helpers ---

def save_chart(name):
    """Save chart to evaluation_assets directory and close the plot."""
    os.makedirs("evaluation_assets", exist_ok=True)
    plt.tight_layout()
    path = os.path.join("evaluation_assets", name)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[evaluate] Saved {path}")


def plot_bar_with_ci(df, metric, ylabel, title, filename, fmt="{:.2f}", is_int=False, suffix="", scale=1.0):
    """Plot a single metric bar chart with 95% Confidence Interval error bars."""
    plt.figure(figsize=(8, 5))
    means = df[f"{metric}_mean"] * scale
    cis = df[f"{metric}_ci"] * scale
    
    ax = plt.gca()
    bars = ax.bar(
        df.index,
        means,
        yerr=cis,
        color=[CONFIG_COLORS.get(m, "#555555") for m in df.index],
        capsize=5,
        edgecolor="black",
        alpha=0.9
    )
    
    plt.ylabel(ylabel)
    plt.xlabel("Routing Workflow Configuration")
    plt.title(title)
    plt.xticks(rotation=15)
    
    # Annotate heights
    for bar in bars:
        height = bar.get_height()
        val_str = fmt.format(height)
        if is_int:
            val_str = str(int(round(height)))
        ax.annotate(
            f"{val_str}{suffix}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),  # 5 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )
    save_chart(filename)


def main():
    # Load aggregated means
    summary_path = "analysis_summary.csv"
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found. Run analyze_results.py first.")
        return

    raw = pd.read_csv(summary_path)

    # Since the PEFT pivot analysis_summary.csv has one row per (mode, arm), so
    # indexing on mode alone would produce duplicate index labels and every
    # df.loc[mode, ...] below would return a Series instead of a scalar. The
    # charts describe one arm at a time; the ablation lives in its own table and
    # in make_paper_figures.py's fig5.
    if "arm" not in raw.columns:
        raw["arm"] = "unknown"
    arms_present = sorted(raw["arm"].dropna().unique())
    chart_arm = os.getenv("EVAL_ARM", "peft")
    if chart_arm not in arms_present:
        chart_arm = arms_present[0] if arms_present else "unknown"
        print(f"[eval] arm 'peft' not present; charting arm '{chart_arm}' instead")
    df = raw[raw["arm"] == chart_arm].drop(columns=["arm"]).set_index("mode")
    print(f"Loaded aggregated metrics for arm='{chart_arm}', modes: {list(df.index)}")

    # 1. E2E Latency
    plot_bar_with_ci(df, "elapsed_s", "Latency (seconds)", "Chart 1: End-to-End Latency by Multi-Agent Workflow", "chart1_e2e_latency.png", suffix="s")

    # 2. Backend Latency
    plot_bar_with_ci(df, "backend_ms", "Backend Execution Time (seconds)", "Chart 2: Backend Graph Execution Latency", "chart2_backend_latency.png", scale=0.001, suffix="s")

    # 3. Network Overhead
    plot_bar_with_ci(df, "net_overhead", "Network & Overhead (seconds)", "Chart 3: API & Network Overhead (E2E minus Graph Latency)", "chart3_network_overhead.png", suffix="s")

    # 4. Execution Step Count
    plot_bar_with_ci(df, "steps", "State Transitions (Steps)", "Chart 4: LangGraph Execution Steps Count", "chart4_execution_steps.png", is_int=True)

    # 5. Average Latency per Step
    plot_bar_with_ci(df, "lat_per_step", "Latency per Step (ms)", "Chart 5: Mean Node Execution Latency per Graph Step", "chart5_latency_per_step.png", fmt="{:.1f}", suffix="ms")

    # 6. Generation Speed
    plot_bar_with_ci(df, "words_per_sec", "Generation Throughput (words/sec)", "Chart 6: Answer Generation Speed (Words / Second)", "chart6_generation_speed.png")

    # 7. Token count
    plot_bar_with_ci(df, "completion_tokens", "Total Tokens Used", "Chart 7: Total LLM Tokens Used", "chart7_token_usage.png", is_int=True)

    # 8. Cost
    plot_bar_with_ci(df, "cost", "Operational Cost (USD)", "Chart 8: API Execution Cost", "chart8_run_cost.png", fmt="${:.4f}")

    # 9. Stacked Bar Chart of Node Latencies
    nodes_order = ["planner", "router", "memory", "retrieval", "legal", "news", "general_qa", "aggregator", "validator", "response"]
    stacked_data = []
    for mode in df.index:
        row = {"mode": mode}
        for n in nodes_order:
            row[n] = df.loc[mode, f"timing_{n}_mean"] / 1000.0  # convert to seconds
        stacked_data.append(row)
    df_stacked = pd.DataFrame(stacked_data).set_index("mode")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df_stacked.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", edgecolor="black")
    plt.ylabel("Execution Time (seconds)")
    plt.xlabel("Routing Workflow Configuration")
    plt.title("Chart 9: Detailed Node-level Latency Stacked Breakdown")
    plt.xticks(rotation=15)
    plt.legend(title="Agent Graph Nodes", bbox_to_anchor=(1.04, 1), loc="upper left")
    save_chart("chart9_node_latency_stacked.png")

    # 10-17. Individual Node Latency Charts
    individual_nodes = ["router", "retrieval", "legal", "news", "general_qa", "aggregator", "validator", "response"]
    for i, node_name in enumerate(individual_nodes):
        plot_bar_with_ci(
            df,
            f"timing_{node_name}",
            "Node Latency (ms)",
            f"Chart {10 + i}: {node_name.replace('_', ' ').capitalize()} Node Latency Profile",
            f"chart{10 + i}_{node_name}_latency.png",
            fmt="{:.1f}",
            suffix="ms"
        )

    # 18. BLEU-1 and BLEU-4
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df.index))
    width = 0.35
    plt.bar(x - width/2, df["bleu_1_mean"], yerr=df["bleu_1_ci"], width=width, label="BLEU-1", color="#3B82F6", capsize=5, edgecolor="black")
    plt.bar(x + width/2, df["bleu_4_mean"], yerr=df["bleu_4_ci"], width=width, label="BLEU-4", color="#1D4ED8", capsize=5, edgecolor="black")
    plt.xticks(x, df.index, rotation=15)
    plt.ylabel("BLEU Metric Score")
    plt.xlabel("Routing Workflow Configuration")
    plt.title("Chart 18: BLEU Reference-Matching Quality Profile")
    plt.ylim(0, 1.0)
    plt.legend()
    save_chart("chart18_bleu_scores.png")

    # 19. ROUGE-1
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, df["rouge_1_p_mean"], yerr=df["rouge_1_p_ci"], width=width/1.5, label="Precision", color="#C084FC", capsize=4, edgecolor="black")
    plt.bar(x, df["rouge_1_r_mean"], yerr=df["rouge_1_r_ci"], width=width/1.5, label="Recall", color="#A855F7", capsize=4, edgecolor="black")
    plt.bar(x + width, df["rouge_1_f_mean"], yerr=df["rouge_1_f_ci"], width=width/1.5, label="F1-Score", color="#6B21A8", capsize=4, edgecolor="black")
    plt.xticks(x, df.index, rotation=15)
    plt.ylabel("ROUGE-1 Score")
    plt.xlabel("Routing Workflow Configuration")
    plt.title("Chart 19: ROUGE-1 Token-Overlap Analysis")
    plt.ylim(0, 1.0)
    plt.legend()
    save_chart("chart19_rouge1_scores.png")

    # 20. ROUGE-2
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, df["rouge_2_p_mean"], yerr=df["rouge_2_p_ci"], width=width/1.5, label="Precision", color="#FDBA74", capsize=4, edgecolor="black")
    plt.bar(x, df["rouge_2_r_mean"], yerr=df["rouge_2_r_ci"], width=width/1.5, label="Recall", color="#F97316", capsize=4, edgecolor="black")
    plt.bar(x + width, df["rouge_2_f_mean"], yerr=df["rouge_2_f_ci"], width=width/1.5, label="F1-Score", color="#C2410C", capsize=4, edgecolor="black")
    plt.xticks(x, df.index, rotation=15)
    plt.ylabel("ROUGE-2 Score")
    plt.xlabel("Routing Workflow Configuration")
    plt.title("Chart 20: ROUGE-2 Bigram-Overlap Analysis")
    plt.ylim(0, 1.0)
    plt.legend()
    save_chart("chart20_rouge2_scores.png")

    # 21. ROUGE-L
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, df["rouge_l_p_mean"], yerr=df["rouge_l_p_ci"], width=width/1.5, label="Precision", color="#86EFAC", capsize=4, edgecolor="black")
    plt.bar(x, df["rouge_l_r_mean"], yerr=df["rouge_l_r_ci"], width=width/1.5, label="Recall", color="#22C55E", capsize=4, edgecolor="black")
    plt.bar(x + width, df["rouge_l_f_mean"], yerr=df["rouge_l_f_ci"], width=width/1.5, label="F1-Score", color="#15803D", capsize=4, edgecolor="black")
    plt.xticks(x, df.index, rotation=15)
    plt.ylabel("ROUGE-L Score")
    plt.xlabel("Routing Workflow Configuration")
    plt.title("Chart 21: ROUGE-L Longest Common Subsequence Analysis")
    plt.ylim(0, 1.0)
    plt.legend()
    save_chart("chart21_rougeL_scores.png")

    # 22. Jaccard Word vs Character
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, df["jaccard_mean"], yerr=df["jaccard_ci"], width=width, label="Word Jaccard", color="#F59E0B", capsize=5, edgecolor="black")
    plt.bar(x + width/2, df["char_jaccard_mean"], yerr=df["char_jaccard_ci"], width=width, label="Char Jaccard", color="#EF4444", capsize=5, edgecolor="black")
    plt.xticks(x, df.index, rotation=15)
    plt.ylabel("Jaccard Index (Set Overlap)")
    plt.xlabel("Routing Workflow Configuration")
    plt.title("Chart 22: Word-level vs. Character-level Jaccard Vocabulary Overlap")
    plt.ylim(0, 1.0)
    plt.legend()
    save_chart("chart22_jaccard_similarity.png")

    # 23. Cosine Similarity
    plot_bar_with_ci(df, "cosine", "TF Cosine Similarity", "Chart 23: Term-Frequency Cosine Similarity to Gold Standard", "chart23_cosine_similarity.png")

    # 24. Normalized Levenshtein Distance
    plot_bar_with_ci(df, "levenshtein", "Normalized Levenshtein Similarity", "Chart 24: Text Character-Sequence Edit-Distance Similarity", "chart24_levenshtein_similarity.png")

    # 25. Word vs Sentence Counts
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = "#4C72B0"
    ax1.set_xlabel("Routing Workflow Configuration")
    ax1.set_ylabel("Word Count (Verbosity)", color=color)
    ax1.bar(df.index, df["word_count_mean"], yerr=df["word_count_ci"], color=color, alpha=0.6, capsize=5, edgecolor="black")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(False)
    
    ax2 = ax1.twinx()
    color = "#C44E52"
    ax2.set_ylabel("Sentence Count", color=color)
    ax2.plot(df.index, df["sentence_count_mean"], color=color, marker="o", linewidth=2.5)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.grid(False)
    plt.xticks(rotation=15)
    plt.title("Chart 25: Word Count vs. Sentence Count Profiling")
    save_chart("chart25_word_sentence_counts.png")

    # 26. Lexical Diversity
    plot_bar_with_ci(df, "ttr", "Type-Token Ratio (Unique / Total Words)", "Chart 26: Lexical Diversity Profile (TTR)", "chart26_lexical_diversity.png")

    # 27. Temporal References Count
    plot_bar_with_ci(df, "temporal_refs", "Temporal Reference Count (Dates/Months)", "Chart 27: Recency / Temporal Reference Density", "chart27_temporal_references.png", is_int=True)

    # 28. Quality vs Latency Trade-off Scatter Plot with Error Bar Whiskers on Both Axes
    plt.figure(figsize=(9, 6))
    for m in df.index:
        plt.errorbar(
            df.loc[m, "elapsed_s_mean"],
            df.loc[m, "rouge_l_mean"],
            xerr=df.loc[m, "elapsed_s_ci"],
            yerr=df.loc[m, "rouge_l_ci"],
            fmt="o",
            color=CONFIG_COLORS.get(m, "#555555"),
            markersize=12,
            capsize=5,
            markeredgecolor='black',
            markeredgewidth=1.5,
            alpha=0.85,
            label=m
        )
        plt.text(df.loc[m, "elapsed_s_mean"] + 0.2, df.loc[m, "rouge_l_mean"], m.upper(), fontsize=9, fontweight='bold')
    plt.xlabel("End-to-End Latency (seconds)")
    plt.ylabel("ROUGE-L F1-Score (Overlap Quality)")
    plt.title("Chart 28: Quality (ROUGE-L F1) vs. Speed (Latency) Trade-off Space")
    plt.grid(True, linestyle="--", alpha=0.5)
    save_chart("chart28_quality_vs_latency.png")

    # 29. Radar Chart / Spider Chart
    radar_df = pd.DataFrame(index=df.index)
    radar_df["Speed"] = df["elapsed_s_mean"].max() / df["elapsed_s_mean"] # higher is better
    radar_df["Step Eff"] = df["steps_mean"].max() / df["steps_mean"]       # higher is better
    radar_df["ROUGE-L"] = df["rouge_l_mean"]
    radar_df["Cosine Sim"] = df["cosine_mean"]
    radar_df["Diversity"] = df["ttr_mean"]
    
    labels = list(radar_df.columns)
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for mode in radar_df.index:
        values = radar_df.loc[mode].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=CONFIG_COLORS.get(mode, "#555555"), linewidth=2, label=mode)
        ax.fill(angles, values, color=CONFIG_COLORS.get(mode, "#555555"), alpha=0.15)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], labels, fontsize=10, fontweight="bold")
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    plt.ylim(0, 1.1)
    plt.title("Chart 29: Normalized Multi-Agent Multi-Dimensional Performance Profile", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    save_chart("chart29_performance_radar.png")

    # 30. NEW Chart: LLM Judge Quality Breakdown
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, df["judge_accuracy_mean"], yerr=df["judge_accuracy_ci"], width=width/1.5, label="Accuracy", color="#10B981", capsize=4, edgecolor="black")
    plt.bar(x, df["judge_completeness_mean"], yerr=df["judge_completeness_ci"], width=width/1.5, label="Completeness", color="#3B82F6", capsize=4, edgecolor="black")
    plt.bar(x + width, df["judge_groundedness_mean"], yerr=df["judge_groundedness_ci"], width=width/1.5, label="Groundedness", color="#F59E0B", capsize=4, edgecolor="black")
    plt.xticks(x, df.index, rotation=15)
    plt.ylabel("LLM Judge score (1-5)")
    plt.xlabel("Routing Workflow Configuration")
    plt.title("Chart 30: LLM Judge Multi-Dimensional Correctness Profile")
    plt.ylim(1.0, 5.2)
    plt.legend()
    save_chart("chart30_llm_judge_breakdown.png")

    # 31. NEW Chart: Retrieval Performance Metrics
    # Filter modes that actually have retrieval metrics (some modes might not use retrieval)
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, df["precision_at_5_mean"], yerr=df["precision_at_5_ci"], width=width, label="Precision@5", color="#EC4899", capsize=5, edgecolor="black")
    plt.bar(x + width/2, df["recall_at_5_mean"], yerr=df["recall_at_5_ci"], width=width, label="Recall@5", color="#8B5CF6", capsize=5, edgecolor="black")
    plt.xticks(x, df.index, rotation=15)
    plt.ylabel("Score")
    plt.xlabel("Routing Workflow Configuration")
    plt.title("Chart 31: Document Retrieval Layer Precision vs Recall Performance")
    plt.ylim(0, 1.0)
    plt.legend()
    save_chart("chart31_retrieval_performance.png")

    print("\n--- Generating LaTeX Tables for Overleaf ---")
    latex_code = generate_latex_table(df, arm=chart_arm)
    with open("metrics_table.tex", "w", encoding="utf-8") as f:
        f.write(latex_code)
    print(f"Saved LaTeX table to metrics_table.tex (arm={chart_arm}, {len(df.index)} columns)")

    # Second table: both arms side by side, the RQ2 ablation in numbers. Only
    # written when both arms were actually benchmarked - an "ablation" table with
    # one arm in it would imply a control that never ran.
    if len(arms_present) > 1:
        ablation_code = generate_ablation_table(raw, arms_present)
        with open("metrics_table_ablation.tex", "w", encoding="utf-8") as f:
            f.write(ablation_code)
        print(
            f"Saved ablation table to metrics_table_ablation.tex "
            f"(arms: {', '.join(arms_present)})"
        )
    else:
        print(
            "[eval] only one arm in the summary; metrics_table_ablation.tex not "
            "written. RQ2 needs both arms benchmarked."
        )

    print("\nEvaluation successfully complete!")


MODE_NAMES = {
    "all": "ALL (Sequential)",
    "parallel": "PARALLEL",
    "graph_engineering": "Graph Engineering",
    "graph": "Graph Engineering",
    "dag": "Graph Engineering",
    # Still-implemented but no longer benchmarked topologies, kept so an older
    # or wider run still renders a table.
    "single": "SINGLE (Router)",
    "legal_first": "LEGAL-FIRST",
    "verify_only": "VERIFY-ONLY",
    "planner_based": "PLANNER-BASED",
    "legal_news_parallel": "LEGAL-NEWS",
}
ARM_NAMES = {"peft": "PEFT", "base": "Base"}


def generate_latex_table(df, arm="peft"):
    """Generate LaTeX tabular string for paper publication.

    Three topology columns since the PEFT pivot, which is why this no longer
    needs the rotation/resizebox workaround the seven-column version required.
    """
    col_headers = " & ".join([f"\\textbf{{{MODE_NAMES.get(m, m.upper())}}}" for m in df.index])
    arm_label = ARM_NAMES.get(arm, arm)

    latex = "% Place this in your Overleaf document. Requires \\usepackage{booktabs} and \\usepackage{multirow}\n"
    latex += "\\begin{table*}[t]\n\\centering\n\\caption{Coordination topologies compared across three "
    latex += f"PEFT-specialised expert agents ({arm_label} arm). Values are means over queries and "
    latex += "repeats; $\\pm$ is the 95\\% confidence interval margin.}\n"
    latex += "\\label{tab:agent_workflows_eval}\n\\small\n\\begin{tabular}{l" + "c" * len(df.index) + "}\n\\toprule\n"
    latex += "\\textbf{Metric} & " + col_headers + " \\\\\n\\midrule\n"
    latex += f"\\multicolumn{{{len(df.index) + 1}}}{{l}}{{\\textit{{Operational \\& Latency Metrics}}}} \\\\\n"
    
    def row(label, metric_key, fmt="{:.2f}", scale=1.0, ci_fmt=None):
        """Emit one metric row. `scale` converts units (e.g. ms -> s) for the mean
        *and* the CI margin; a row whose metric is absent is skipped so that older
        summary files still render."""
        mean_col = f"{metric_key}_mean"
        ci_col = f"{metric_key}_ci"
        if mean_col not in df.columns:
            print(f"[latex] skipping '{label}': {mean_col} not in summary")
            return ""

        ci_format = ci_fmt or fmt
        cells = []
        for m in df.index:
            mean_val = df.loc[m, mean_col]
            ci_val = df.loc[m, ci_col] if ci_col in df.columns else float("nan")
            if pd.isna(mean_val):
                cells.append("--")
                continue
            mean_txt = fmt.format(mean_val * scale)
            if pd.isna(ci_val):
                cells.append(mean_txt)
            else:
                cells.append(f"{mean_txt} $\\pm$ {ci_format.format(ci_val * scale)}")
        return f"\\textbf{{{label}}} & " + " & ".join(cells) + " \\\\\n"

    latex += row("End-to-End Latency (s)", "elapsed_s", "{:.2f}")
    # backend_ms is milliseconds; the row is labelled in seconds, so scale it.
    latex += row("Backend Graph Latency (s)", "backend_ms", "{:.2f}", scale=0.001)
    latex += row("Graph Execution Steps", "steps", "{:.1f}")
    latex += row("Mean Step Latency (ms)", "lat_per_step", "{:.1f}")
    latex += row("Generation Speed (words/s)", "words_per_sec", "{:.2f}")
    # Real token counts reported by the model, split by direction (no words x 1.3 estimate).
    latex += row("Prompt Tokens (measured)", "prompt_tokens", "{:.0f}")
    latex += row("Completion Tokens (measured)", "completion_tokens", "{:.0f}")
    latex += row("Execution Cost (USD)", "cost", "\\${:.4f}")
    latex += row("System Abstention Rate (\\%)", "abstained_flag", "{:.1f}", scale=100.0)
    latex += row("Expert Abstention Rate (\\%)", "expert_abstention_rate", "{:.1f}", scale=100.0)
    
    latex += f"\\midrule\n\\multicolumn{{{len(df.index) + 1}}}{{l}}{{\\textit{{Text Quality \\& NLG Reference-Matching (vs. Gold Standard)}}}} \\\\\n"
    latex += row("BLEU-1 Score", "bleu_1", "{:.4f}")
    latex += row("BLEU-4 Score", "bleu_4", "{:.4f}")
    latex += row("ROUGE-1 F1-Score", "rouge_1_f", "{:.4f}")
    latex += row("ROUGE-2 F1-Score", "rouge_2_f", "{:.4f}")
    latex += row("ROUGE-L F1-Score", "rouge_l_f", "{:.4f}")
    latex += row("Word Jaccard Similarity", "jaccard", "{:.4f}")
    latex += row("Character Jaccard Similarity", "char_jaccard", "{:.4f}")
    latex += row("TF Cosine Similarity", "cosine", "{:.4f}")
    latex += row("Levenshtein Similarity", "levenshtein", "{:.4f}")
    
    latex += f"\\midrule\n\\multicolumn{{{len(df.index) + 1}}}{{l}}{{\\textit{{LLM Judge Quality Scopes}}}} \\\\\n"
    latex += row("Judge Accuracy Score", "judge_accuracy", "{:.2f}")
    latex += row("Judge Completeness Score", "judge_completeness", "{:.2f}")
    latex += row("Judge Groundedness Score", "judge_groundedness", "{:.2f}")
    latex += row("Judge Average Score", "judge_average", "{:.2f}")
    
    latex += f"\\midrule\n\\multicolumn{{{len(df.index) + 1}}}{{l}}{{\\textit{{Document Retrieval Layer Performance}}}} \\\\\n"
    latex += row("Retrieval Precision@5", "precision_at_5", "{:.4f}")
    latex += row("Retrieval Recall@5", "recall_at_5", "{:.4f}")
    latex += row("Retrieval MRR", "mrr", "{:.4f}")
    
    latex += f"\\midrule\n\\multicolumn{{{len(df.index) + 1}}}{{l}}{{\\textit{{Text Structural \\& Domain-Specific Metrics}}}} \\\\\n"
    latex += row("Word Count (Verbosity)", "word_count", "{:.1f}")
    latex += row("Sentence Count", "sentence_count", "{:.1f}")
    latex += row("Paragraph Count", "paragraph_count", "{:.1f}")
    latex += row("Lexical Diversity (TTR)", "ttr", "{:.4f}")
    latex += row("Temporal References Count", "temporal_refs", "{:.1f}")
    latex += row("Bullet Points Count", "bullet_points", "{:.1f}")
    
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table*}"
    return latex


def generate_ablation_table(raw, arms_present):
    """RQ2 in numbers: every topology under both arms, side by side.

    Columns are grouped by arm so a reader compares within a topology
    (specialisation effect) by reading across the group boundary, and within an
    arm (topology effect) by reading inside a group. Deliberately a shorter
    metric list than the main table - the point here is the arm contrast, and
    32 rows x 6 columns would not fit a two-column page.
    """
    mode_order = [m for m in ("all", "parallel", "graph_engineering", "graph", "dag") if m in set(raw["mode"])]
    mode_order += sorted(set(raw["mode"]) - set(mode_order))
    arm_order = [a for a in ("peft", "base") if a in arms_present]
    arm_order += [a for a in arms_present if a not in arm_order]

    cells = [(arm, mode) for arm in arm_order for mode in mode_order]
    n_cols = len(cells)

    by_key = {}
    for _, row in raw.iterrows():
        by_key[(row["arm"], row["mode"])] = row

    latex = "% RQ2 ablation table. Requires \\usepackage{booktabs}.\n"
    latex += "\\begin{table*}[t]\n\\centering\n"
    latex += (
        "\\caption{RQ2 ablation: each coordination topology run with LoRA-specialised "
        "experts (PEFT) and with the identical untuned base models (Base). Values are "
        "means over queries and repeats; $\\pm$ is the 95\\% confidence interval margin.}\n"
    )
    latex += "\\label{tab:peft_ablation}\n\\footnotesize\n"
    latex += "\\begin{tabular}{l" + "c" * n_cols + "}\n\\toprule\n"

    # Two header rows: arm group spans, then topology names.
    group_header = "\\textbf{Metric}"
    for arm in arm_order:
        group_header += (
            f" & \\multicolumn{{{len(mode_order)}}}{{c}}{{\\textbf{{{ARM_NAMES.get(arm, arm)}}}}}"
        )
    latex += group_header + " \\\\\n"

    # cmidrule under each arm group
    rules = []
    start = 2
    for arm in arm_order:
        rules.append(f"\\cmidrule(lr){{{start}-{start + len(mode_order) - 1}}}")
        start += len(mode_order)
    latex += "".join(rules) + "\n"

    latex += (
        " & "
        + " & ".join(MODE_NAMES.get(mode, mode.upper()) for _, mode in cells)
        + " \\\\\n\\midrule\n"
    )

    def row(label, metric_key, fmt="{:.2f}", scale=1.0):
        mean_col, ci_col = f"{metric_key}_mean", f"{metric_key}_ci"
        if mean_col not in raw.columns:
            print(f"[latex] ablation: skipping '{label}': {mean_col} not in summary")
            return ""
        rendered = []
        for key in cells:
            source = by_key.get(key)
            if source is None or pd.isna(source.get(mean_col)):
                rendered.append("--")
                continue
            mean_txt = fmt.format(source[mean_col] * scale)
            ci_val = source.get(ci_col, float("nan"))
            if pd.isna(ci_val):
                rendered.append(mean_txt)
            else:
                rendered.append(f"{mean_txt} $\\pm$ {fmt.format(ci_val * scale)}")
        return f"\\textbf{{{label}}} & " + " & ".join(rendered) + " \\\\\n"

    latex += f"\\multicolumn{{{n_cols + 1}}}{{l}}{{\\textit{{Judged quality}}}} \\\\\n"
    latex += row("Judge Average (1--5)", "judge_average")
    latex += row("Judge Accuracy", "judge_accuracy")
    latex += row("Judge Completeness", "judge_completeness")
    latex += row("Judge Groundedness", "judge_groundedness")

    latex += f"\\midrule\n\\multicolumn{{{n_cols + 1}}}{{l}}{{\\textit{{Reference matching}}}} \\\\\n"
    latex += row("ROUGE-L F1", "rouge_l_f", "{:.3f}")
    latex += row("BLEU-4", "bleu_4", "{:.3f}")

    latex += f"\\midrule\n\\multicolumn{{{n_cols + 1}}}{{l}}{{\\textit{{Behaviour and cost}}}} \\\\\n"
    # Both abstention measures, because they disagree and the disagreement is
    # the RQ2 result. System abstention is 0 in every cell -- the aggregator
    # always returns something -- so on its own this row reads as "nothing
    # happened" while the expert rate moves by an order of magnitude between
    # arms.
    latex += row("System Abstention Rate (\\%)", "abstained_flag", "{:.1f}", scale=100.0)
    latex += row("Expert Abstention Rate (\\%)", "expert_abstention_rate", "{:.1f}", scale=100.0)
    latex += row("Article Citations", "citations", "{:.1f}")
    latex += row("End-to-End Latency (s)", "elapsed_s")
    latex += row("Completion Tokens", "completion_tokens", "{:.0f}")

    latex += "\\bottomrule\n\\end{tabular}\n\\end{table*}"
    return latex


if __name__ == "__main__":
    main()
