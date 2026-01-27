#!/usr/bin/env python3
"""Generate latency breakdown visualizations and LaTeX tables from profiling results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_profile_results(json_file):
    """Load profiling results from JSON."""
    with open(json_file, "r") as f:
        return json.load(f)


def create_breakdown_chart(results, output_path):
    """Create bar chart for component latency breakdown."""
    stats = results["component_statistics"]

    # Components to show (excluding total)
    components = ["embedding", "faiss_search", "bm25_search", "fusion", "rerank"]
    p50_values = [stats[c]["p50"] for c in components if c in stats]
    p95_values = [stats[c]["p95"] for c in components if c in stats]
    total_p50 = stats["total"]["p50"]

    # Format component names for display
    display_names = ["Embedding", "FAISS", "BM25", "Fusion", "Reranking"]

    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

    x = np.arange(len(components))
    width = 0.35

    bars1 = ax.bar(x - width / 2, p50_values, width, label="p50", alpha=0.8, color="#2E86AB")
    bars2 = ax.bar(x + width / 2, p95_values, width, label="p95", alpha=0.8, color="#A23B72")

    # Add value labels and percentages on bars
    for bars, values in [(bars1, p50_values), (bars2, p95_values)]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            pct = (val / total_p50 * 100) if total_p50 > 0 else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.0f}ms\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Component", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Component Latency Breakdown (100 Queries)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Chart saved: {output_path}")
    plt.close()


def create_latex_table(results, output_path):
    """Create LaTeX table from profiling results."""
    stats = results["component_statistics"]
    total_p50 = stats["total"]["p50"]

    # Component mapping for nice names
    comp_names = {
        "embedding": "Query Embedding",
        "faiss_search": "FAISS Search",
        "bm25_search": "BM25 Search",
        "fusion": "Fusion \\& Ranking",
        "rerank": "Cross-Encoder Reranking",
    }

    # Build table rows
    rows = []
    for component in ["embedding", "faiss_search", "bm25_search", "fusion", "rerank"]:
        if component in stats:
            s = stats[component]
            p50 = s["p50"]
            p95 = s["p95"]
            p99 = s["p99"]
            percentage = (p50 / total_p50) * 100 if total_p50 > 0 else 0

            comp_name = comp_names[component]
            rows.append(
                f"{comp_name:30s} & {p50:7.1f} & {p95:7.1f} & {p99:7.1f} & {percentage:5.1f} \\\\"
            )

    # Add total row
    total = stats["total"]
    total_name = r"\textbf{Total}"
    rows.append(
        f"{total_name:30s} & {total['p50']:7.1f} & {total['p95']:7.1f} & {total['p99']:7.1f} & {100.0:5.1f} \\\\"
    )

    # Create LaTeX table
    latex = r"""\begin{table}[htb]
\centering
\small
\caption{Component Latency Breakdown (milliseconds) based on 100 queries over SciFact dataset.}
\label{tab:latency_breakdown}
\begin{tabular}{|l|r|r|r|r|}
\hline
\textbf{Component} & \textbf{p50} & \textbf{p95} & \textbf{p99} & \textbf{\%} \\
\hline
"""
    latex += "\n".join(rows)
    latex += r"""\hline
\end{tabular}
\end{table}
"""

    with open(output_path, "w") as f:
        f.write(latex)
    print(f"✓ LaTeX table saved: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate latency breakdown visualizations")
    parser.add_argument(
        "json_file",
        nargs="?",
        default="results/latency_breakdown_full_100.json",
        help="Path to profiling results JSON",
    )
    parser.add_argument("--chart-output", default="paper/figs/latency_breakdown.png")
    parser.add_argument("--table-output", default="results/latency_breakdown_table.tex")

    args = parser.parse_args()

    # Load results
    if not Path(args.json_file).exists():
        print(f"Error: {args.json_file} not found")
        return

    results = load_profile_results(args.json_file)

    # Create output directories
    Path(args.chart_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.table_output).parent.mkdir(parents=True, exist_ok=True)

    # Generate artifacts
    create_breakdown_chart(results, args.chart_output)
    create_latex_table(results, args.table_output)

    print("\n✓ Latency breakdown artifacts generated successfully")


if __name__ == "__main__":
    main()
