#!/usr/bin/env python3
"""Generate component latency breakdown visualization and LaTeX table for paper."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_profile_results(json_path: Path) -> dict:
    """Load profiling results from JSON file."""
    with open(json_path) as f:
        return json.load(f)

def create_breakdown_chart(stats: dict, output_path: Path):
    """Create stacked bar chart for latency breakdown."""
    components = list(stats['component_statistics'].keys())
    # Skip total
    components = [c for c in components if c != 'total']
    
    p50_values = [stats['component_statistics'][c]['p50'] for c in components]
    p95_values = [stats['component_statistics'][c]['p95'] for c in components]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors for components
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'][:len(components)]
    
    # Plot p50
    bars1 = ax1.bar(range(len(components)), p50_values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Query Latency Breakdown (p50)', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(components)))
    ax1.set_xticklabels([c.replace('_', ' ').title() for c in components], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, p50_values)):
        pct = (val / sum(p50_values)) * 100
        ax1.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.0f}ms\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot p95
    bars2 = ax2.bar(range(len(components)), p95_values, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Query Latency Breakdown (p95)', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(components)))
    ax2.set_xticklabels([c.replace('_', ' ').title() for c in components], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, p95_values)):
        pct = (val / sum(p95_values)) * 100
        ax2.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.0f}ms\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved figure to {output_path}")
    plt.close()

def create_latex_table(stats: dict) -> str:
    """Generate LaTeX table from component stats."""
    components = list(stats['component_statistics'].keys())
    components = [c for c in components if c != 'total']
    
    # Get total p50 for percentage calculation
    total_p50 = stats['component_statistics']['total']['p50']
    
    latex_lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\small',
        r'\resizebox{\columnwidth}{!}{%',
        r'\begin{tabular}{lrrrr}',
        r'\toprule',
        r'\textbf{Component} & \textbf{p50 (ms)} & \textbf{p95 (ms)} & \textbf{p99 (ms)} & \textbf{\% of Total} \\',
        r'\midrule',
    ]
    
    for component in components:
        data = stats['component_statistics'][component]
        p50 = data['p50']
        p95 = data['p95']
        p99 = data['p99']
        pct = (p50 / total_p50) * 100 if total_p50 > 0 else 0
        
        comp_name = component.replace('_', ' ').title()
        if component == 'faiss_search':
            comp_name = 'FAISS Search'
        elif component == 'bm25_search':
            comp_name = 'BM25 Search'
        
        latex_lines.append(
            f'{comp_name} & {p50:.1f} & {p95:.1f} & {p99:.1f} & {pct:.1f}\\% \\\\'
        )
    
    # Add total row
    total = stats['component_statistics']['total']
    latex_lines.extend([
        r'\midrule',
        rf'\textbf{{Total}} & \textbf{{{total["p50"]:.1f}}} & \textbf{{{total["p95"]:.1f}}} & \textbf{{{total["p99"]:.1f}}} & \textbf{{100.0\%}} \\',
        r'\bottomrule',
        r'\end{tabular}%',
        r'}',
        r'\caption{Query latency breakdown by component across all queries. Time measured in milliseconds.}',
        r'\label{tab:latency-breakdown}',
        r'\end{table}',
    ])
    
    return '\n'.join(latex_lines)

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/generate_latency_breakdown.py <profile_results.json>")
        sys.exit(1)
    
    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        sys.exit(1)
    
    # Load results
    stats = load_profile_results(json_path)
    
    # Create chart
    output_chart = Path('paper/figs/latency_breakdown.png')
    output_chart.parent.mkdir(parents=True, exist_ok=True)
    create_breakdown_chart(stats, output_chart)
    
    # Generate LaTeX table
    latex_table = create_latex_table(stats)
    output_latex = Path('results/latency_breakdown_table.tex')
    output_latex.parent.mkdir(parents=True, exist_ok=True)
    with open(output_latex, 'w') as f:
        f.write(latex_table)
    print(f"✅ Saved LaTeX table to {output_latex}")
    
    # Print summary
    print("\n" + "="*60)
    print("COMPONENT LATENCY BREAKDOWN SUMMARY")
    print("="*60)
    comp_stats = stats['component_statistics']
    for comp in [c for c in comp_stats.keys() if c != 'total']:
        data = comp_stats[comp]
        print(f"\n{comp.upper()}:")
        print(f"  p50: {data['p50']:.1f} ms")
        print(f"  p95: {data['p95']:.1f} ms")
        print(f"  p99: {data['p99']:.1f} ms")
        print(f"  mean: {data['mean']:.1f} ms")
    
    total = comp_stats['total']
    print(f"\nTOTAL LATENCY:")
    print(f"  p50: {total['p50']:.1f} ms")
    print(f"  p95: {total['p95']:.1f} ms")
    print(f"  p99: {total['p99']:.1f} ms")
    print(f"  mean: {total['mean']:.1f} ms")
    
    print("\n" + "="*60)
    print("LaTeX TABLE (ready to insert into paper.tex):")
    print("="*60)
    print(latex_table)

if __name__ == "__main__":
    main()
