#!/usr/bin/env python3
"""Generate publication-ready plots from memory profiling data.

This script reads memory_profile.jsonl files and creates plots for the ACL paper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
except ImportError:
    print("Error: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


def load_profile(jsonl_path: Path) -> dict:
    """Load memory profile from JSONL file."""
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    return {
        'elapsed': [s['elapsed_s'] for s in samples],
        'rss_mb': [s['rss_mb'] for s in samples],
        'tags': [s['tag'] for s in samples],
    }


def plot_single_run(profile: dict, corpus_label: str, output_path: Path):
    """Create a single plot for one corpus run."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot RSS over time
    ax.plot(profile['elapsed'], profile['rss_mb'], 
            linewidth=1.5, color='#2E86AB', alpha=0.8)
    
    # Add horizontal lines for min/max
    min_rss = min(profile['rss_mb'])
    max_rss = max(profile['rss_mb'])
    ax.axhline(min_rss, color='green', linestyle='--', alpha=0.5, label=f'Min: {min_rss:.1f} MB')
    ax.axhline(max_rss, color='red', linestyle='--', alpha=0.5, label=f'Max: {max_rss:.1f} MB')
    
    # Styling
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('RSS Memory (MB)', fontsize=12)
    ax.set_title(f'Memory Usage During Ingestion: {corpus_label}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Add delta annotation
    delta = max_rss - min_rss
    ax.text(0.02, 0.98, f'Δ = {delta:.1f} MB', 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_multi_corpus(profiles: dict, output_path: Path):
    """Create comparison plot across multiple corpus sizes."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, (label, profile) in enumerate(profiles.items()):
        color = colors[i % len(colors)]
        ax.plot(profile['elapsed'], profile['rss_mb'], 
                linewidth=1.5, color=color, alpha=0.7, label=label)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('RSS Memory (MB)', fontsize=12)
    ax.set_title('O(1) Memory Scaling: Constant RSS Across Corpus Sizes', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {output_path}")


def main():
    results_dir = Path("results/memory_profiles")
    
    if not results_dir.exists():
        print(f"Error: {results_dir} not found")
        sys.exit(1)
    
    # Find all memory profile files
    profile_files = list(results_dir.glob("*/memory_profile.jsonl"))
    
    if not profile_files:
        print("No memory profile files found")
        sys.exit(1)
    
    print(f"Found {len(profile_files)} profile(s)")
    
    # Load all profiles
    profiles = {}
    for profile_path in profile_files:
        corpus_name = profile_path.parent.name
        profile = load_profile(profile_path)
        profiles[corpus_name] = profile
        
        # Generate individual plot
        plot_path = profile_path.parent / f"{corpus_name}_memory_plot.png"
        plot_single_run(profile, corpus_name, plot_path)
    
    # Generate comparison plot if multiple runs
    if len(profiles) > 1:
        comparison_path = results_dir / "memory_scaling_comparison.png"
        plot_multi_corpus(profiles, comparison_path)
    
    print("\n✅ All plots generated successfully")


if __name__ == "__main__":
    main()
