#!/usr/bin/env python3
"""
Publication-Quality Pareto Front Visualization for Thin-Film Optimization.

Reads Pareto front data from CSV and generates high-resolution figures
suitable for journal submission (Nature, ACS, Wiley).

Author: Y. Aykut
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import AutoMinorLocator

# Add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config.settings as config


def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })


def find_knee_point(a_vals, b_vals):
    """
    Find the knee point of the Pareto front using the maximum distance
    from the line connecting the two extreme points.
    """
    if len(a_vals) < 3:
        return 0
    
    # Normalize to [0,1] for fair distance comparison
    a_norm = (a_vals - a_vals.min()) / (a_vals.max() - a_vals.min() + 1e-12)
    b_norm = (b_vals - b_vals.min()) / (b_vals.max() - b_vals.min() + 1e-12)
    
    # Line from first point to last point (in sorted order by A)
    p1 = np.array([a_norm[0], b_norm[0]])
    p2 = np.array([a_norm[-1], b_norm[-1]])
    
    # Distance from each point to this line
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-12:
        return 0
    
    distances = np.abs(
        (p2[0] - p1[0]) * (p1[1] - b_norm) - (p1[0] - a_norm) * (p2[1] - p1[1])
    ) / line_len
    
    return np.argmax(distances)


def plot_pareto_front(
    pareto_csv=None,
    all_csv=None,
    output_dir=None,
    show_all=True,
    show_knee=True,
    a_constraint=None,
    figsize=(8, 6)
):
    """
    Generate publication-quality Pareto front plot.
    
    Args:
        pareto_csv: Path to Pareto front CSV
        all_csv: Path to all solutions CSV (optional, for background scatter)
        output_dir: Output directory for figures
        show_all: Whether to show all tested designs as background
        show_knee: Whether to highlight the knee point
        a_constraint: If set, draws a vertical line at this A value
        figsize: Figure size tuple
    """
    setup_publication_style()
    
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, 'pareto')
    os.makedirs(output_dir, exist_ok=True)
    
    if pareto_csv is None:
        pareto_csv = os.path.join(output_dir, 'pareto_front.csv')
    if all_csv is None:
        all_csv = os.path.join(output_dir, 'pareto_all_solutions.csv')
    
    # Load data
    df_pareto = pd.read_csv(pareto_csv)
    df_pareto = df_pareto.sort_values('A').reset_index(drop=True)
    
    has_all = os.path.exists(all_csv) if show_all else False
    if has_all:
        df_all = pd.read_csv(all_csv)
    
    # ────────────────────────────────────────
    #  FIGURE 1: Main Pareto Front (A vs B, CRI colored)
    # ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    
    # Background: all solutions
    if has_all:
        ax.scatter(
            df_all['A'], df_all['B'],
            c='#E0E0E0', s=20, alpha=0.4, zorder=1,
            label='All evaluated designs', edgecolors='none'
        )
    
    # Pareto front with CRI color coding
    a = df_pareto['A'].values
    b = df_pareto['B'].values
    cri = df_pareto['CRI'].values
    
    # Color map: CRI values
    cri_min_val = max(0, cri.min() - 5)
    cri_max_val = min(100, cri.max() + 5)
    norm = mcolors.Normalize(vmin=cri_min_val, vmax=cri_max_val)
    cmap = plt.cm.RdYlGn  # Red (low CRI) → Green (high CRI)
    
    # Plot Pareto line
    sorted_idx = np.argsort(a)
    ax.plot(
        a[sorted_idx], b[sorted_idx],
        color='#333333', linewidth=1.5, linestyle='--', alpha=0.6, zorder=2
    )
    
    # Plot Pareto points with CRI coloring
    scatter = ax.scatter(
        a, b, c=cri, cmap=cmap, norm=norm,
        s=80, edgecolors='black', linewidths=0.8, zorder=3,
        label='Pareto-optimal designs'
    )
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Color Rendering Index (CRI)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Knee point
    if show_knee and len(a) >= 3:
        knee_idx = find_knee_point(a[sorted_idx], b[sorted_idx])
        knee_a = a[sorted_idx][knee_idx]
        knee_b = b[sorted_idx][knee_idx]
        knee_cri = cri[sorted_idx][knee_idx]
        
        ax.scatter(
            knee_a, knee_b,
            s=250, facecolors='none', edgecolors='#E63946', linewidths=2.5,
            zorder=4, label=f'Knee point (A={knee_a:.1f}%, B={knee_b:.1f})'
        )
        
        # Annotation
        ax.annotate(
            f'Knee Point\nA={knee_a:.1f}%\nB={knee_b:.1f}\nCRI={knee_cri:.0f}',
            xy=(knee_a, knee_b),
            xytext=(20, 25), textcoords='offset points',
            fontsize=9, ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#E63946', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.5),
            zorder=5
        )
    
    # A constraint line
    if a_constraint is not None:
        ax.axvline(
            x=a_constraint, color='#457B9D', linestyle=':', linewidth=1.5, alpha=0.7,
            label=f'A ≥ {a_constraint}% constraint'
        )
        ax.axvspan(a_constraint, ax.get_xlim()[1], alpha=0.04, color='#457B9D')
    
    # Trade-off arrow annotation
    mid_idx = len(sorted_idx) // 2
    if mid_idx > 0 and mid_idx < len(sorted_idx) - 1:
        ax.annotate(
            '', xy=(a[sorted_idx][-1], b[sorted_idx][-1]),
            xytext=(a[sorted_idx][0], b[sorted_idx][0]),
            arrowprops=dict(
                arrowstyle='<->', color='#6C757D', lw=1.2,
                connectionstyle='arc3,rad=0.15'
            )
        )
        # Label the trade-off
        ax.text(
            0.03, 0.03, 'Trade-off: ← Higher B    |    Higher A →',
            transform=ax.transAxes, fontsize=9, color='#6C757D',
            style='italic', alpha=0.7,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )
    
    ax.set_xlabel('Average Visible Transmittance, A (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Photocurrent Density, J$_{ph}$ (mA/cm²)', fontsize=14, fontweight='bold')
    ax.set_title('Pareto Front: A vs J$_{ph}$ Trade-off\n(Gradient-Based Multi-Objective Optimization)', fontsize=14)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='#CCCCCC')
    
    fig.tight_layout()
    
    # Save
    png_path = os.path.join(output_dir, 'pareto_front_a_b.png')
    pdf_path = os.path.join(output_dir, 'pareto_front_a_b.pdf')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Figure saved → {png_path}")
    print(f"  Figure saved → {pdf_path}")
    
    # ────────────────────────────────────────
    #  FIGURE 2: Design Space Parallel Coordinates
    # ────────────────────────────────────────
    num_layers = len(config.LAYERS)
    d_cols = [f'd{i+1}' for i in range(num_layers)]
    
    if all(col in df_pareto.columns for col in d_cols):
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        
        # Normalize each dimension to [0, 1] for visualization
        d_data = df_pareto[d_cols].values
        d_min = d_data.min(axis=0)
        d_max = d_data.max(axis=0)
        d_range = d_max - d_min
        d_range[d_range == 0] = 1  # avoid div by zero
        d_norm = (d_data - d_min) / d_range
        
        # Color by B
        b_norm = mcolors.Normalize(vmin=b.min(), vmax=b.max())
        b_cmap = plt.cm.plasma
        
        x_coords = np.arange(num_layers)
        
        for i in range(len(df_pareto)):
            color = b_cmap(b_norm(b[i]))
            ax2.plot(x_coords, d_norm[i], color=color, alpha=0.5, linewidth=1.0)
        
        # Add ticks with actual ranges
        ax2.set_xticks(x_coords)
        layer_labels = [f'd{i+1}\n({config.LAYERS[i]})\n[{d_min[i]:.0f}–{d_max[i]:.0f} nm]'
                        for i in range(num_layers)]
        ax2.set_xticklabels(layer_labels, fontsize=9)
        ax2.set_ylabel('Normalized Thickness', fontsize=12)
        ax2.set_title('Pareto-Optimal Design Space (Parallel Coordinates)', fontsize=14)
        
        # Colorbar for B
        sm = plt.cm.ScalarMappable(cmap=b_cmap, norm=b_norm)
        sm.set_array([])
        cbar2 = plt.colorbar(sm, ax=ax2, shrink=0.85, pad=0.02)
        cbar2.set_label('J$_{ph}$ (mA/cm²)', fontsize=11)
        
        fig2.tight_layout()
        
        pc_png = os.path.join(output_dir, 'pareto_parallel_coordinates.png')
        pc_pdf = os.path.join(output_dir, 'pareto_parallel_coordinates.pdf')
        fig2.savefig(pc_png, dpi=300, bbox_inches='tight')
        fig2.savefig(pc_pdf, bbox_inches='tight')
        plt.close(fig2)
        
        print(f"  Figure saved → {pc_png}")
        print(f"  Figure saved → {pc_pdf}")
    
    # ────────────────────────────────────────
    #  FIGURE 3: CRI vs A with B bubbles
    # ────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    
    b_sizes = 30 + 200 * (b - b.min()) / (b.max() - b.min() + 1e-12)
    
    scatter3 = ax3.scatter(
        a, cri, s=b_sizes, c=b, cmap='viridis',
        edgecolors='black', linewidths=0.5, alpha=0.8
    )
    
    cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.85, pad=0.02)
    cbar3.set_label('J$_{ph}$ (mA/cm²)', fontsize=12)
    
    ax3.set_xlabel('Average Visible Transmittance, A (%)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Color Rendering Index (CRI)', fontsize=14, fontweight='bold')
    ax3.set_title('Pareto Designs: A vs CRI\n(Bubble size ~ J$_{ph}$)', fontsize=14)
    
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    
    fig3.tight_layout()
    
    bubble_png = os.path.join(output_dir, 'pareto_a_cri_bubble.png')
    bubble_pdf = os.path.join(output_dir, 'pareto_a_cri_bubble.pdf')
    fig3.savefig(bubble_png, dpi=300, bbox_inches='tight')
    fig3.savefig(bubble_pdf, bbox_inches='tight')
    plt.close(fig3)
    
    print(f"  Figure saved → {bubble_png}")
    print(f"  Figure saved → {bubble_pdf}")
    
    print("\n✅ All figures generated successfully.")


def main():
    parser = argparse.ArgumentParser(description='Plot Pareto Front for TMM Optimization')
    parser.add_argument('--pareto_csv', type=str, default=None,
                        help='Path to Pareto front CSV')
    parser.add_argument('--all_csv', type=str, default=None,
                        help='Path to all solutions CSV')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for figures')
    parser.add_argument('--a_constraint', type=float, default=25.0,
                        help='Draw A constraint line at this value (default: 25)')
    parser.add_argument('--no_all', action='store_true',
                        help='Do not show background scatter of all solutions')
    parser.add_argument('--no_knee', action='store_true',
                        help='Do not highlight knee point')
    
    args = parser.parse_args()
    
    plot_pareto_front(
        pareto_csv=args.pareto_csv,
        all_csv=args.all_csv,
        output_dir=args.output_dir,
        show_all=not args.no_all,
        show_knee=not args.no_knee,
        a_constraint=args.a_constraint,
    )


if __name__ == '__main__':
    main()
