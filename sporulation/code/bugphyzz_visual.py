#!/usr/bin/env python3
"""
Visualization script to replicate bugphyzz figure from microbe.cards Table S1.
Generates a four-panel (2x2) figure:
  A) Most present phyla (WA vs LA subsets)
  B) Distribution of phenotype labels (WA subset only)
  C) % missing annotations comparison (WA vs LA)
  D) Spore formation distribution by phylum (%)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

# SVG export settings - no clipping, editable text
mpl.rcParams['svg.fonttype'] = 'none'  # Keep text as text, not paths


def load_data(excel_path: str) -> pd.DataFrame:
    """Load the microbe.cards Table S1 data."""
    df = pd.read_excel(excel_path, sheet_name='Table S1 V2')
    return df


def plot_most_present_phyla(ax, df: pd.DataFrame, top_n: int = 8):
    """
    Panel A: Bar chart of most present phyla, split by WA/LA.
    """
    # Count species per phylum for WA and LA subsets
    wa_counts = df[df['Member of WA subset'] == True].groupby('Phylum').size()
    la_counts = df[df['Member of WA subset'] == False].groupby('Phylum').size()

    # Get total counts and select top N phyla
    total_counts = df.groupby('Phylum').size().sort_values(ascending=False)
    top_phyla = total_counts.head(top_n).index.tolist()

    # Prepare data for plotting
    wa_values = [wa_counts.get(p, 0) for p in top_phyla]
    la_values = [la_counts.get(p, 0) for p in top_phyla]

    x = np.arange(len(top_phyla))
    width = 0.35

    # Plot bars
    bars_wa = ax.bar(x - width/2, wa_values, width, label='Well-annotated set (WA)',
                     color='#4a4a4a', edgecolor='black', linewidth=0.5)
    bars_la = ax.bar(x + width/2, la_values, width, label='Low-annotated set (LA)',
                     color='#b0b0b0', edgecolor='black', linewidth=0.5)

    # Formatting
    ax.set_ylabel('Binomial names', fontsize=10)
    ax.set_title('Most present phyla', fontsize=11, fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(top_phyla, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=7, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add panel label
    ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top')


def plot_phenotype_distribution(ax, df: pd.DataFrame):
    """
    Panel B: Stacked bar chart of phenotype label distributions (WA subset only).
    """
    # Filter to WA subset only
    df = df[df['Member of WA subset'] == True].copy()

    # Define phenotypes and their possible values with colors and labels
    phenotypes_config = {
        'Gram staining': {
            'values': ['gram stain positive', 'gram stain negative'],
            'colors': ['#4a4a4a', '#b0b0b0'],
            'labels': ['+', '-'],
            'short_label': 'Gram staining'
        },
        'Biosafety level': {
            'values': ['biosafety level 1', 'biosafety level 2', 'biosafety level 3'],
            'colors': ['#4a4a4a', '#e07020', '#c03030'],
            'labels': ['1.', '2.', '3.'],
            'short_label': 'Biosafety level'
        },
        'Extreme environment tolerance': {
            'values': [True, False],
            'colors': ['#4a4a4a', '#b0b0b0'],
            'labels': ['T', 'F'],
            'short_label': 'Extreme env. tol.'
        },
        'Spore formation': {
            'values': [True, False],
            'colors': ['#4a4a4a', '#b0b0b0'],
            'labels': ['T', 'F'],
            'short_label': 'Spore formation'
        },
        'Host association': {
            'values': [True, False],
            'colors': ['#4a4a4a', '#b0b0b0'],
            'labels': ['T', 'F'],
            'short_label': 'Host association'
        },
        'Animal pathogenicity': {
            'values': [True, False],
            'colors': ['#4a4a4a', '#b0b0b0'],
            'labels': ['T', 'F'],
            'short_label': 'Animal pathogenicity'
        },
        'Biofilm formation': {
            'values': [True, False],
            'colors': ['#4a4a4a', '#b0b0b0'],
            'labels': ['T', 'F'],
            'short_label': 'Biofilm formation'
        }
    }

    # Calculate total counts for each phenotype and sort by descending count
    phenotype_totals = {}
    for phenotype, config in phenotypes_config.items():
        total = sum((df[phenotype] == val).sum() for val in config['values'])
        phenotype_totals[phenotype] = total

    sorted_phenotypes = sorted(phenotype_totals.keys(),
                                key=lambda x: phenotype_totals[x],
                                reverse=True)

    x = np.arange(len(sorted_phenotypes))
    width = 0.6

    # For each phenotype, create stacked bars
    for i, phenotype in enumerate(sorted_phenotypes):
        config = phenotypes_config[phenotype]
        bottom = 0
        for val, color, label in zip(config['values'], config['colors'], config['labels']):
            count = (df[phenotype] == val).sum()
            if count > 0:
                bar = ax.bar(i, count, width, bottom=bottom, color=color,
                             edgecolor='black', linewidth=0.5)
                # Add category label on segment if large enough
                if count > 150:
                    ax.text(i, bottom + count/2, label,
                            ha='center', va='center', fontsize=9,
                            color='white' if color == '#4a4a4a' else 'black',
                            fontweight='bold')
                bottom += count

    # Formatting
    ax.set_ylabel('Number of species', fontsize=10)
    ax.set_title('Distribution of phenotype labels', fontsize=11, fontweight='bold', loc='left')
    ax.set_xticks(x)
    short_labels = [phenotypes_config[p]['short_label'] for p in sorted_phenotypes]
    ax.set_xticklabels(short_labels, fontsize=8, rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add panel label
    ax.text(-0.1, 1.05, 'B', transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top')


def plot_missing_comparison(ax, df: pd.DataFrame):
    """
    Panel C: Grouped bar chart comparing % missing between WA and LA subsets.
    """
    phenotypes = [
        'Gram staining',
        'Biosafety level',
        'Extreme environment tolerance',
        'Spore formation',
        'Host association',
        'Animal pathogenicity',
        'Biofilm formation'
    ]

    short_labels = [
        'Gram\nstaining',
        'Biosafety\nlevel',
        'Extreme\nenv. tol.',
        'Spore\nformation',
        'Host\nassociation',
        'Animal\npath.',
        'Biofilm\nformation'
    ]

    # Split data
    df_wa = df[df['Member of WA subset'] == True]
    df_la = df[df['Member of WA subset'] == False]

    # Calculate % missing for each phenotype
    wa_missing = []
    la_missing = []

    for phenotype in phenotypes:
        wa_pct = df_wa[phenotype].isna().sum() / len(df_wa) * 100
        la_pct = df_la[phenotype].isna().sum() / len(df_la) * 100
        wa_missing.append(wa_pct)
        la_missing.append(la_pct)

    x = np.arange(len(phenotypes))
    width = 0.35

    # Plot bars
    bars_wa = ax.bar(x - width/2, wa_missing, width, label='WA',
                     color='#4a4a4a', edgecolor='black', linewidth=0.5)
    bars_la = ax.bar(x + width/2, la_missing, width, label='LA',
                     color='#b0b0b0', edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar in bars_wa:
        height = bar.get_height()
        if height > 5:
            ax.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{height:.0f}%', ha='center', va='center',
                    fontsize=6, color='white', fontweight='bold')

    for bar in bars_la:
        height = bar.get_height()
        if height > 5:
            ax.text(bar.get_x() + bar.get_width()/2, height/2,
                    f'{height:.0f}%', ha='center', va='center',
                    fontsize=6, color='black', fontweight='bold')

    # Formatting
    ax.set_ylabel('% missing', fontsize=10)
    ax.set_title('Missing annotations (WA vs LA)', fontsize=11, fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=7, ha='center')
    ax.legend(loc='upper left', fontsize=7, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 105)

    # Add panel label
    ax.text(-0.1, 1.05, 'C', transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top')


def plot_spore_by_phyla(ax, df: pd.DataFrame, top_n: int = 10):
    """
    Panel D: Percentage-based stacked bar chart of spore formation by phylum.
    """
    # Filter to species with spore formation annotation
    df_annotated = df[df['Spore formation'].notna()].copy()

    # Get top phyla by total annotated count
    phyla_counts = df_annotated.groupby('Phylum').size().sort_values(ascending=False)
    top_phyla = phyla_counts.head(top_n).index.tolist()

    # Calculate percentages for each phylum
    true_pcts = []
    false_pcts = []

    for phylum in top_phyla:
        phylum_df = df_annotated[df_annotated['Phylum'] == phylum]
        total = len(phylum_df)
        true_count = (phylum_df['Spore formation'] == True).sum()
        false_count = (phylum_df['Spore formation'] == False).sum()
        true_pcts.append(true_count / total * 100)
        false_pcts.append(false_count / total * 100)

    x = np.arange(len(top_phyla))
    width = 0.6

    # Plot stacked bars (percentage-based)
    bars_true = ax.bar(x, true_pcts, width, label='Spore-forming (T)',
                       color='#4a4a4a', edgecolor='black', linewidth=0.5)
    bars_false = ax.bar(x, false_pcts, width, bottom=true_pcts, label='Non-spore-forming (F)',
                        color='#b0b0b0', edgecolor='black', linewidth=0.5)

    # Add labels on segments
    for i, (t_pct, f_pct) in enumerate(zip(true_pcts, false_pcts)):
        if t_pct > 8:
            ax.text(i, t_pct / 2, 'T',
                    ha='center', va='center', fontsize=8,
                    color='white', fontweight='bold')
        if f_pct > 8:
            ax.text(i, t_pct + f_pct / 2, 'F',
                    ha='center', va='center', fontsize=8,
                    color='black', fontweight='bold')

    # Formatting
    ax.set_ylabel('% of species', fontsize=10)
    ax.set_title('Spore formation by phylum', fontsize=11, fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(top_phyla, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right', fontsize=7, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 105)

    # Add panel label
    ax.text(-0.12, 1.05, 'D', transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top')


def main():
    # Paths
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / 'microbe.cards table S1.xlsx'
    reports_dir = script_dir.parent / 'reports'
    reports_dir.mkdir(exist_ok=True)
    output_pdf = reports_dir / 'bugphyzz_figure.pdf'
    output_svg = reports_dir / 'bugphyzz_figure.svg'

    # Load data
    print(f"Loading data from {data_path}")
    df = load_data(data_path)
    print(f"Loaded {len(df)} species")

    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Plot panels
    plot_most_present_phyla(axes[0, 0], df)
    plot_phenotype_distribution(axes[0, 1], df)
    plot_missing_comparison(axes[1, 0], df)
    plot_spore_by_phyla(axes[1, 1], df)

    # Adjust layout
    plt.tight_layout()

    # Disable clipping for SVG editability
    for ax in axes.flat:
        for artist in ax.get_children():
            artist.set_clip_on(False)

    # Save figures
    fig.savefig(output_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(output_svg, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_pdf} and {output_svg}")

    plt.show()


if __name__ == '__main__':
    main()
