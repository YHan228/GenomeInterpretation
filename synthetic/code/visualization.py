"""
Visualization and analysis functions for synthetic sequence experiments.
Includes skill metric calculations and all plotting functions from toy_slurm.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle


# --------------------------------------------------------------------------- #
# Skill Metric Calculations
# --------------------------------------------------------------------------- #

def calculate_skills(auc_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate skill metrics from AUC values.
    
    Parameters:
    - auc_values: Array of AUC values
    
    Returns:
    - skill: Linear skill (2 * AUC - 1)
    - root_skill: Square root skill with sign preservation
    - effective_root_skill: Positive-only root skill
    """
    # Linear skill maps AUC [0,1] to skill [-1,1]
    skill = 2 * auc_values - 1
    
    # Root skill applies square root transformation while preserving sign
    root_skill = np.sign(skill) * np.sqrt(np.abs(skill))
    
    # Effective root skill only considers positive domain (AUC > 0.5)
    effective_root_skill = np.maximum(0, root_skill)
    
    return skill, root_skill, effective_root_skill


def add_skill_metrics(df: pd.DataFrame, auc_column: str = 'SaliencyAUC') -> pd.DataFrame:
    """
    Add skill metrics to a dataframe containing AUC values.
    
    Parameters:
    - df: DataFrame with AUC values
    - auc_column: Name of the AUC column
    
    Returns:
    - DataFrame with added skill columns
    """
    df = df.copy()
    
    if auc_column in df.columns:
        skill, root_skill, effective_root_skill = calculate_skills(df[auc_column].values)
        df['Skill'] = skill
        df['RootSkill'] = root_skill
        df['EffectiveRootSkill'] = effective_root_skill
    
    return df


def calculate_delta_metrics(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate improvement metrics relative to standard model.
    
    Parameters:
    - df_results: DataFrame with experiment results
    
    Returns:
    - DataFrame with delta metrics added
    """
    df = df_results.copy()
    
    # Find baseline (standard model) results
    baseline_df = df[df['model_type'] == 'Standard'].copy()
    
    # For each non-standard model, calculate deltas
    results = []
    for idx, row in df.iterrows():
        if row['model_type'] == 'Standard':
            continue
            
        # Find matching baseline
        baseline = baseline_df[
            (baseline_df['gc_pos'] == row['gc_pos']) &
            (baseline_df['conservation'] == row['conservation']) &
            (baseline_df['seed'] == row['seed'])
        ]
        
        if len(baseline) == 0:
            continue
            
        baseline = baseline.iloc[0]
        
        # Calculate deltas for all metrics
        result_row = row.to_dict()
        
        # Basic metrics
        for metric in ['wIoU', 'Accuracy', 'SaliencyAUC', 'SaliencySNR']:
            if metric in row and metric in baseline:
                result_row[f'delta_{metric}'] = row[metric] - baseline[metric]
                result_row[f'{metric}_base'] = baseline[metric]
        
        # Skill metrics
        if 'SaliencyAUC' in row and 'SaliencyAUC' in baseline:
            # Calculate skills for robust and baseline
            skill_robust, root_skill_robust, eff_root_skill_robust = calculate_skills(np.array([row['SaliencyAUC']]))
            skill_base, root_skill_base, eff_root_skill_base = calculate_skills(np.array([baseline['SaliencyAUC']]))
            
            result_row['delta_Skill'] = (skill_robust[0] - skill_base[0]) / 2.0  # Normalized
            result_row['delta_RootSkill'] = root_skill_robust[0] - root_skill_base[0]
            result_row['delta_EffectiveRootSkill'] = eff_root_skill_robust[0] - eff_root_skill_base[0]
            
            # Store base values
            result_row['Skill_base'] = skill_base[0]
            result_row['RootSkill_base'] = root_skill_base[0]
            result_row['EffectiveRootSkill_base'] = eff_root_skill_base[0]
        
        results.append(result_row)
    
    return pd.DataFrame(results)


# --------------------------------------------------------------------------- #
# Data Loading and Processing
# --------------------------------------------------------------------------- #

def load_experiment_results(output_dir: Path) -> pd.DataFrame:
    """
    Load and consolidate experiment results into a DataFrame.
    
    Parameters:
    - output_dir: Directory containing experiment results
    
    Returns:
    - DataFrame with all results
    """
    results = []
    
    # Find all result files
    for result_file in output_dir.glob("**/multi_seed_results.npz"):
        # Load NPZ file
        data = np.load(result_file, allow_pickle=True)
        
        # Extract experiment info from path
        parts = result_file.parts
        if 'scheduled' in parts or 'no_schedule' in parts:
            schedule = 'scheduled' in parts
        else:
            schedule = None
            
        # Parse hyperparameters from filename or directory
        parent_dir = result_file.parent.name
        if parent_dir.startswith('gc_pos_'):
            # Vanilla experiment
            gc_pos = float(parent_dir.split('_')[2])
            conservation = float(parent_dir.split('_')[4])
            gc_gap = None
        elif parent_dir.startswith('gc_gap_'):
            # Complex experiment
            gc_gap = float(parent_dir.split('_')[2])
            conservation = float(parent_dir.split('_')[4])
            gc_pos = 0.5 + gc_gap
        else:
            continue
            
        # Extract results
        experiment_mode = str(data.get('experiment_mode', 'unknown'))
        seeds = data['seeds']
        
        # Standard model results
        for i, seed in enumerate(seeds):
            result = {
                'experiment_mode': experiment_mode,
                'model_type': 'Standard',
                'gc_pos': gc_pos,
                'gc_gap': gc_gap,
                'conservation': conservation,
                'seed': seed,
                'schedule': schedule,
                'param_val': 0.0,
            }
            
            # Add metrics
            if 'std_wious' in data:
                result['wIoU'] = data['std_wious'][i]
            result['Accuracy'] = data['std_accs'][i]
            result['SaliencyAUC'] = data['std_aucs'][i]
            result['SaliencySNR'] = data['std_snrs'][i]
            
            if 'std_motif_aucs' in data:
                result['MotifSaliencyAUC'] = data['std_motif_aucs'][i]
                result['MotifSaliencySNR'] = data['std_motif_snrs'][i]
                
            results.append(result)
        
        # Robust model results
        param_values = data.get('param_values', data.get('epsilons', []))
        for j, param_val in enumerate(param_values):
            for i, seed in enumerate(seeds):
                result = {
                    'experiment_mode': experiment_mode,
                    'model_type': 'HotFlip' if 'hotflip' in experiment_mode else 'Smoothing',
                    'gc_pos': gc_pos,
                    'gc_gap': gc_gap,
                    'conservation': conservation,
                    'seed': seed,
                    'schedule': schedule,
                    'param_val': param_val,
                }
                
                # Add metrics
                if 'rob_wious' in data:
                    result['wIoU'] = data['rob_wious'][i][j]
                result['Accuracy'] = data['rob_accs'][i][j]
                result['SaliencyAUC'] = data['rob_aucs'][i][j]
                result['SaliencySNR'] = data['rob_snrs'][i][j]
                
                if 'rob_motif_aucs' in data:
                    result['MotifSaliencyAUC'] = data['rob_motif_aucs'][i][j]
                    result['MotifSaliencySNR'] = data['rob_motif_snrs'][i][j]
                    
                results.append(result)
    
    # Also load from pickle files if available
    for result_file in output_dir.glob("**/*_results.pkl"):
        with open(result_file, 'rb') as f:
            data = pickle.load(f)
            
        # Parse filename
        parts = result_file.stem.split('_')
        exp_name = '_'.join(parts[:-2])
        seed = int(parts[-2].replace('seed', ''))
        
        # Parse directory for hyperparameters
        parent_dir = result_file.parent.name
        if parent_dir.startswith('gc'):
            parts = parent_dir.split('_')
            if len(parts) >= 2:
                gc_val = float(parts[0].replace('gc', ''))
                cons_val = float(parts[1].replace('cons', ''))
                
                result = {
                    'experiment': exp_name,
                    'seed': seed,
                    'conservation': cons_val,
                }
                
                # Determine if vanilla or complex based on gc value
                if gc_val < 1.0:  # Likely gc_pos
                    result['gc_pos'] = gc_val
                    result['experiment_type'] = 'vanilla'
                else:  # Likely gc_gap (usually small values)
                    result['gc_gap'] = gc_val
                    result['gc_pos'] = 0.5 + gc_val
                    result['experiment_type'] = 'complex'
                
                # Add all metrics from the data
                result.update(data)
                results.append(result)
    
    if not results:
        return pd.DataFrame()
        
    df = pd.DataFrame(results)
    
    # Add skill metrics
    df = add_skill_metrics(df)
    
    return df


# --------------------------------------------------------------------------- #
# Plotting Functions
# --------------------------------------------------------------------------- #

def plot_metric_improvement(df: pd.DataFrame, metric: str, output_dir: Path,
                           experiment_type: str = 'vanilla'):
    """
    Plot improvement in a metric vs standard model.
    
    Parameters:
    - df: DataFrame with delta metrics
    - metric: Metric name to plot
    - output_dir: Directory to save plots
    - experiment_type: 'vanilla' or 'complex'
    """
    plt.figure(figsize=(12, 8))
    
    # Filter to robust models only
    df_plot = df[df['model_type'] != 'Standard'].copy()
    
    if f'delta_{metric}' not in df_plot.columns:
        print(f"Warning: delta_{metric} not found in data")
        return
        
    # Create boxplot
    if experiment_type == 'vanilla':
        g = sns.boxplot(data=df_plot, x='param_val', y=f'delta_{metric}',
                       hue='model_type', palette='Set2')
    else:
        g = sns.boxplot(data=df_plot, x='param_val', y=f'delta_{metric}',
                       hue='model_type', palette='Set2')
    
    # Customize plot
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    if metric == 'SaliencyAUC':
        plt.ylabel("ΔEffectiveRootSkill")
        plt.title(f"Change in Saliency Effective Root Skill vs. Standard Model")
    else:
        plt.ylabel(f"Δ{metric}")
        plt.title(f"Improvement in {metric} vs. Standard Model")
    
    plt.xlabel("ε (Max Flip Fraction)" if 'hotflip' in df_plot['experiment_mode'].iloc[0] else "ε (Smoothing)")
    
    # Save plot
    plot_path = output_dir / f"delta_{metric}_boxplot.pdf"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved plot to {plot_path}")


def plot_heatmap_grid(df: pd.DataFrame, metric: str, output_dir: Path,
                     experiment_type: str = 'vanilla'):
    """
    Plot heatmap grid showing metric by hyperparameters.
    
    Parameters:
    - df: DataFrame with results
    - metric: Metric to plot
    - output_dir: Directory to save plots
    - experiment_type: 'vanilla' or 'complex'
    """
    # Find best epsilon for each hyperparameter combination
    df_best = df.groupby(['gc_pos', 'conservation', 'model_type']).apply(
        lambda x: x.loc[x[f'delta_{metric}'].idxmax()] if f'delta_{metric}' in x.columns else x.iloc[0]
    ).reset_index(drop=True)
    
    # Create figure with subplots for each model type
    model_types = df_best['model_type'].unique()
    n_models = len(model_types)
    
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for idx, model_type in enumerate(model_types):
        ax = axes[idx]
        df_model = df_best[df_best['model_type'] == model_type]
        
        # Create pivot table
        if experiment_type == 'vanilla':
            pivot_df = df_model.pivot(index='conservation', columns='gc_pos', values=f'delta_{metric}')
        else:
            pivot_df = df_model.pivot(index='conservation', columns='gc_gap', values=f'delta_{metric}')
        
        # Plot heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                   cbar_kws={'label': f'Δ{metric}'}, ax=ax)
        
        ax.set_title(f'{model_type}')
        ax.set_xlabel('GC Content (Positive)' if experiment_type == 'vanilla' else 'GC Gap')
        ax.set_ylabel('Conservation')
    
    if metric == 'EffectiveRootSkill':
        fig.suptitle(f'Change in Saliency Effective Root Skill by Hyperparameters', fontsize=14)
    else:
        fig.suptitle(f'Change in {metric} by Hyperparameters', fontsize=14)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"heatmap_{metric}.pdf"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved heatmap to {plot_path}")


def create_summary_table(df: pd.DataFrame, output_dir: Path):
    """
    Create summary table of results.
    
    Parameters:
    - df: DataFrame with results
    - output_dir: Directory to save table
    """
    # Calculate summary statistics
    summary = df.groupby(['model_type', 'gc_pos', 'conservation']).agg({
        'Accuracy': ['mean', 'std'],
        'SaliencyAUC': ['mean', 'std'],
        'SaliencySNR': ['mean', 'std'],
        'wIoU': ['mean', 'std'] if 'wIoU' in df.columns else None,
    }).round(3)
    
    # Save to CSV
    summary_path = output_dir / 'summary_table.csv'
    summary.to_csv(summary_path)
    print(f"Saved summary table to {summary_path}")
    
    return summary


def run_all_analyses(output_dir: Path, experiment_type: str = 'vanilla'):
    """
    Run all analyses and generate all plots.
    
    Parameters:
    - output_dir: Directory containing results and where to save plots
    - experiment_type: 'vanilla' or 'complex'
    """
    print(f"Running {experiment_type} experiment analysis...")
    
    # Create plots subdirectory
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Load results
    print("Loading experiment results...")
    df = load_experiment_results(output_dir)
    
    if df.empty:
        print("No results found!")
        return
        
    print(f"Loaded {len(df)} results")
    
    # Calculate delta metrics
    print("Calculating improvement metrics...")
    df_with_deltas = calculate_delta_metrics(df)
    
    # Generate plots
    metrics_to_plot = ['Accuracy', 'SaliencyAUC', 'SaliencySNR']
    if 'wIoU' in df.columns and experiment_type == 'vanilla':
        metrics_to_plot.insert(0, 'wIoU')
    
    for metric in metrics_to_plot:
        print(f"Plotting {metric}...")
        plot_metric_improvement(df_with_deltas, metric, plots_dir, experiment_type)
    
    # Plot skill metrics
    print("Plotting skill metrics...")
    plot_metric_improvement(df_with_deltas, 'EffectiveRootSkill', plots_dir, experiment_type)
    plot_metric_improvement(df_with_deltas, 'Skill', plots_dir, experiment_type)
    
    # Create heatmaps
    print("Creating heatmap visualizations...")
    plot_heatmap_grid(df_with_deltas, 'EffectiveRootSkill', plots_dir, experiment_type)
    
    # Create summary table
    print("Creating summary table...")
    create_summary_table(df, plots_dir)
    
    print(f"Analysis complete! Results saved to {plots_dir}")


# --------------------------------------------------------------------------- #
# Main Entry Point
# --------------------------------------------------------------------------- #

def main():
    """Main entry point for visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--experiment_type', type=str, default='vanilla',
                       choices=['vanilla', 'complex'],
                       help='Type of experiment')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Output directory {output_dir} does not exist!")
        return
        
    run_all_analyses(output_dir, args.experiment_type)


if __name__ == '__main__':
    main() 