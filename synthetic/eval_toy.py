#!/usr/bin/env python3
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def run_aggregation(output_dir: str) -> None:
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    npz_path = os.path.join(output_dir, "npz_results")
    if not os.path.isdir(npz_path):
        print(f"Error: The results directory '{npz_path}' could not be found.")
        sys.exit(1)

    npz_files = glob.glob(os.path.join(npz_path, '**', 'multi_seed_results.npz'), recursive=True)
    if not npz_files:
        print(f"No .npz result files found in {npz_path}; nothing to aggregate.")
        sys.exit(1)

    print(f"Found {len(npz_files)} result files, building master dataframe...")
    all_data = []
    for f_path in npz_files:
        try:
            data = np.load(f_path, allow_pickle=True)

            # Infer experiment type from path
            if 'iterative_hotflip' in f_path:
                mode = 'iterative_hotflip'
            elif 'direct_hotflip' in f_path:
                mode = 'direct_hotflip'
            else:
                mode = 'iterative_hotflip'

            is_scheduled = data.get('scheduling') and data['scheduling'].item()
            scheduling_mode = "scheduled" if is_scheduled else "no_schedule"

            gc_pos = float(data['gc_pos'].item())
            cons = float(data['conservation'].item())
            seeds = data['seeds']

            std_metrics = {
                'wIoU': data.get('std_wious', []), 'Accuracy': data.get('std_accs', []),
                'SaliencyAUC': data.get('std_aucs', []), 'SaliencySNR': data.get('std_snrs', [])
            }
            std_pgd_stats = data.get('std_pgd_stats', [{} for _ in seeds])

            rob_metrics = {
                'wIoU': data.get('rob_wious', []), 'Accuracy': data.get('rob_accs', []),
                'SaliencyAUC': data.get('rob_aucs', []), 'SaliencySNR': data.get('rob_snrs', [])
            }
            rob_pgd_stats = data.get('rob_pgd_stats', [[] for _ in seeds])

            if mode == 'iterative_hotflip':
                params = data['epsilons']
                param_name = 'epsilon'
            elif mode == 'direct_hotflip':
                params = data['direct_hotflip_epsilons'] if 'direct_hotflip_epsilons' in data else data['epsilons']
                param_name = 'epsilon'
            else:
                params = []
                param_name = 'param'

            for i, seed in enumerate(seeds):
                # Standard model results (param_val = 0)
                all_data.append({
                    'scheduling_mode': scheduling_mode, 'mode': mode,
                    'gc_pos': gc_pos, 'conservation': cons, 'seed': seed,
                    'param_name': param_name, 'param_val': 0,
                    'wIoU': std_metrics['wIoU'][i], 'Accuracy': std_metrics['Accuracy'][i],
                    'SaliencyAUC': std_metrics['SaliencyAUC'][i], 'SaliencySNR': std_metrics['SaliencySNR'][i],
                    'pgd_success_rate': std_pgd_stats[i].get('pgd_success_rate', 0),
                    'pgd_mean_iters_to_flip': std_pgd_stats[i].get('pgd_mean_iters_to_flip', 0)
                })

                # Robust model results
                for j, p_val in enumerate(params):
                    pgd_stats_list = rob_pgd_stats[i] if rob_pgd_stats is not None and i < len(rob_pgd_stats) else []
                    current_pgd_stats = pgd_stats_list[j] if pgd_stats_list is not None and j < len(pgd_stats_list) else {}

                    all_data.append({
                        'scheduling_mode': scheduling_mode, 'mode': mode,
                        'gc_pos': gc_pos, 'conservation': cons, 'seed': seed,
                        'param_name': param_name, 'param_val': p_val,
                        'wIoU': rob_metrics['wIoU'][i][j], 'Accuracy': rob_metrics['Accuracy'][i][j],
                        'SaliencyAUC': rob_metrics['SaliencyAUC'][i][j], 'SaliencySNR': rob_metrics['SaliencySNR'][i][j],
                        'pgd_success_rate': current_pgd_stats.get('pgd_success_rate', 0),
                        'pgd_mean_iters_to_flip': current_pgd_stats.get('pgd_mean_iters_to_flip', 0),
                    })
        except Exception as e:
            print(f"Could not process file {f_path}: {e}")

    df = pd.DataFrame(all_data)
    if df.empty:
        print("Master dataframe is empty, cannot generate plots.")
        sys.exit(1)

    master_csv_path = os.path.join(plots_dir, 'full_results_long_format.csv')
    df.to_csv(master_csv_path, index=False)
    print(f"Saved master data table to {master_csv_path}")

    def get_model_type(row):
        if row['param_val'] == 0:
            return 'Standard'
        if row['mode'] == 'iterative_hotflip':
            return 'Iterative HotFlip (Scheduled)' if row['scheduling_mode'] == 'scheduled' else 'Iterative HotFlip (No Schedule)'
        if row['mode'] == 'direct_hotflip':
            return 'Direct HotFlip (Scheduled)' if row['scheduling_mode'] == 'scheduled' else 'Direct HotFlip (No Schedule)'
        return 'Unknown'

    df['model_type'] = df.apply(get_model_type, axis=1)

    baseline_metrics = df[df['model_type'] == 'Standard'].groupby(
        ['gc_pos', 'conservation', 'seed']
    ).mean(numeric_only=True).reset_index()

    baseline_model_metrics = df[df['model_type'] == 'Standard'][['gc_pos', 'conservation', 'seed', 'Accuracy']].drop_duplicates()

    df_robust = df[df['model_type'] != 'Standard'].copy()

    df_robust = pd.merge(
        df_robust,
        baseline_metrics[['gc_pos', 'conservation', 'seed', 'wIoU', 'SaliencyAUC', 'SaliencySNR']],
        on=['gc_pos', 'conservation', 'seed'],
        suffixes=('', '_base')
    )
    df_robust = pd.merge(
        df_robust,
        baseline_model_metrics.rename(columns={'Accuracy': 'Accuracy_base'}),
        on=['gc_pos', 'conservation', 'seed']
    )

    metrics_to_plot = ['wIoU', 'Accuracy', 'SaliencyAUC', 'SaliencySNR']
    metric_display_names = {
        'wIoU': 'Overlap',
        'Accuracy': 'Accuracy',
        'SaliencyAUC': 'SaliencyAUC',
        'SaliencySNR': 'SaliencySNR'
    }

    def calculate_skill(auc_series):
        return 2 * auc_series - 1

    skill_robust = calculate_skill(df_robust['SaliencyAUC'])
    skill_base = calculate_skill(df_robust['SaliencyAUC_base'])
    df_robust['delta_EffectiveSkill'] = np.maximum(0, skill_robust) - np.maximum(0, skill_base)
    df_robust['delta_Skill'] = (skill_robust - skill_base) / 2.0

    for metric in metrics_to_plot:
        if metric == 'SaliencyAUC':
            df_robust[f'delta_{metric}'] = df_robust['delta_EffectiveSkill']
        else:
            df_robust[f'delta_{metric}'] = df_robust[metric] - df_robust[f'{metric}_base']

    print("\n--- Generating Combined Boxplots ---")
    df_plot_box = df_robust[df_robust['model_type'].isin([
        'Iterative HotFlip (No Schedule)', 'Iterative HotFlip (Scheduled)',
        'Direct HotFlip (No Schedule)', 'Direct HotFlip (Scheduled)'
    ])]

    for metric in metrics_to_plot:
        display_name = metric_display_names[metric]
        print(f"  - Plotting combined boxplot for {display_name}...")
        if df_plot_box.empty:
            print(f"    Skipping {display_name}, no data.")
            continue

        g = sns.catplot(
            data=df_plot_box, x="param_val", y=f"delta_{metric}",
            hue="conservation", col="model_type", row="gc_pos",
            kind="box", height=3, aspect=1.2, palette='Blues_d',
            fliersize=0, linewidth=1.0, showfliers=False, sharey=False, sharex=False,
            margin_titles=True,
            col_order=['Iterative HotFlip (No Schedule)', 'Iterative HotFlip (Scheduled)', 'Direct HotFlip (No Schedule)', 'Direct HotFlip (Scheduled)']
        )

        if metric == 'SaliencyAUC':
            g.fig.suptitle(f"Change in Saliency Effective Skill (ΔEffectiveSkill) vs. Standard", y=1.05, fontsize=16)
        else:
            g.fig.suptitle(f"Improvement in {display_name} vs. Standard, by Training Strategy", y=1.05, fontsize=16)

        for (row_idx, col_idx), ax in np.ndenumerate(g.axes):
            if row_idx >= len(g.row_names) or col_idx >= len(g.col_names):
                continue
            gc_val = g.row_names[row_idx]
            model_type = g.col_names[col_idx]
            title = f"GC={gc_val} | {model_type}"
            ax.set_title(title, fontsize=9)
            ax.axhline(0, ls='--', color='red', zorder=0)
            ax.set_xlabel("Epsilon")
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            if col_idx == 0:
                ax.set_ylabel("ΔEffectiveSkill" if metric == 'SaliencyAUC' else f"Improvement in {metric}")
            else:
                ax.set_ylabel("")

        sns.move_legend(g, "upper center", bbox_to_anchor=(.5, 0.99), ncol=len(df_plot_box['conservation'].unique()), title="Conservation", frameon=False)
        g.tight_layout(rect=[0, 0, 1, 0.95])
        plot_path = os.path.join(plots_dir, f"combined_delta_{metric}_boxplot.pdf")
        g.savefig(plot_path, dpi=300, format='pdf')
        plt.close(g.fig)
        print(f"    Saved to {plot_path}")

    print("\n--- Preparing data for absolute value plots ---")
    df_abs_plot_list = []
    df_standard_models = df[df['model_type'] == 'Standard'].copy()
    model_type_to_sched = {
        'Iterative HotFlip (No Schedule)': 'no_schedule',
        'Iterative HotFlip (Scheduled)': 'scheduled',
        'Direct HotFlip (No Schedule)': 'no_schedule',
        'Direct HotFlip (Scheduled)': 'scheduled',
    }
    for mtype, sched_mode in model_type_to_sched.items():
        df_robust_subset = df[df['model_type'] == mtype]
        df_std_subset = df_standard_models[df_standard_models['scheduling_mode'] == sched_mode].copy()
        df_std_subset['model_type'] = mtype
        df_combined = pd.concat([df_robust_subset, df_std_subset])
        df_abs_plot_list.append(df_combined)
    df_plot_abs = pd.concat(df_abs_plot_list, ignore_index=True)

    print("\n--- Generating Combined Absolute Value Boxplots ---")
    for metric in metrics_to_plot:
        print(f"  - Plotting combined boxplot for absolute {metric}...")
        if df_plot_abs.empty:
            print(f"    Skipping {metric}, no data.")
            continue
        g = sns.catplot(
            data=df_plot_abs, x="param_val", y=metric,
            hue="conservation", col="model_type", row="gc_pos",
            kind="box", height=3, aspect=1.2, palette='Blues_d',
            fliersize=0, linewidth=1.0, showfliers=False, sharey=False, sharex=False,
            margin_titles=True,
            col_order=['Iterative HotFlip (No Schedule)', 'Iterative HotFlip (Scheduled)', 'Direct HotFlip (No Schedule)', 'Direct HotFlip (Scheduled)']
        )
        g.fig.suptitle(f"Absolute {metric} by Training Strategy", y=1.05, fontsize=16)
        for (row_idx, col_idx), ax in np.ndenumerate(g.axes):
            if row_idx >= len(g.row_names) or col_idx >= len(g.col_names):
                continue
            gc_val = g.row_names[row_idx]
            model_type = g.col_names[col_idx]
            title = f"GC={gc_val} | {model_type}"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Epsilon")
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"Absolute {metric}")
            else:
                ax.set_ylabel("")
        sns.move_legend(g, "upper center", bbox_to_anchor=(.5, 0.99), ncol=len(df_plot_abs['conservation'].unique()), title="Conservation", frameon=False)
        g.tight_layout(rect=[0, 0, 1, 0.95])
        plot_path = os.path.join(plots_dir, f"combined_absolute_{metric}_boxplot.pdf")
        g.savefig(plot_path, dpi=300, format='pdf')
        plt.close(g.fig)
        print(f"    Saved to {plot_path}")

    print("\n--- Generating 2x2 Heatmap Grid ---")
    best_eps_df = df_robust.groupby(['model_type', 'gc_pos', 'conservation', 'param_val']).agg({
        'SaliencyAUC': 'mean'
    }).reset_index()
    idx_best = best_eps_df.groupby(['model_type', 'gc_pos', 'conservation'])['SaliencyAUC'].idxmax()
    best_eps_df = best_eps_df.loc[idx_best]
    df_best_eps = pd.merge(df_robust, best_eps_df[['model_type', 'gc_pos', 'conservation', 'param_val']], on=['model_type', 'gc_pos', 'conservation', 'param_val'])
    df_heatmap = df_best_eps.groupby(['model_type', 'gc_pos', 'conservation']).agg({'delta_SaliencyAUC': 'mean'}).reset_index()
    model_types_for_heatmap = [
        'Iterative HotFlip (No Schedule)', 'Iterative HotFlip (Scheduled)',
        'Direct HotFlip (No Schedule)', 'Direct HotFlip (Scheduled)'
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Change in Saliency Effective Skill (ΔEffectiveSkill) by Signal and Confounder Strength\n(Best ε selected per combination)', fontsize=14)
    for idx, model_type in enumerate(model_types_for_heatmap):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        df_model = df_heatmap[df_heatmap['model_type'] == model_type]
        if not df_model.empty:
            pivot_df = df_model.pivot(index='conservation', columns='gc_pos', values='delta_SaliencyAUC')
            sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='coolwarm', center=0, cbar_kws={'label': 'ΔEffectiveSkill'}, ax=ax, vmin=-1, vmax=1)
            ax.set_title(model_type)
            ax.set_xlabel('GC Content (Confounder Strength)')
            ax.set_ylabel('Conservation (Signal Strength)')
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_type)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(plots_dir, "heatmap_grid_saliency_auc_improvement.pdf")
    plt.savefig(plot_path, dpi=300, format='pdf')
    plt.close(fig)
    print(f"  Saved heatmap grid to {plot_path}")

    # Best method heatmap
    print("\n--- Generating Best Method Heatmap ---")
    df_best_method = df_heatmap.loc[df_heatmap.groupby(['gc_pos', 'conservation'])['delta_SaliencyAUC'].idxmax()]
    method_map = {
        'Iterative HotFlip (No Schedule)': 'Iter-NoSched',
        'Iterative HotFlip (Scheduled)': 'Iter-Sched',
        'Direct HotFlip (No Schedule)': 'Direct-NoSched',
        'Direct HotFlip (Scheduled)': 'Direct-Sched'
    }
    method_order = list(method_map.values())
    df_best_method['method_short'] = df_best_method['model_type'].map(method_map)
    pivot_values = df_best_method.pivot(index='conservation', columns='gc_pos', values='delta_SaliencyAUC')
    pivot_methods_categorical = df_best_method.pivot(index='conservation', columns='gc_pos', values='method_short')
    pivot_methods_int = pivot_methods_categorical.applymap(lambda x: method_order.index(x) if pd.notna(x) else -1)
    base_cmap = plt.get_cmap('tab10', len(method_order))
    opaque_colors = base_cmap.colors
    transparent_colors = base_cmap.colors.copy()
    transparent_colors[:, 3] = 0.2
    new_colors = np.vstack((opaque_colors, transparent_colors))
    new_cmap = ListedColormap(new_colors)
    pivot_coloring = pivot_methods_int.copy()
    low_improvement_mask = pivot_values < 0.1
    pivot_coloring[low_improvement_mask] += len(method_order)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot_coloring, annot=pivot_values, fmt='.2f', cmap=new_cmap, ax=ax, linewidths=.5, cbar=False, annot_kws={'fontsize': 9})
    norm = plt.cm.colors.BoundaryNorm(np.arange(len(method_order) + 1) - 0.5, base_cmap.N)
    sm = plt.cm.ScalarMappable(cmap=base_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=np.arange(len(method_order)))
    cbar.set_ticklabels(method_order)
    cbar.set_label('Best Performing Method', rotation=270, labelpad=20)
    ax.set_title('Best Performing Method and Improvement (ΔEffectiveSkill)\nby Signal and Confounder Strength (cells with < 0.1 improvement are faded)', fontsize=14)
    ax.set_xlabel('GC Content (Confounder Strength)')
    ax.set_ylabel('Conservation (Signal Strength)')
    ax.invert_yaxis()
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "best_method_heatmap.pdf")
    plt.savefig(plot_path, dpi=300, format='pdf')
    plt.close(fig)
    print(f"  Saved best method heatmap to {plot_path}")

    # Linear skill heatmap
    print("\n--- Generating Linear Skill (ΔSkill) Heatmap Grid ---")
    df_heatmap_skill = df_best_eps.groupby(['model_type', 'gc_pos', 'conservation']).agg({'delta_Skill': 'mean'}).reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Change in Linear Saliency Skill (Normalized ΔSkill) by Signal and Confounder Strength\n(Models selected by best ΔEffectiveSkill)', fontsize=14)
    for idx, model_type in enumerate(model_types_for_heatmap):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        df_model = df_heatmap_skill[df_heatmap_skill['model_type'] == model_type]
        if not df_model.empty:
            pivot_df = df_model.pivot(index='conservation', columns='gc_pos', values='delta_Skill')
            sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='coolwarm', center=0, cbar_kws={'label': 'Normalized ΔSkill (Linear)'}, ax=ax, vmin=-1, vmax=1)
            ax.set_title(model_type)
            ax.set_xlabel('GC Content (Confounder Strength)')
            ax.set_ylabel('Conservation (Signal Strength)')
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_type)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(plots_dir, "heatmap_grid_linear_skill.pdf")
    plt.savefig(plot_path, dpi=300, format='pdf')
    plt.close(fig)
    print(f"  Saved linear skill heatmap to {plot_path}")

    # --- Absolute SaliencyAUC Heatmap Grid (Standard vs Best Robust) ---
    print("\n--- Generating Absolute SaliencyAUC Heatmap Grid (Standard vs Best Robust) ---")

    # Standard model
    df_std_heatmap = df[df['model_type'] == 'Standard'].groupby(['gc_pos', 'conservation']).agg(
        SaliencyAUC=('SaliencyAUC', 'mean')
    ).reset_index()
    pivot_std = df_std_heatmap.pivot(index='conservation', columns='gc_pos', values='SaliencyAUC')

    # Best Robust model by SaliencyAUC across model types
    df_best_robust_agg = df_best_eps.groupby(['gc_pos', 'conservation', 'model_type']).agg(
        SaliencyAUC=('SaliencyAUC', 'mean')
    ).reset_index()
    idx_best_overall = df_best_robust_agg.groupby(['gc_pos', 'conservation'])['SaliencyAUC'].idxmax()
    df_best_overall = df_best_robust_agg.loc[idx_best_overall]
    pivot_best_robust = df_best_overall.pivot(index='conservation', columns='gc_pos', values='SaliencyAUC')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    fig.suptitle('Absolute SaliencyAUC', fontsize=16, y=1.02)
    sns.heatmap(pivot_std, annot=True, fmt='.2f', cmap='coolwarm', center=0.5, vmin=0, vmax=1, ax=axes[0], cbar_kws={'label': 'SaliencyAUC'})
    axes[0].set_title('Standard Model')
    axes[0].set_xlabel('GC Content (Confounder Strength)')
    axes[0].set_ylabel('Conservation (Signal Strength)')
    axes[0].invert_yaxis()
    sns.heatmap(pivot_best_robust, annot=True, fmt='.2f', cmap='coolwarm', center=0.5, vmin=0, vmax=1, ax=axes[1], cbar_kws={'label': 'SaliencyAUC'})
    axes[1].set_title('Best Robust Model (by SaliencyAUC)')
    axes[1].set_xlabel('GC Content (Confounder Strength)')
    axes[1].set_ylabel('')
    axes[1].invert_yaxis()
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plot_path = os.path.join(plots_dir, "heatmap_absolute_saliency_auc.pdf")
    plt.savefig(plot_path, dpi=300, format='pdf')
    plt.close(fig)
    print(f"  Saved absolute SaliencyAUC heatmap to {plot_path}")

    # --- Absolute Overlap Heatmap Grid (Standard vs Best Robust) ---
    print("\n--- Generating Absolute Overlap Heatmap Grid (Standard vs Best Robust) ---")
    df_std_overlap = df[df['model_type'] == 'Standard'].groupby(['gc_pos', 'conservation']).agg(
        wIoU=('wIoU', 'mean')
    ).reset_index()
    pivot_std_overlap = df_std_overlap.pivot(index='conservation', columns='gc_pos', values='wIoU')

    df_best_robust_overlap = df_best_eps.groupby(['gc_pos', 'conservation', 'model_type']).agg(
        wIoU=('wIoU', 'mean')
    ).reset_index()
    idx_best_overlap = df_best_robust_overlap.groupby(['gc_pos', 'conservation'])['wIoU'].idxmax()
    df_best_overall_overlap = df_best_robust_overlap.loc[idx_best_overlap]
    pivot_best_robust_overlap = df_best_overall_overlap.pivot(index='conservation', columns='gc_pos', values='wIoU')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    fig.suptitle('Absolute Overlap', fontsize=16, y=1.02)
    sns.heatmap(pivot_std_overlap, annot=True, fmt='.2f', cmap='Reds', vmin=0, vmax=1, ax=axes[0], cbar_kws={'label': 'Overlap'})
    axes[0].set_title('Standard Model')
    axes[0].set_xlabel('GC Content (Confounder Strength)')
    axes[0].set_ylabel('Conservation (Signal Strength)')
    axes[0].invert_yaxis()
    sns.heatmap(pivot_best_robust_overlap, annot=True, fmt='.2f', cmap='Reds', vmin=0, vmax=1, ax=axes[1], cbar_kws={'label': 'Overlap'})
    axes[1].set_title('Best Robust Model (by Overlap)')
    axes[1].set_xlabel('GC Content (Confounder Strength)')
    axes[1].set_ylabel('')
    axes[1].invert_yaxis()
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plot_path = os.path.join(plots_dir, "heatmap_absolute_overlap.pdf")
    plt.savefig(plot_path, dpi=300, format='pdf')
    plt.close(fig)
    print(f"  Saved absolute Overlap heatmap to {plot_path}")

    # --- Absolute SaliencySNR Heatmap Grid (Standard vs Best Robust) ---
    print("\n--- Generating Absolute SaliencySNR Heatmap Grid (Standard vs Best Robust) ---")
    df_std_snr = df[df['model_type'] == 'Standard'].groupby(['gc_pos', 'conservation']).agg(
        SaliencySNR=('SaliencySNR', 'mean')
    ).reset_index()
    pivot_std_snr = df_std_snr.pivot(index='conservation', columns='gc_pos', values='SaliencySNR')

    df_best_robust_snr = df_best_eps.groupby(['gc_pos', 'conservation', 'model_type']).agg(
        SaliencySNR=('SaliencySNR', 'mean')
    ).reset_index()
    idx_best_snr = df_best_robust_snr.groupby(['gc_pos', 'conservation'])['SaliencySNR'].idxmax()
    df_best_overall_snr = df_best_robust_snr.loc[idx_best_snr]
    pivot_best_robust_snr = df_best_overall_snr.pivot(index='conservation', columns='gc_pos', values='SaliencySNR')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    fig.suptitle('Absolute SaliencySNR', fontsize=16, y=1.02)
    sns.heatmap(pivot_std_snr, annot=True, fmt='.2f', cmap='Reds', vmin=0, vmax=1, ax=axes[0], cbar_kws={'label': 'SaliencySNR'})
    axes[0].set_title('Standard Model')
    axes[0].set_xlabel('GC Content (Confounder Strength)')
    axes[0].set_ylabel('Conservation (Signal Strength)')
    axes[0].invert_yaxis()
    sns.heatmap(pivot_best_robust_snr, annot=True, fmt='.2f', cmap='Reds', vmin=0, vmax=1, ax=axes[1], cbar_kws={'label': 'SaliencySNR'})
    axes[1].set_title('Best Robust Model (by SaliencySNR)')
    axes[1].set_xlabel('GC Content (Confounder Strength)')
    axes[1].set_ylabel('')
    axes[1].invert_yaxis()
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plot_path = os.path.join(plots_dir, "heatmap_absolute_saliency_snr.pdf")
    plt.savefig(plot_path, dpi=300, format='pdf')
    plt.close(fig)
    print(f"  Saved absolute SaliencySNR heatmap to {plot_path}")

    print("\nAll plotting complete.")


if __name__ == '__main__':
    # Minimal CLI: python synthetic/eval_toy.py --output_dir RESULTS_ROOT
    import argparse
    p = argparse.ArgumentParser(description='Aggregate and plot toy_slurm results')
    p.add_argument('--output_dir', type=str, required=True)
    args = p.parse_args()
    run_aggregation(args.output_dir)

