#!/usr/bin/env python
"""
Compares interpretability performance of attribution scores for task 3 with different flip fractions

Figures generated from this script include:
- Comparison of different flip fractions (0.01, 0.05, 0.1) for robust training
- Standard vs robust training comparisons
- Model performance across different configurations
"""

import os
import numpy as np
from six.moves import cPickle
import matplotlib.pyplot as plt
import helper

# Configuration
results_path = os.path.join('koo/results', 'task3_robust_3models')
params_path = os.path.join(results_path, 'model_params')
save_path = os.path.join(results_path, 'scores')

# Load data
data_path = 'koo/data/synthetic_code_dataset.h5'
data = helper.load_data(data_path)
x_train, y_train, x_valid, y_valid, x_test, y_test = data

# Function to calculate saliency SNR (vectorized)
def calculate_saliency_snr(X, score, X_model, precomputed_masks=None):
    """
    Calculate saliency SNR (Signal-to-Noise Ratio) for attribution scores.

    SNR measures the fraction of attribution "energy" (sum of squared scores)
    within the true causal motif relative to the total energy.

    Args:
        X: Input sequences
        score: Attribution scores
        X_model: Ground truth model (PWM)
        precomputed_masks: Optional pre-computed boolean masks for ground truth positions

    Returns:
        Array of SNR values for each sequence
    """
    # Sum scores across nucleotide dimension
    score_summed = np.sum(score, axis=2)  # (N, L)
    score_sq = score_summed ** 2

    # Compute masks if not provided
    if precomputed_masks is None:
        # Vectorized: (N, A, L) -> (N, L)
        gt_info = np.log2(4) + np.sum(X_model * np.log2(X_model + 1e-10), axis=1)
        masks = gt_info > 0.01
    else:
        masks = precomputed_masks

    # Vectorized SNR calculation
    sum_sq_inside = np.sum(score_sq * masks, axis=1)
    sum_sq_total = np.sum(score_sq, axis=1)
    snr_scores = sum_sq_inside / (sum_sq_total + 1e-9)

    return snr_scores


def precompute_ground_truth_masks(X_model):
    """Pre-compute ground truth masks (constant across trials)."""
    gt_info = np.log2(4) + np.sum(X_model * np.log2(X_model + 1e-10), axis=1)
    return gt_info > 0.01


def calculate_interpretability_batch(X, scores_all_trials, X_model, masks, labels):
    """
    Calculate interpretability metrics for all trials at once.

    Args:
        X: Input sequences (N, L, A)
        scores_all_trials: Attribution scores for all trials (T, N, L, A)
        X_model: Ground truth model (N, A, L)
        masks: Pre-computed ground truth masks (N, L) for SNR
        labels: Pre-computed labels (N, L) for AUROC/AUPR

    Returns:
        roc_scores: (T,) mean AUROC per trial
        pr_scores: (T,) mean AUPR per trial
        snr_scores: (T,) mean SNR per trial
    """
    num_trials = scores_all_trials.shape[0]

    roc_all = []
    pr_all = []
    snr_all = []

    for trial in range(num_trials):
        trial_scores = scores_all_trials[trial] * X
        roc_score, pr_score = helper.interpretability_performance(
            X, trial_scores, X_model, precomputed_labels=labels
        )
        snr_score = calculate_saliency_snr(X, trial_scores, X_model, precomputed_masks=masks)

        roc_all.append(np.mean(roc_score))
        pr_all.append(np.mean(pr_score))
        snr_all.append(np.mean(snr_score))

    return np.array(roc_all), np.array(pr_all), np.array(snr_all)

# Load ground truth values
test_model = helper.load_synthetic_models(data_path, dataset='test')
true_index = np.where(y_test[:,0] == 1)[0]
X = x_test[true_index][:500]
X_model = test_model[true_index][:500]

# Pre-compute ground truth masks and labels once (constant across all trials and models)
gt_masks = precompute_ground_truth_masks(X_model)
gt_labels = gt_masks.astype(np.float32)  # Same computation, just cast for sklearn
print(f"Pre-computed ground truth masks: {gt_masks.shape}")

# Calculate interpretability performance
num_trials = 10
model_names = ['cnn-local', 'cnn-dist', 'cnn-local-deep']
activations = ['relu', 'exponential']

# Display labels for plots (cnn-local-deep was mislabeled as cnn-dist in original repo)
MODEL_LABELS = {
    'cnn-local': 'CNN-Local',
    'cnn-dist': 'CNN-Dist',
    'cnn-local-deep': 'CNN-Local-Deep',
}
training_modes = ['robust', 'standard']
flip_fractions = [0.01, 0.05, 0.1, 0.15, 0.2]
score_names = ['integrated_scores']  # Only using integrated gradients now

results = {}
for model_name in model_names:
    for activation in activations:
        for training_mode in training_modes:
            
            # set flip fraction for robust training
            if training_mode == 'robust':
                loop_fractions = flip_fractions
            else:
                loop_fractions = [0.0] # for standard training, loop once without flip

            for flip_fraction in loop_fractions:
            
                # create base name
                if training_mode == 'robust':
                    name = f"{model_name}_{activation}_robust_{flip_fraction}"
                else:
                    name = f"{model_name}_{activation}_standard"
                results[name] = {}
                    
                file_path = os.path.join(save_path, f"{name}.pickle")
                try:
                    with open(file_path, 'rb') as f:
                        integrated_scores = cPickle.load(f)

                    # Use batch processing with pre-computed masks and labels
                    scores_array = np.array(integrated_scores)  # (T, N, L, A)
                    shap_roc, shap_pr, shap_snr = calculate_interpretability_batch(
                        X, scores_array, X_model, gt_masks, gt_labels
                    )

                    results[name]['integrated_scores'] = [shap_roc, shap_pr, shap_snr]
                    print('%s: %.4f+/-%.4f (SNR: %.4f+/-%.4f)\t'%(name+'_integrated_scores',
                                               np.mean(shap_roc), np.std(shap_roc),
                                               np.mean(shap_snr), np.std(shap_snr)))
                except FileNotFoundError:
                    print(f"File not found: {file_path}")
                    continue
                
                # Load shuffle baseline results for all models (full ablation)
                shuffle_file_path = os.path.join(save_path, f"{name}_shuffle.pickle")
                try:
                    with open(shuffle_file_path, 'rb') as f:
                        integrated_scores_shuffle = cPickle.load(f)

                    # Use batch processing with pre-computed masks and labels
                    scores_array = np.array(integrated_scores_shuffle)
                    shap_roc, shap_pr, shap_snr = calculate_interpretability_batch(
                        X, scores_array, X_model, gt_masks, gt_labels
                    )

                    results[name]['integrated_scores_shuffle'] = [shap_roc, shap_pr, shap_snr]
                    print('%s: %.4f+/-%.4f (SNR: %.4f+/-%.4f)\t'%(name+'_integrated_scores_shuffle',
                                               np.mean(shap_roc), np.std(shap_roc),
                                               np.mean(shap_snr), np.std(shap_snr)))
                except FileNotFoundError:
                    print(f"Shuffle baseline file not found: {shuffle_file_path}")

# Save results
file_path = os.path.join(results_path, 'task3_robust_attr_results.pickle')
with open(file_path, 'wb') as f:
    cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)

# Print results table
print("\nGenerating results tables...")

# Load results
file_path = os.path.join(results_path, 'task3_robust_attr_results.pickle')
with open(file_path, 'rb') as f:
    results = cPickle.load(f)

# Save AUROC results
save_path_auroc = os.path.join(results_path, 'task3_robust_attr_results_auroc.tsv')
with open(save_path_auroc, 'w') as f:
    f.write('model\tintegrated_gradients_auroc\n')
    for model_name in model_names:
        for activation in activations:
            for training_mode in training_modes:
                
                # set flip fraction for robust training
                if training_mode == 'robust':
                    loop_fractions = flip_fractions
                else:
                    loop_fractions = [0.0] # for standard training, loop once without flip

                for flip_fraction in loop_fractions:
                
                    # create base name
                    if training_mode == 'robust':
                        name = f"{model_name}_{activation}_robust_{flip_fraction}"
                    else:
                        name = f"{model_name}_{activation}_standard"
                    
                    if name in results and 'integrated_scores' in results[name]:
                        f.write('%s\t'%(name)) 
                        f.write('%.4f+/-%.4f\t'%(np.mean(results[name]['integrated_scores'][0]), 
                                                 np.std(results[name]['integrated_scores'][0])))
                        f.write('\n')

# Save AUPR results
save_path_aupr = os.path.join(results_path, 'task3_robust_attr_results_aupr.tsv')
with open(save_path_aupr, 'w') as f:
    f.write('model\tintegrated_gradients_aupr\n')
    for model_name in model_names:
        for activation in activations:
            for training_mode in training_modes:
                
                # set flip fraction for robust training
                if training_mode == 'robust':
                    loop_fractions = flip_fractions
                else:
                    loop_fractions = [0.0] # for standard training, loop once without flip

                for flip_fraction in loop_fractions:
                
                    # create base name
                    if training_mode == 'robust':
                        name = f"{model_name}_{activation}_robust_{flip_fraction}"
                    else:
                        name = f"{model_name}_{activation}_standard"
                    
                    if name in results and 'integrated_scores' in results[name]:
                        f.write('%s\t'%(name)) 
                        f.write('%.4f+/-%.4f\t'%(np.mean(results[name]['integrated_scores'][1]), 
                                                 np.std(results[name]['integrated_scores'][1])))
                        f.write('\n')

# Save SNR results
save_path_snr = os.path.join(results_path, 'task3_robust_attr_results_snr.tsv')
with open(save_path_snr, 'w') as f:
    f.write('model\tintegrated_gradients_snr\n')
    for model_name in model_names:
        for activation in activations:
            for training_mode in training_modes:
                
                # set flip fraction for robust training
                if training_mode == 'robust':
                    loop_fractions = flip_fractions
                else:
                    loop_fractions = [0.0] # for standard training, loop once without flip

                for flip_fraction in loop_fractions:
                
                    # create base name
                    if training_mode == 'robust':
                        name = f"{model_name}_{activation}_robust_{flip_fraction}"
                    else:
                        name = f"{model_name}_{activation}_standard"
                    
                    if name in results and 'integrated_scores' in results[name]:
                        f.write('%s\t'%(name)) 
                        f.write('%.4f+/-%.4f\t'%(np.mean(results[name]['integrated_scores'][2]), 
                                                 np.std(results[name]['integrated_scores'][2])))
                        f.write('\n')

# Generate plots
print("\nGenerating plots...")

# Helper function to find best robust flip fraction
def find_best_robust(model_name, activation, metric_idx):
    """Find the flip fraction with highest mean for given metric (0=AUROC, 1=AUPR, 2=SNR)"""
    best_flip = None
    best_mean = -1
    best_scores = None
    for flip_fraction in flip_fractions:
        robust_name = f"{model_name}_{activation}_robust_{flip_fraction}"
        if robust_name in results and 'integrated_scores' in results[robust_name]:
            scores = results[robust_name]['integrated_scores'][metric_idx]
            if np.mean(scores) > best_mean:
                best_mean = np.mean(scores)
                best_flip = flip_fraction
                best_scores = scores
    return best_flip, best_scores

# =============================================================================
# COMPACT COMBINED FIGURE: All metrics in one journal-ready figure
# =============================================================================
print("Generating compact combined figure...")

# Collect data for compact plot
configs = []
std_data = {'auroc': [], 'aupr': [], 'snr': []}
rob_data = {'auroc': [], 'aupr': [], 'snr': []}
best_flips = []

for model_name in model_names:
    for activation in activations:
        model_label = MODEL_LABELS.get(model_name, model_name)
        act_short = 'exp' if activation == 'exponential' else activation
        configs.append(f"{model_label}\n{act_short}")

        standard_name = f"{model_name}_{activation}_standard"

        # Standard (shuffle baseline)
        if standard_name in results and 'integrated_scores_shuffle' in results[standard_name]:
            std_data['auroc'].append(results[standard_name]['integrated_scores_shuffle'][0])
            std_data['aupr'].append(results[standard_name]['integrated_scores_shuffle'][1])
            std_data['snr'].append(results[standard_name]['integrated_scores_shuffle'][2])
        else:
            std_data['auroc'].append(np.array([np.nan]))
            std_data['aupr'].append(np.array([np.nan]))
            std_data['snr'].append(np.array([np.nan]))

        # Best robust (by AUROC)
        best_flip, _ = find_best_robust(model_name, activation, 0)
        best_flips.append(best_flip)
        if best_flip is not None:
            robust_name = f"{model_name}_{activation}_robust_{best_flip}"
            rob_data['auroc'].append(results[robust_name]['integrated_scores'][0])
            rob_data['aupr'].append(results[robust_name]['integrated_scores'][1])
            rob_data['snr'].append(results[robust_name]['integrated_scores'][2])
        else:
            rob_data['auroc'].append(np.array([np.nan]))
            rob_data['aupr'].append(np.array([np.nan]))
            rob_data['snr'].append(np.array([np.nan]))

n_configs = len(configs)
metrics = ['auroc', 'aupr', 'snr']
metric_labels = ['AUROC', 'AUPR', 'SNR']

# Create compact figure: 1 row × 3 columns (one per metric)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
plt.subplots_adjust(wspace=0.30, left=0.06, right=0.98, bottom=0.18, top=0.88)

x = np.arange(n_configs)
width = 0.35

colors_std = '#4878A8'  # blue
colors_rob = '#E85D4C'  # red

# Shorter config labels (single line)
config_labels = []
for model_name in model_names:
    for activation in activations:
        model_short = {'cnn-local': 'Local', 'cnn-dist': 'Dist', 'cnn-local-deep': 'Deep'}
        act_short = 'E' if activation == 'exponential' else 'R'
        config_labels.append(f"{model_short.get(model_name, model_name)}-{act_short}")

for ax_idx, (metric, ylabel) in enumerate(zip(metrics, metric_labels)):
    ax = axes[ax_idx]

    std_means = [np.mean(d) for d in std_data[metric]]
    std_stds = [np.std(d) for d in std_data[metric]]
    rob_means = [np.mean(d) for d in rob_data[metric]]
    rob_stds = [np.std(d) for d in rob_data[metric]]

    bars1 = ax.bar(x - width/2, std_means, width, yerr=std_stds,
                   label='Std', color=colors_std, capsize=2,
                   error_kw={'linewidth': 0.8})
    bars2 = ax.bar(x + width/2, rob_means, width, yerr=rob_stds,
                   label='Rb', color=colors_rob, capsize=2,
                   error_kw={'linewidth': 0.8})

    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=8, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=8)

    # Set y limits with some padding
    all_vals = std_means + rob_means + [m + s for m, s in zip(std_means, std_stds)] + \
               [m + s for m, s in zip(rob_means, rob_stds)]
    valid_vals = [v for v in all_vals if not np.isnan(v)]
    if valid_vals:
        ymin = min(0.3, min(valid_vals) - 0.05)
        ymax = max(valid_vals) + 0.05
        ax.set_ylim(ymin, min(1.0, ymax))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Single legend at top center
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=9, frameon=False)

fig.savefig(os.path.join(results_path, 'task3_combined_metrics.pdf'),
            format='pdf', dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(results_path, 'task3_combined_metrics.svg'),
            format='svg', bbox_inches='tight')
plt.close()

print("Saved: task3_combined_metrics.pdf/svg")

# =============================================================================
# APPENDIX: Full 2x2 ablation (Standard/Robust × Shuffle/PGD)
# =============================================================================
print("Generating appendix combined figure (full 2x2 ablation)...")

# Collect Standard PGD baseline data
std_pgd_data = {'auroc': [], 'aupr': [], 'snr': []}
for model_name in model_names:
    for activation in activations:
        standard_name = f"{model_name}_{activation}_standard"
        if standard_name in results and 'integrated_scores' in results[standard_name]:
            std_pgd_data['auroc'].append(results[standard_name]['integrated_scores'][0])
            std_pgd_data['aupr'].append(results[standard_name]['integrated_scores'][1])
            std_pgd_data['snr'].append(results[standard_name]['integrated_scores'][2])
        else:
            std_pgd_data['auroc'].append(np.array([np.nan]))
            std_pgd_data['aupr'].append(np.array([np.nan]))
            std_pgd_data['snr'].append(np.array([np.nan]))

# Collect Robust shuffle baseline data
rob_shuffle_data = {'auroc': [], 'aupr': [], 'snr': []}
for model_name in model_names:
    for activation in activations:
        # Find best robust flip fraction
        best_flip, _ = find_best_robust(model_name, activation, 0)
        if best_flip is not None:
            robust_name = f"{model_name}_{activation}_robust_{best_flip}"
            if robust_name in results and 'integrated_scores_shuffle' in results[robust_name]:
                rob_shuffle_data['auroc'].append(results[robust_name]['integrated_scores_shuffle'][0])
                rob_shuffle_data['aupr'].append(results[robust_name]['integrated_scores_shuffle'][1])
                rob_shuffle_data['snr'].append(results[robust_name]['integrated_scores_shuffle'][2])
            else:
                rob_shuffle_data['auroc'].append(np.array([np.nan]))
                rob_shuffle_data['aupr'].append(np.array([np.nan]))
                rob_shuffle_data['snr'].append(np.array([np.nan]))
        else:
            rob_shuffle_data['auroc'].append(np.array([np.nan]))
            rob_shuffle_data['aupr'].append(np.array([np.nan]))
            rob_shuffle_data['snr'].append(np.array([np.nan]))

# Create appendix figure: 1 row × 3 columns with 4 bars per group
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
plt.subplots_adjust(wspace=0.30, left=0.05, right=0.98, bottom=0.20, top=0.85)

x = np.arange(n_configs)
width = 0.18

colors_std_shuffle = '#4878A8'  # blue - standard shuffle
colors_std_pgd = '#7FB3D5'      # light blue - standard PGD
colors_rob_shuffle = '#F5B041'  # orange - robust shuffle
colors_rob_pgd = '#E85D4C'      # red - robust PGD

for ax_idx, (metric, ylabel) in enumerate(zip(metrics, metric_labels)):
    ax = axes[ax_idx]

    std_shuffle_means = [np.mean(d) for d in std_data[metric]]
    std_shuffle_stds = [np.std(d) for d in std_data[metric]]
    std_pgd_means = [np.mean(d) for d in std_pgd_data[metric]]
    std_pgd_stds = [np.std(d) for d in std_pgd_data[metric]]
    rob_shuffle_means = [np.mean(d) for d in rob_shuffle_data[metric]]
    rob_shuffle_stds = [np.std(d) for d in rob_shuffle_data[metric]]
    rob_pgd_means = [np.mean(d) for d in rob_data[metric]]
    rob_pgd_stds = [np.std(d) for d in rob_data[metric]]

    ax.bar(x - 1.5*width, std_shuffle_means, width, yerr=std_shuffle_stds,
           label='Std-S', color=colors_std_shuffle, capsize=2,
           error_kw={'linewidth': 0.8})
    ax.bar(x - 0.5*width, std_pgd_means, width, yerr=std_pgd_stds,
           label='Std-P', color=colors_std_pgd, capsize=2,
           error_kw={'linewidth': 0.8})
    ax.bar(x + 0.5*width, rob_shuffle_means, width, yerr=rob_shuffle_stds,
           label='Rb-S', color=colors_rob_shuffle, capsize=2,
           error_kw={'linewidth': 0.8})
    ax.bar(x + 1.5*width, rob_pgd_means, width, yerr=rob_pgd_stds,
           label='Rb-P', color=colors_rob_pgd, capsize=2,
           error_kw={'linewidth': 0.8})

    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=8, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=8)

    all_vals = std_shuffle_means + std_pgd_means + rob_shuffle_means + rob_pgd_means
    valid_vals = [v for v in all_vals if not np.isnan(v)]
    if valid_vals:
        ymin = min(0.3, min(valid_vals) - 0.05)
        ymax = max(valid_vals) + 0.08
        ax.set_ylim(ymin, min(1.0, ymax))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Single legend at top center
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=9, frameon=False)

fig.savefig(os.path.join(results_path, 'task3_combined_metrics_appendix.pdf'),
            format='pdf', dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(results_path, 'task3_combined_metrics_appendix.svg'),
            format='svg', bbox_inches='tight')
plt.close()

print("Saved: task3_combined_metrics_appendix.pdf/svg (full 2x2 ablation)")

# =============================================================================
# Keep original separate boxplots (for detailed inspection)
# =============================================================================

# Plot AUROC comparison: Standard vs Best Robust
n_models = len(model_names)
n_activations = len(activations)
fig = plt.figure(figsize=(4 * n_activations, 4 * n_models))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

subplot_idx = 1
for model_name in model_names:
    for activation in activations:
        ax = plt.subplot(n_models, n_activations, subplot_idx)

        vals = []
        labels = []

        # Standard training (use shuffle baseline for fair comparison)
        standard_name = f"{model_name}_{activation}_standard"
        if standard_name in results:
            if 'integrated_scores_shuffle' in results[standard_name]:
                vals.append(results[standard_name]['integrated_scores_shuffle'][0])
                labels.append('Standard\n(shuffle)')
            elif 'integrated_scores' in results[standard_name]:
                vals.append(results[standard_name]['integrated_scores'][0])
                labels.append('Standard\n(PGD)')

        # Best robust training
        best_flip, best_scores = find_best_robust(model_name, activation, 0)
        if best_scores is not None:
            vals.append(best_scores)
            labels.append(f'Robust\n(f={best_flip})')

        if vals:
            ax.boxplot(vals, widths=0.6)
            ax.set_ylabel('AUROC', fontsize=10)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, fontsize=9)
            model_label = MODEL_LABELS.get(model_name, model_name)
            ax.set_title(f"{model_label}, {activation}", fontsize=11)

        subplot_idx += 1

outfile = os.path.join(results_path, 'task3_standard_vs_best_robust_auroc.pdf')
fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
plt.close()

# Plot AUPR comparison: Standard vs Best Robust
fig = plt.figure(figsize=(4 * n_activations, 4 * n_models))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

subplot_idx = 1
for model_name in model_names:
    for activation in activations:
        ax = plt.subplot(n_models, n_activations, subplot_idx)

        vals = []
        labels = []

        # Standard training (use shuffle baseline for fair comparison)
        standard_name = f"{model_name}_{activation}_standard"
        if standard_name in results:
            if 'integrated_scores_shuffle' in results[standard_name]:
                vals.append(results[standard_name]['integrated_scores_shuffle'][1])
                labels.append('Standard\n(shuffle)')
            elif 'integrated_scores' in results[standard_name]:
                vals.append(results[standard_name]['integrated_scores'][1])
                labels.append('Standard\n(PGD)')

        # Best robust training
        best_flip, best_scores = find_best_robust(model_name, activation, 1)
        if best_scores is not None:
            vals.append(best_scores)
            labels.append(f'Robust\n(f={best_flip})')

        if vals:
            ax.boxplot(vals, widths=0.6)
            ax.set_ylabel('AUPR', fontsize=10)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, fontsize=9)
            model_label = MODEL_LABELS.get(model_name, model_name)
            ax.set_title(f"{model_label}, {activation}", fontsize=11)

        subplot_idx += 1

outfile = os.path.join(results_path, 'task3_standard_vs_best_robust_aupr.pdf')
fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
plt.close()

# Plot SNR comparison: Standard vs Best Robust
fig = plt.figure(figsize=(4 * n_activations, 4 * n_models))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

subplot_idx = 1
for model_name in model_names:
    for activation in activations:
        ax = plt.subplot(n_models, n_activations, subplot_idx)

        vals = []
        labels = []

        # Standard training (use shuffle baseline for fair comparison)
        standard_name = f"{model_name}_{activation}_standard"
        if standard_name in results:
            if 'integrated_scores_shuffle' in results[standard_name]:
                vals.append(results[standard_name]['integrated_scores_shuffle'][2])
                labels.append('Standard\n(shuffle)')
            elif 'integrated_scores' in results[standard_name]:
                vals.append(results[standard_name]['integrated_scores'][2])
                labels.append('Standard\n(PGD)')

        # Best robust training
        best_flip, best_scores = find_best_robust(model_name, activation, 2)
        if best_scores is not None:
            vals.append(best_scores)
            labels.append(f'Robust\n(f={best_flip})')

        if vals:
            ax.boxplot(vals, widths=0.6)
            ax.set_ylabel('Saliency SNR', fontsize=10)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, fontsize=9)
            model_label = MODEL_LABELS.get(model_name, model_name)
            ax.set_title(f"{model_label}, {activation}", fontsize=11)

        subplot_idx += 1

outfile = os.path.join(results_path, 'task3_standard_vs_best_robust_snr.pdf')
fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
plt.close()

# Detailed comparison showing effect of flip fractions on interpretability

fig = plt.figure(figsize=(16, 12))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# Create comparison plots for each flip fraction vs standard
subplot_idx = 1
for model_name in model_names:
    for activation in activations:
        ax = plt.subplot(2, 4, subplot_idx)
        
        # Get standard training result
        standard_name = f"{model_name}_{activation}_standard"
        
        means = []
        stds = []
        labels = ['Standard\n(shuffle)']

        # Use shuffle baseline for fair comparison
        if standard_name in results:
            if 'integrated_scores_shuffle' in results[standard_name]:
                std_scores = results[standard_name]['integrated_scores_shuffle'][0]
                means.append(np.mean(std_scores))
                stds.append(np.std(std_scores))
            elif 'integrated_scores' in results[standard_name]:
                std_scores = results[standard_name]['integrated_scores'][0]
                means.append(np.mean(std_scores))
                stds.append(np.std(std_scores))
                labels[0] = 'Standard\n(PGD)'
            else:
                means.append(0)
                stds.append(0)
        else:
            means.append(0)
            stds.append(0)
        
        # Get robust training results for different flip fractions
        for flip_fraction in flip_fractions:
            robust_name = f"{model_name}_{activation}_robust_{flip_fraction}"
            labels.append(f'Robust-{flip_fraction}')
            
            if robust_name in results and 'integrated_scores' in results[robust_name]:
                rob_scores = results[robust_name]['integrated_scores'][0]
                means.append(np.mean(rob_scores))
                stds.append(np.std(rob_scores))
            else:
                means.append(0)
                stds.append(0)
        
        # Create bar plot with error bars
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        
        # Color bars differently
        bars[0].set_color('lightblue')  # Standard
        for i in range(1, len(bars)):
            bars[i].set_color('lightcoral')  # Robust variants
        
        ax.set_xlabel('Training Mode', fontsize=10)
        ax.set_ylabel('AUROC', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, fontsize=9)
        ax.tick_params(axis='y', labelsize=9)

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            if mean > 0:
                ax.text(i, mean + std + 0.01, f'{mean:.3f}',
                       ha='center', va='bottom', fontsize=8)

        subplot_idx += 1

outfile = os.path.join(results_path, 'task3_flip_fraction_detailed_comparison.pdf')
fig.savefig(outfile, format='pdf', dpi=200, bbox_inches='tight')
plt.close()

# Summary comparison of robust training with different flip fractions vs standard training

print("\nSummary of Integrated Gradients Performance with Different Flip Fractions:")
print("=" * 80)
for model_name in model_names:
    for activation in activations:
        print(f"\n{model_name}_{activation}:")
        print("-" * 50)
        
        # Standard training - use shuffle baseline as the reference for improvement calculation
        standard_name = f"{model_name}_{activation}_standard"
        std_auroc = None
        std_aupr = None
        std_snr = None

        if standard_name in results:
            # Report PGD baseline results if available
            if 'integrated_scores' in results[standard_name]:
                roc_scores = results[standard_name]['integrated_scores'][0]
                pr_scores = results[standard_name]['integrated_scores'][1]
                snr_scores = results[standard_name]['integrated_scores'][2]
                print(f"  STANDARD (PGD baseline):")
                print(f"    AUROC: {np.mean(roc_scores):.4f} ± {np.std(roc_scores):.4f}")
                print(f"    AUPR:  {np.mean(pr_scores):.4f} ± {np.std(pr_scores):.4f}")
                print(f"    SNR:   {np.mean(snr_scores):.4f} ± {np.std(snr_scores):.4f}")

            # Report and USE shuffle baseline as reference (conventional approach)
            if 'integrated_scores_shuffle' in results[standard_name]:
                roc_scores_shuffle = results[standard_name]['integrated_scores_shuffle'][0]
                pr_scores_shuffle = results[standard_name]['integrated_scores_shuffle'][1]
                snr_scores_shuffle = results[standard_name]['integrated_scores_shuffle'][2]
                print(f"  STANDARD (Shuffle baseline) [REFERENCE]:")
                print(f"    AUROC: {np.mean(roc_scores_shuffle):.4f} ± {np.std(roc_scores_shuffle):.4f}")
                print(f"    AUPR:  {np.mean(pr_scores_shuffle):.4f} ± {np.std(pr_scores_shuffle):.4f}")
                print(f"    SNR:   {np.mean(snr_scores_shuffle):.4f} ± {np.std(snr_scores_shuffle):.4f}")
                # Use shuffle baseline as reference for improvement calculation
                std_auroc = np.mean(roc_scores_shuffle)
                std_aupr = np.mean(pr_scores_shuffle)
                std_snr = np.mean(snr_scores_shuffle)
            elif 'integrated_scores' in results[standard_name]:
                # Fallback to PGD if shuffle not available
                print(f"  (Note: Shuffle baseline not available, using PGD as reference)")
                std_auroc = np.mean(roc_scores)
                std_aupr = np.mean(pr_scores)
                std_snr = np.mean(snr_scores)
        else:
            print(f"  STANDARD: No data available")
        
        # Robust training with different flip fractions
        for flip_fraction in flip_fractions:
            robust_name = f"{model_name}_{activation}_robust_{flip_fraction}"
            if robust_name in results and 'integrated_scores' in results[robust_name]:
                roc_scores = results[robust_name]['integrated_scores'][0]
                pr_scores = results[robust_name]['integrated_scores'][1]
                snr_scores = results[robust_name]['integrated_scores'][2]
                print(f"  ROBUST (flip={flip_fraction}):")
                print(f"    AUROC: {np.mean(roc_scores):.4f} ± {np.std(roc_scores):.4f}")
                print(f"    AUPR:  {np.mean(pr_scores):.4f} ± {np.std(pr_scores):.4f}")
                print(f"    SNR:   {np.mean(snr_scores):.4f} ± {np.std(snr_scores):.4f}")
                
                # Calculate improvement vs standard (shuffle baseline) if available
                if std_auroc is not None and std_aupr is not None and std_snr is not None:
                    rob_auroc = np.mean(roc_scores)
                    rob_aupr = np.mean(pr_scores)
                    rob_snr = np.mean(snr_scores)
                    improvement_auroc = rob_auroc - std_auroc
                    improvement_aupr = rob_aupr - std_aupr
                    improvement_snr = rob_snr - std_snr

                    print(f"    IMPROVEMENT vs Standard (shuffle baseline):")
                    print(f"      ΔAUROC: {improvement_auroc:+.4f}")
                    print(f"      ΔAUPR:  {improvement_aupr:+.4f}")
                    print(f"      ΔSNR:   {improvement_snr:+.4f}")
            else:
                print(f"  ROBUST (flip={flip_fraction}): No data available")
        print()

# Find the best flip fraction for each model-activation combination
print("\nBest Flip Fraction for Each Configuration (based on AUROC):")
print("=" * 60)
for model_name in model_names:
    for activation in activations:
        best_flip = None
        best_auroc = -1
        
        for flip_fraction in flip_fractions:
            robust_name = f"{model_name}_{activation}_robust_{flip_fraction}"
            if robust_name in results and 'integrated_scores' in results[robust_name]:
                auroc = np.mean(results[robust_name]['integrated_scores'][0])
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_flip = flip_fraction
        
        if best_flip is not None:
            print(f"{model_name}_{activation}: Best flip fraction = {best_flip} (AUROC = {best_auroc:.4f})")
        else:
            print(f"{model_name}_{activation}: No robust training data available")

# Find the best flip fraction based on SNR
print("\nBest Flip Fraction for Each Configuration (based on SNR):")
print("=" * 60)
for model_name in model_names:
    for activation in activations:
        best_flip = None
        best_snr = -1
        
        for flip_fraction in flip_fractions:
            robust_name = f"{model_name}_{activation}_robust_{flip_fraction}"
            if robust_name in results and 'integrated_scores' in results[robust_name]:
                snr = np.mean(results[robust_name]['integrated_scores'][2])
                if snr > best_snr:
                    best_snr = snr
                    best_flip = flip_fraction
        
        if best_flip is not None:
            print(f"{model_name}_{activation}: Best flip fraction = {best_flip} (SNR = {best_snr:.4f})")
        else:
            print(f"{model_name}_{activation}: No robust training data available")

# Load model performance results
perf_file_path = os.path.join(results_path, 'task3_performance_results.pickle')
with open(perf_file_path, 'rb') as f:
    perf_results = cPickle.load(f)

# Plot Interpretability AUC vs. Model Prediction AUC
print("\nGenerating Interpretability vs. Model Prediction AUC plot...")
fig = plt.figure(figsize=(12, 8))
ax = plt.subplot(1, 1, 1)

for model_name in model_names:
    for activation in activations:
        
        # Standard model (use shuffle baseline for fair comparison)
        standard_name = f"{model_name}_{activation}_standard"
        if standard_name in results and standard_name in perf_results:
            if 'integrated_scores_shuffle' in results[standard_name]:
                interp_auroc = np.mean(results[standard_name]['integrated_scores_shuffle'][0])
            elif 'integrated_scores' in results[standard_name]:
                interp_auroc = np.mean(results[standard_name]['integrated_scores'][0])
            else:
                continue
            model_auroc = np.mean(perf_results[standard_name][0])
            ax.scatter(model_auroc, interp_auroc, c='blue', marker='o', label=f'{standard_name} (Standard)')
        
        for flip_fraction in flip_fractions:
            robust_name = f"{model_name}_{activation}_robust_{flip_fraction}"
            if robust_name in results and 'integrated_scores' in results[robust_name] and robust_name in perf_results:
                interp_aurocs = results[robust_name]['integrated_scores'][0]
                interp_auroc = np.mean(results[robust_name]['integrated_scores'][0])
                model_auroc = np.mean(perf_results[robust_name][0])
                ax.scatter(model_auroc, interp_auroc, c='red', marker='x', label=f'{robust_name} (Robust)')
outfile = os.path.join(results_path, 'task3_interpretability_vs_prediction_auc_faceted.pdf')
plt.xlabel('Model Prediction AUROC', fontsize=12)
plt.ylabel('Interpretability AUROC', fontsize=12)
plt.title('Interpretability vs. Model Prediction Performance', fontsize=14)
plt.grid(True)
#plt.legend()
outfile = os.path.join(results_path, 'task3_interpretability_vs_prediction_auc.pdf')

# =============================================================================
# Generate clean summary table: Standard vs Best Robust only
# =============================================================================
print("\nGenerating clean summary table...")

summary_rows = []
for model_name in model_names:
    for activation in activations:
        config = f"{model_name}_{activation}"

        # Get standard results (using shuffle baseline)
        standard_name = f"{config}_standard"
        std_auroc, std_aupr, std_snr = None, None, None
        if standard_name in results:
            if 'integrated_scores_shuffle' in results[standard_name]:
                std_auroc = results[standard_name]['integrated_scores_shuffle'][0]
                std_aupr = results[standard_name]['integrated_scores_shuffle'][1]
                std_snr = results[standard_name]['integrated_scores_shuffle'][2]
            elif 'integrated_scores' in results[standard_name]:
                std_auroc = results[standard_name]['integrated_scores'][0]
                std_aupr = results[standard_name]['integrated_scores'][1]
                std_snr = results[standard_name]['integrated_scores'][2]

        # Find best robust for each metric
        best_robust = {'auroc': (None, None, -1), 'aupr': (None, None, -1), 'snr': (None, None, -1)}
        for flip_fraction in flip_fractions:
            robust_name = f"{config}_robust_{flip_fraction}"
            if robust_name in results and 'integrated_scores' in results[robust_name]:
                scores = results[robust_name]['integrated_scores']
                if np.mean(scores[0]) > best_robust['auroc'][2]:
                    best_robust['auroc'] = (flip_fraction, scores[0], np.mean(scores[0]))
                if np.mean(scores[1]) > best_robust['aupr'][2]:
                    best_robust['aupr'] = (flip_fraction, scores[1], np.mean(scores[1]))
                if np.mean(scores[2]) > best_robust['snr'][2]:
                    best_robust['snr'] = (flip_fraction, scores[2], np.mean(scores[2]))

        # Use AUROC-best robust for all metrics (consistent model selection)
        best_flip = best_robust['auroc'][0]
        if best_flip is not None:
            robust_name = f"{config}_robust_{best_flip}"
            rob_auroc = results[robust_name]['integrated_scores'][0]
            rob_aupr = results[robust_name]['integrated_scores'][1]
            rob_snr = results[robust_name]['integrated_scores'][2]
        else:
            rob_auroc, rob_aupr, rob_snr = None, None, None

        summary_rows.append({
            'model': model_name,
            'activation': activation,
            'std_auroc': std_auroc,
            'std_aupr': std_aupr,
            'std_snr': std_snr,
            'best_flip': best_flip,
            'rob_auroc': rob_auroc,
            'rob_aupr': rob_aupr,
            'rob_snr': rob_snr,
        })

# Save clean summary table
summary_file = os.path.join(results_path, 'summary_standard_vs_best_robust.tsv')
with open(summary_file, 'w') as f:
    f.write('model\tactivation\t')
    f.write('std_AUROC\tstd_AUPR\tstd_SNR\t')
    f.write('best_flip\trob_AUROC\trob_AUPR\trob_SNR\n')

    for row in summary_rows:
        f.write(f"{row['model']}\t{row['activation']}\t")

        if row['std_auroc'] is not None:
            f.write(f"{np.mean(row['std_auroc']):.4f}±{np.std(row['std_auroc']):.4f}\t")
            f.write(f"{np.mean(row['std_aupr']):.4f}±{np.std(row['std_aupr']):.4f}\t")
            f.write(f"{np.mean(row['std_snr']):.4f}±{np.std(row['std_snr']):.4f}\t")
        else:
            f.write("N/A\tN/A\tN/A\t")

        if row['rob_auroc'] is not None:
            f.write(f"{row['best_flip']}\t")
            f.write(f"{np.mean(row['rob_auroc']):.4f}±{np.std(row['rob_auroc']):.4f}\t")
            f.write(f"{np.mean(row['rob_aupr']):.4f}±{np.std(row['rob_aupr']):.4f}\t")
            f.write(f"{np.mean(row['rob_snr']):.4f}±{np.std(row['rob_snr']):.4f}\n")
        else:
            f.write("N/A\tN/A\tN/A\tN/A\n")

print(f"Summary table saved to: {summary_file}")

# Also print to console in readable format
print("\n" + "="*100)
print("SUMMARY: Standard (shuffle baseline) vs Best Robust")
print("="*100)
print(f"{'Model':<15} {'Act':<12} {'Std AUROC':<16} {'Std AUPR':<16} {'Std SNR':<16} {'Best f':<8} {'Rob AUROC':<16}")
print("-"*100)
for row in summary_rows:
    std_auroc_str = f"{np.mean(row['std_auroc']):.4f}±{np.std(row['std_auroc']):.4f}" if row['std_auroc'] is not None else "N/A"
    std_aupr_str = f"{np.mean(row['std_aupr']):.4f}±{np.std(row['std_aupr']):.4f}" if row['std_aupr'] is not None else "N/A"
    std_snr_str = f"{np.mean(row['std_snr']):.4f}±{np.std(row['std_snr']):.4f}" if row['std_snr'] is not None else "N/A"
    rob_auroc_str = f"{np.mean(row['rob_auroc']):.4f}±{np.std(row['rob_auroc']):.4f}" if row['rob_auroc'] is not None else "N/A"
    flip_str = f"{row['best_flip']}" if row['best_flip'] is not None else "N/A"
    print(f"{row['model']:<15} {row['activation']:<12} {std_auroc_str:<16} {std_aupr_str:<16} {std_snr_str:<16} {flip_str:<8} {rob_auroc_str:<16}")

print("\nAll plots and results saved successfully!")