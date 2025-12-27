#!/usr/bin/env python
"""
Generate comparison plots of sequence logos for different attribution methods

Figures generated from this script include:
- Attribution logo comparisons between standard and robust training
- Architecture comparison plots across different models
"""

import os
import numpy as np
from six.moves import cPickle
import matplotlib.pyplot as plt
import logomaker
import pandas as pd
from tfomics import utils
import helper

# Configuration
num_trials = 10
model_names = ['cnn-local', 'cnn-dist']
activations = ['relu', 'exponential']
training_modes = ['robust', 'standard']
flip_fractions = [0.01, 0.05, 0.1, 0.15, 0.2]

results_path = os.path.join('koo/results', 'task3_robust_3models')
params_path = os.path.join(results_path, 'model_params')
save_path = os.path.join(results_path, 'scores')
plot_path = utils.make_directory(results_path, 'attr_logo_plots')

# Load the performance results to determine best flip fractions
try:
    import pickle
    perf_file = os.path.join(results_path, 'task3_robust_attr_results.pickle')
    with open(perf_file, 'rb') as f:
        perf_results = pickle.load(f)
    
    # Find best flip fraction for each model-activation combination
    best_flip_fractions = {}
    for model_name in model_names:
        for activation in activations:
            best_flip = None
            best_auroc = -1
            
            for flip_fraction in flip_fractions:
                robust_name = f"{model_name}_{activation}_robust_{flip_fraction}"
                if robust_name in perf_results and 'integrated_scores' in perf_results[robust_name]:
                    auroc = np.mean(perf_results[robust_name]['integrated_scores'][0])
                    if auroc > best_auroc:
                        best_auroc = auroc
                        best_flip = flip_fraction
            
            if best_flip is not None:
                best_flip_fractions[f"{model_name}_{activation}"] = best_flip
                print(f"Best flip fraction for {model_name}_{activation}: {best_flip} (AUROC={best_auroc:.4f})")
            else:
                # Default to middle value if no results found
                best_flip_fractions[f"{model_name}_{activation}"] = 0.05
                print(f"No results found for {model_name}_{activation}, defaulting to flip_fraction=0.05")
                
except FileNotFoundError:
    print("Performance results not found. Using default flip_fraction=0.05 for all robust models.")
    best_flip_fractions = {f"{m}_{a}": 0.05 for m in model_names for a in activations}

# Load data
data_path = 'koo/data/synthetic_code_dataset.h5'
data = helper.load_data(data_path)
x_train, y_train, x_valid, y_valid, x_test, y_test = data

# Load ground truth values
test_model = helper.load_synthetic_models(data_path, dataset='test')
true_index = np.where(y_test[:,0] == 1)[0]
X = x_test[true_index][:500]
X_model = test_model[true_index][:500]

# Pre-compute labels once for all samples
gt_info = np.log2(4) + np.sum(X_model * np.log2(X_model + 1e-10), axis=1)
gt_labels = (gt_info > 0.01).astype(np.float32)

# Pre-compute GT masks for SNR calculation
gt_masks = gt_info > 0.01  # (N, L)

def calculate_single_snr(score, mask):
    """Calculate saliency SNR for a single sequence.

    SNR = sum(score^2 at GT positions) / sum(score^2 at all positions)
    """
    score_summed = np.sum(score, axis=1)  # (L,) - sum across nucleotides
    score_sq = score_summed ** 2
    sum_sq_inside = np.sum(score_sq * mask)
    sum_sq_total = np.sum(score_sq)
    return sum_sq_inside / (sum_sq_total + 1e-9)

# Load attribution scores
score_names = ['integrated_scores']

all_scores = {}
for model_name in model_names:
    for activation in activations:
        # Load standard training results with PGD baseline
        standard_name = f"{model_name}_{activation}_standard"
        print(standard_name)
        
        file_path = os.path.join(save_path, standard_name+'.pickle')
        try:
            with open(file_path, 'rb') as f:            
                # Only load the single item which should be integrated gradients scores
                integrated_scores = cPickle.load(f)

            # Only use integrated gradients (the single method available)
            scores = []
            scores.append(integrated_scores[0] * X)  # Use first trial
            all_scores[standard_name] = np.array(scores)
        except FileNotFoundError:
            print(f"  File not found: {file_path}")
            continue
        
        # Load standard training results with shuffle baseline
        shuffle_name = f"{model_name}_{activation}_standard_shuffle"
        print(shuffle_name)
        
        file_path = os.path.join(save_path, standard_name+'_shuffle.pickle')
        try:
            with open(file_path, 'rb') as f:            
                # Only load the single item which should be integrated gradients scores
                integrated_scores_shuffle = cPickle.load(f)

            # Only use integrated gradients (the single method available)
            scores = []
            scores.append(integrated_scores_shuffle[0] * X)  # Use first trial
            all_scores[shuffle_name] = np.array(scores)
        except FileNotFoundError:
            print(f"  Shuffle file not found: {file_path}")
        
        # Load robust training results - only the best flip fraction
        best_flip = best_flip_fractions.get(f"{model_name}_{activation}", 0.05)
        robust_name = f"{model_name}_{activation}_robust_{best_flip}"
        print(robust_name)

        file_path = os.path.join(save_path, robust_name+'.pickle')
        try:
            with open(file_path, 'rb') as f:
                # Only load the single item which should be integrated gradients scores
                integrated_scores = cPickle.load(f)

            # Only use integrated gradients (the single method available)
            scores = []
            scores.append(integrated_scores[0] * X)  # Use first trial
            all_scores[robust_name] = np.array(scores)

            # Also store with a simplified name for easier access
            all_scores[f"{model_name}_{activation}_robust"] = all_scores[robust_name]
        except FileNotFoundError:
            print(f"  File not found: {file_path}")
            continue

        # Load robust training results with shuffle baseline
        robust_shuffle_name = f"{model_name}_{activation}_robust_{best_flip}_shuffle"
        print(robust_shuffle_name)

        file_path = os.path.join(save_path, robust_name+'_shuffle.pickle')
        try:
            with open(file_path, 'rb') as f:
                integrated_scores_shuffle = cPickle.load(f)

            scores = []
            scores.append(integrated_scores_shuffle[0] * X)  # Use first trial
            all_scores[robust_shuffle_name] = np.array(scores)

            # Also store with a simplified name for easier access
            all_scores[f"{model_name}_{activation}_robust_shuffle"] = all_scores[robust_shuffle_name]
        except FileNotFoundError:
            print(f"  Robust shuffle file not found: {file_path}")

# Load attribution results  (generated from task3_plot_attr_score_comparisons.py)
# Note: This may not be needed since we're loading scores directly from individual pickle files
try:
    file_path = os.path.join(results_path, 'task3_robust_attr_results.pickle')
    with open(file_path, 'rb') as f:
        results = cPickle.load(f)
    print("Loaded task3_robust_attr_results.pickle successfully")
except FileNotFoundError:
    print("task3_robust_attr_results.pickle not found - will use scores from individual files instead")
    results = None

# Compare robust vs standard training modes for selected examples

# Choose which configuration to use for plotting
plot_model = 'cnn-dist'
plot_activation = 'exponential'  

# Get sequence dimensions
_, L, A = X.shape  # L: sequence length, A: alphabet size

# Use the standard model for sorting to find interesting examples
standard_model_name = f'{plot_model}_{plot_activation}_standard'
if standard_model_name in all_scores:
    sort_index = np.argsort(all_scores[standard_model_name][0].mean(axis=(1,2)))[::-1]
else:
    # Fallback to any available model if standard not found
    available_models = list(all_scores.keys())
    if available_models:
        sort_index = np.argsort(all_scores[available_models[0]][0].mean(axis=(1,2)))[::-1]
    else:
        print("No models available for plotting")
        sort_index = range(5)

# Generate comparison plots
num_plots = 10
best_flip = best_flip_fractions.get(f"{plot_model}_{plot_activation}", 0.05)

for plot_idx, index in enumerate(sort_index[:num_plots]):
    print(f"Generating plot {plot_idx + 1}/{num_plots} for sequence index {index}")
    
    fig = plt.figure(figsize=(12, 2.8))
    
    # Collect scores for all variants
    scores = []
    snr_scores = []
    mode_labels = []

    # Standard model with shuffle baseline only
    shuffle_name = f'{plot_model}_{plot_activation}_standard_shuffle'
    if shuffle_name in all_scores:
        score = all_scores[shuffle_name][0, index, :, :]
        scores.append(score)
        mode_labels.append('Std')
        snr_scores.append(calculate_single_snr(score, gt_masks[index]))

    # Robust model with best flip fraction
    robust_name = f'{plot_model}_{plot_activation}_robust'
    if robust_name in all_scores:
        score = all_scores[robust_name][0, index, :, :]
        scores.append(score)
        mode_labels.append('Rb')
        snr_scores.append(calculate_single_snr(score, gt_masks[index]))
    
    # Plot attribution logos for each variant
    for k, (score, snr_score, mode_label) in enumerate(zip(scores, snr_scores, mode_labels)):
        # Vectorized DataFrame creation
        counts_df = pd.DataFrame(score, columns=list('ACGT'), index=list(range(L)))

        ax = plt.subplot(len(scores) + 1, 1, k + 1)
        logomaker.Logo(counts_df, ax=ax)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        plt.xticks([])
        plt.yticks([])
        plt.ylabel(mode_label, fontsize=10)

        # Add performance score on the right
        ax2 = ax.twinx()
        plt.ylabel(f'SNR={snr_score:.3f}', fontsize=9)
        plt.yticks([])

    # Add ground truth logo
    w = X_model[index].T
    I = np.log2(4) + np.sum(w * np.log2(w + 1e-7), axis=1, keepdims=True)
    logo = I * w
    # Vectorized DataFrame creation
    counts_df = pd.DataFrame(logo, columns=list('ACGT'), index=list(range(L)))

    ax = plt.subplot(len(scores) + 1, 1, len(scores) + 1)
    logomaker.Logo(counts_df, ax=ax)
    plt.ylabel('GT', fontsize=10)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    # Save as both PDF and SVG
    outfile_base = os.path.join(plot_path, f'task3_logo_example_{plot_idx + 1}_seq_{index}')
    fig.savefig(outfile_base + '.pdf', format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(outfile_base + '.svg', format='svg', bbox_inches='tight')
    plt.close()
    
print(f"Generated {num_plots} attribution logo comparison plots.")

# =============================================================================
# APPENDIX: Logo plots with full 2x2 ablation (Standard/Robust × Shuffle/PGD)
# =============================================================================
print("\nGenerating appendix logo plots (full ablation)...")

appendix_plot_path = utils.make_directory(plot_path, 'appendix')

for plot_idx, index in enumerate(sort_index[:num_plots]):
    print(f"Generating appendix plot {plot_idx + 1}/{num_plots} for sequence index {index}")

    fig = plt.figure(figsize=(12, 4.5))

    # Collect scores for all variants (full 2x2 ablation)
    scores = []
    snr_scores = []
    mode_labels = []

    # Standard model with shuffle baseline
    shuffle_name = f'{plot_model}_{plot_activation}_standard_shuffle'
    if shuffle_name in all_scores:
        score = all_scores[shuffle_name][0, index, :, :]
        scores.append(score)
        mode_labels.append('Std-S')
        snr_scores.append(calculate_single_snr(score, gt_masks[index]))

    # Standard model with PGD baseline
    standard_name = f'{plot_model}_{plot_activation}_standard'
    if standard_name in all_scores:
        score = all_scores[standard_name][0, index, :, :]
        scores.append(score)
        mode_labels.append('Std-P')
        snr_scores.append(calculate_single_snr(score, gt_masks[index]))

    # Robust model with shuffle baseline
    robust_shuffle_name = f'{plot_model}_{plot_activation}_robust_shuffle'
    if robust_shuffle_name in all_scores:
        score = all_scores[robust_shuffle_name][0, index, :, :]
        scores.append(score)
        mode_labels.append('Rb-S')
        snr_scores.append(calculate_single_snr(score, gt_masks[index]))

    # Robust model with PGD baseline
    robust_name = f'{plot_model}_{plot_activation}_robust'
    if robust_name in all_scores:
        score = all_scores[robust_name][0, index, :, :]
        scores.append(score)
        mode_labels.append('Rb-P')
        snr_scores.append(calculate_single_snr(score, gt_masks[index]))

    # Plot attribution logos for each variant
    for k, (score, snr_score, mode_label) in enumerate(zip(scores, snr_scores, mode_labels)):
        counts_df = pd.DataFrame(score, columns=list('ACGT'), index=list(range(L)))
        ax = plt.subplot(len(scores) + 1, 1, k + 1)
        logomaker.Logo(counts_df, ax=ax)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        plt.xticks([])
        plt.yticks([])
        plt.ylabel(mode_label, fontsize=10)
        ax2 = ax.twinx()
        plt.ylabel(f'SNR={snr_score:.3f}', fontsize=9)
        plt.yticks([])

    # Add ground truth logo
    w = X_model[index].T
    I = np.log2(4) + np.sum(w * np.log2(w + 1e-7), axis=1, keepdims=True)
    logo = I * w
    counts_df = pd.DataFrame(logo, columns=list('ACGT'), index=list(range(L)))
    ax = plt.subplot(len(scores) + 1, 1, len(scores) + 1)
    logomaker.Logo(counts_df, ax=ax)
    plt.ylabel('GT', fontsize=10)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    outfile_base = os.path.join(appendix_plot_path, f'task3_logo_appendix_{plot_idx + 1}_seq_{index}')
    fig.savefig(outfile_base + '.pdf', format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(outfile_base + '.svg', format='svg', bbox_inches='tight')
    plt.close()

print(f"Generated {num_plots} appendix logo plots with full 2x2 ablation.")

print("\nAll logo comparison plots generated successfully!")