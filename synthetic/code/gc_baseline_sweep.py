#!/usr/bin/env python
"""
GC% Baseline Classifier Sweep for Synthetic Datasets

This script trains logistic regression classifiers using only GC% as predictor
across all cached synthetic datasets with varying GC_pos levels.
Plots AUC vs GC_pos to show how GC content difference affects baseline performance.
"""

import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
DATASET_CACHE_DIR = "dataset_cache"
OUTPUT_DIR = "synthetic/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_gc_content(x):
    """
    Calculate GC% for one-hot encoded sequences.
    x: (N, 4, L) one-hot encoded sequences where channels are [A, C, G, T]
    Returns: (N,) array of GC fractions
    """
    # Sum G and C channels (indices 1 and 2) across sequence length
    # x[:, 1, :] is C channel, x[:, 2, :] is G channel
    gc_count = np.sum(x[:, 1, :] + x[:, 2, :], axis=1)
    total = x.shape[2]  # sequence length is axis 2
    return gc_count / total

def evaluate_gc_classifier(X, y):
    """Train and evaluate GC-only logistic classifier."""
    gc = calculate_gc_content(X)

    # Stratified split to ensure both classes in train and test
    gc_train, gc_test, y_train, y_test = train_test_split(
        gc, y, test_size=0.2, random_state=42, stratify=y
    )

    # Reshape for sklearn
    X_train = gc_train.reshape(-1, 1)
    X_test = gc_test.reshape(-1, 1)

    # Train logistic regression
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)

    # GC stats
    gc_pos = gc[y == 1].mean()
    gc_neg = gc[y == 0].mean()

    return auc, gc_pos, gc_neg

# Scan all cached datasets
print("Scanning cached datasets...")
cache_files = [f for f in os.listdir(DATASET_CACHE_DIR) if f.endswith('.npz')]
print(f"Found {len(cache_files)} cached datasets")

# Parse and group by gc_pos and conservation
results = []

for cache_file in sorted(cache_files):
    # Parse filename: gc_0.500_cons_0.600.npz
    parts = cache_file.replace('.npz', '').split('_')
    gc_pos_param = float(parts[1])
    cons_param = float(parts[3])

    # Load data
    cache_path = os.path.join(DATASET_CACHE_DIR, cache_file)
    data = np.load(cache_path)
    X = data['X']
    y = data['y']

    # Evaluate
    auc, gc_pos_actual, gc_neg_actual = evaluate_gc_classifier(X, y)

    results.append({
        'gc_pos_param': gc_pos_param,
        'cons_param': cons_param,
        'auc': auc,
        'gc_pos_actual': gc_pos_actual,
        'gc_neg_actual': gc_neg_actual,
        'gc_diff': gc_pos_actual - gc_neg_actual
    })

    print(f"gc={gc_pos_param:.3f}, cons={cons_param:.2f}: AUC={auc:.4f}, "
          f"GC_pos={gc_pos_actual:.4f}, GC_neg={gc_neg_actual:.4f}, diff={gc_pos_actual-gc_neg_actual:.4f}")

# Convert to arrays for plotting
results_df = {k: np.array([r[k] for r in results]) for k in results[0].keys()}

# Get unique GC_pos values and average AUC across conservation levels
gc_pos_values = sorted(set(results_df['gc_pos_param']))
print(f"\nGC_pos levels: {gc_pos_values}")

# Average AUC for each GC_pos (across all conservation levels)
avg_auc = []
for gc_pos in gc_pos_values:
    mask = results_df['gc_pos_param'] == gc_pos
    avg_auc.append(results_df['auc'][mask].mean())

avg_auc = np.array(avg_auc)

# Simple plot: GC_pos vs AUC
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(gc_pos_values, avg_auc, 'o-', color='#2E86AB', markersize=6, linewidth=1.5)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)

ax.set_xlabel('GC content of positive sequences', fontsize=11)
ax.set_ylabel('Test AUC', fontsize=11)
ax.set_ylim([0.45, 1.05])
ax.set_xlim([0.48, 0.82])

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'gc_baseline_sweep.pdf'), format='pdf', dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'gc_baseline_sweep.svg'), format='svg', bbox_inches='tight')
plt.close()

print(f"\nPlot saved to {OUTPUT_DIR}/gc_baseline_sweep.pdf/svg")

# Save results table
results_table_path = os.path.join(OUTPUT_DIR, 'gc_baseline_sweep.tsv')
with open(results_table_path, 'w') as f:
    f.write("gc_pos_param\tcons_param\tauc\tgc_pos_actual\tgc_neg_actual\tgc_diff\n")
    for r in sorted(results, key=lambda x: (x['gc_pos_param'], x['cons_param'])):
        f.write(f"{r['gc_pos_param']:.3f}\t{r['cons_param']:.2f}\t{r['auc']:.4f}\t"
                f"{r['gc_pos_actual']:.4f}\t{r['gc_neg_actual']:.4f}\t{r['gc_diff']:.4f}\n")

print(f"Results table saved to {results_table_path}")

# Summary statistics
print("\n" + "="*60)
print("Summary Statistics")
print("="*60)
print(f"AUC range: {results_df['auc'].min():.4f} - {results_df['auc'].max():.4f}")
print(f"Mean AUC: {results_df['auc'].mean():.4f}")
print(f"GC diff range: {results_df['gc_diff'].min():.4f} - {results_df['gc_diff'].max():.4f}")
