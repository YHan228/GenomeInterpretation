#!/usr/bin/env python
"""
GC% Baseline Classifier for Koo Task 3

This script trains a simple logistic regression classifier using only the GC%
of each sequence as the predictor. This serves as a baseline to demonstrate
that CNN models are learning more than just nucleotide composition.
"""

import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import helper

# Load data
data_path = 'koo/data/synthetic_code_dataset.h5'
data = helper.load_data(data_path)
x_train, y_train, x_valid, y_valid, x_test, y_test = data

print("Dataset shapes:")
print(f"  Train: {x_train.shape}, labels: {y_train.shape}")
print(f"  Valid: {x_valid.shape}, labels: {y_valid.shape}")
print(f"  Test:  {x_test.shape}, labels: {y_test.shape}")

def calculate_gc_content(x):
    """
    Calculate GC% for one-hot encoded sequences.

    x: (N, L, 4) one-hot encoded sequences where columns are [A, C, G, T]
    Returns: (N,) array of GC fractions
    """
    # Sum G and C channels (indices 1 and 2)
    gc_count = np.sum(x[:, :, 1] + x[:, :, 2], axis=1)
    # Total nucleotides per sequence
    total = x.shape[1]
    return gc_count / total

# Calculate GC% for all sets
gc_train = calculate_gc_content(x_train)
gc_valid = calculate_gc_content(x_valid)
gc_test = calculate_gc_content(x_test)

# Binary labels (positive class)
y_train_binary = y_train[:, 0]
y_valid_binary = y_valid[:, 0]
y_test_binary = y_test[:, 0]

print(f"\nGC% statistics:")
print(f"  Train - Positive: {gc_train[y_train_binary == 1].mean():.4f} ± {gc_train[y_train_binary == 1].std():.4f}")
print(f"  Train - Negative: {gc_train[y_train_binary == 0].mean():.4f} ± {gc_train[y_train_binary == 0].std():.4f}")
print(f"  Test  - Positive: {gc_test[y_test_binary == 1].mean():.4f} ± {gc_test[y_test_binary == 1].std():.4f}")
print(f"  Test  - Negative: {gc_test[y_test_binary == 0].mean():.4f} ± {gc_test[y_test_binary == 0].std():.4f}")

# Train logistic regression
X_train = gc_train.reshape(-1, 1)
X_valid = gc_valid.reshape(-1, 1)
X_test = gc_test.reshape(-1, 1)

clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train_binary)

# Predictions
y_train_pred = clf.predict_proba(X_train)[:, 1]
y_valid_pred = clf.predict_proba(X_valid)[:, 1]
y_test_pred = clf.predict_proba(X_test)[:, 1]

# Calculate AUC
train_auc = roc_auc_score(y_train_binary, y_train_pred)
valid_auc = roc_auc_score(y_valid_binary, y_valid_pred)
test_auc = roc_auc_score(y_test_binary, y_test_pred)

print(f"\nLogistic Regression (GC% only) Results:")
print(f"  Train AUC: {train_auc:.4f}")
print(f"  Valid AUC: {valid_auc:.4f}")
print(f"  Test AUC:  {test_auc:.4f}")

# Model coefficients
print(f"\nModel coefficients:")
print(f"  Intercept: {clf.intercept_[0]:.4f}")
print(f"  GC% coefficient: {clf.coef_[0][0]:.4f}")

# Save results
results_path = 'koo/results/task3_robust_3models'
os.makedirs(results_path, exist_ok=True)

with open(os.path.join(results_path, 'gc_baseline_results.txt'), 'w') as f:
    f.write("GC% Baseline Classifier Results\n")
    f.write("=" * 40 + "\n\n")
    f.write("Dataset:\n")
    f.write(f"  Train: {x_train.shape[0]} sequences\n")
    f.write(f"  Valid: {x_valid.shape[0]} sequences\n")
    f.write(f"  Test:  {x_test.shape[0]} sequences\n\n")
    f.write("GC% Statistics:\n")
    f.write(f"  Train - Positive: {gc_train[y_train_binary == 1].mean():.4f} ± {gc_train[y_train_binary == 1].std():.4f}\n")
    f.write(f"  Train - Negative: {gc_train[y_train_binary == 0].mean():.4f} ± {gc_train[y_train_binary == 0].std():.4f}\n")
    f.write(f"  Test  - Positive: {gc_test[y_test_binary == 1].mean():.4f} ± {gc_test[y_test_binary == 1].std():.4f}\n")
    f.write(f"  Test  - Negative: {gc_test[y_test_binary == 0].mean():.4f} ± {gc_test[y_test_binary == 0].std():.4f}\n\n")
    f.write("AUC Results:\n")
    f.write(f"  Train AUC: {train_auc:.4f}\n")
    f.write(f"  Valid AUC: {valid_auc:.4f}\n")
    f.write(f"  Test AUC:  {test_auc:.4f}\n\n")
    f.write("Model:\n")
    f.write(f"  Intercept: {clf.intercept_[0]:.4f}\n")
    f.write(f"  GC% coefficient: {clf.coef_[0][0]:.4f}\n")

print(f"\nResults saved to {os.path.join(results_path, 'gc_baseline_results.txt')}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test_binary, y_test_pred)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'GC% only (AUC = {test_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curve: GC% Baseline Classifier', fontsize=12)
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_aspect('equal')
plt.tight_layout()

fig.savefig(os.path.join(results_path, 'gc_baseline_roc.pdf'), format='pdf', dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(results_path, 'gc_baseline_roc.svg'), format='svg', bbox_inches='tight')
plt.close()

print(f"ROC curve saved to {os.path.join(results_path, 'gc_baseline_roc.pdf/svg')}")
