#!/usr/bin/env python
"""
Simplified ground truth analysis:
1. Motif length distribution
2. Motif counts
3. Empirical count of motif-like but unmarked segments
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import helper

# Load data
data_path = 'koo/data/synthetic_code_dataset.h5'
data = helper.load_data(data_path)
x_train, y_train, x_valid, y_valid, x_test, y_test = data

# Load ground truth models
test_model = helper.load_synthetic_models(data_path, dataset='test')

# Get positive sequences only
true_index = np.where(y_test[:, 0] == 1)[0]
X = x_test[true_index]  # (N, L, A) one-hot
X_model = test_model[true_index]  # (N, A, L) PWM

N, L, A = X.shape
print(f"Positive sequences: {N}, Length: {L}")

# Compute ground truth masks
gt_info = np.log2(4) + np.sum(X_model * np.log2(X_model + 1e-10), axis=1)  # (N, L)
gt_mask = gt_info > 0.01  # (N, L)

output_dir = 'koo/results/ground_truth_analysis'
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# 1. Extract all motifs and their lengths
# =============================================================================
print("\n" + "="*60)
print("1. MOTIF EXTRACTION")
print("="*60)

def onehot_to_seq(onehot):
    """Convert one-hot (L, 4) to string sequence."""
    bases = 'ACGT'
    return ''.join(bases[np.argmax(onehot[i])] for i in range(len(onehot)))

# Extract all unique motifs from all positive sequences
all_motifs = set()
motif_lengths = []

for i in range(N):
    mask = gt_mask[i]
    seq_onehot = X[i]  # (L, A)

    # Find contiguous motif regions
    in_motif = False
    start = 0
    for j in range(L):
        if mask[j] and not in_motif:
            start = j
            in_motif = True
        elif not mask[j] and in_motif:
            # End of motif
            motif_seq = onehot_to_seq(seq_onehot[start:j])
            all_motifs.add(motif_seq)
            motif_lengths.append(j - start)
            in_motif = False
    # Handle motif at end
    if in_motif:
        motif_seq = onehot_to_seq(seq_onehot[start:L])
        all_motifs.add(motif_seq)
        motif_lengths.append(L - start)

motif_lengths = np.array(motif_lengths)
print(f"Total motif instances: {len(motif_lengths)}")
print(f"Unique motif sequences: {len(all_motifs)}")

# =============================================================================
# 2. Motif length distribution
# =============================================================================
print("\n" + "="*60)
print("2. MOTIF LENGTH DISTRIBUTION")
print("="*60)

unique_lengths, counts = np.unique(motif_lengths, return_counts=True)
for length, count in zip(unique_lengths, counts):
    print(f"  Length {length:2d}: {count:5d} instances ({100*count/len(motif_lengths):.1f}%)")

print(f"\nMean length: {np.mean(motif_lengths):.1f} bp")
print(f"Median length: {np.median(motif_lengths):.0f} bp")

# =============================================================================
# 3. Empirical search for motif-like but unmarked segments
# =============================================================================
print("\n" + "="*60)
print("3. UNMARKED MOTIF-LIKE SEGMENTS (EMPIRICAL)")
print("="*60)

# Group motifs by length for efficient lookup
motifs_by_length = {}
for motif in all_motifs:
    length = len(motif)
    if length not in motifs_by_length:
        motifs_by_length[length] = set()
    motifs_by_length[length].add(motif)

print(f"Motif lengths to search: {sorted(motifs_by_length.keys())}")

# Search through positive sequences for unmarked matches
unmarked_match_counts = []
total_unmarked_positions = 0

for i in range(N):
    mask = gt_mask[i]
    seq_onehot = X[i]
    full_seq = onehot_to_seq(seq_onehot)

    matches_this_seq = 0

    # For each motif length, check all windows
    for motif_len, motif_set in motifs_by_length.items():
        for j in range(L - motif_len + 1):
            # Check if this window is in an unmarked region
            # (none of the positions in the window are marked)
            if not np.any(mask[j:j+motif_len]):
                window_seq = full_seq[j:j+motif_len]
                if window_seq in motif_set:
                    matches_this_seq += 1

    unmarked_match_counts.append(matches_this_seq)
    total_unmarked_positions += np.sum(~mask)

unmarked_match_counts = np.array(unmarked_match_counts)

print(f"\nResults across {N} positive sequences:")
print(f"  Mean unmarked motif-matches per sequence: {np.mean(unmarked_match_counts):.2f}")
print(f"  Median: {np.median(unmarked_match_counts):.0f}")
print(f"  Max: {np.max(unmarked_match_counts)}")
print(f"  Sequences with ≥1 unmarked match: {np.sum(unmarked_match_counts > 0)} ({100*np.mean(unmarked_match_counts > 0):.1f}%)")

# =============================================================================
# Summary plot (placeholder - will be replaced by comprehensive figure below)
# =============================================================================

# =============================================================================
# Save summary
# =============================================================================
summary_file = os.path.join(output_dir, 'ground_truth_summary.txt')
with open(summary_file, 'w') as f:
    f.write("GROUND TRUTH SUMMARY\n")
    f.write("="*40 + "\n\n")
    f.write(f"Positive sequences: {N}\n")
    f.write(f"Sequence length: {L} bp\n\n")

    f.write("MOTIF COUNTS\n")
    f.write(f"  Total instances: {len(motif_lengths)}\n")
    f.write(f"  Unique sequences: {len(all_motifs)}\n\n")

    f.write("MOTIF LENGTHS\n")
    for length, count in zip(unique_lengths, counts):
        f.write(f"  {length} bp: {count}\n")
    f.write(f"  Mean: {np.mean(motif_lengths):.1f} bp\n\n")

    f.write("UNMARKED MOTIF-MATCHES\n")
    f.write(f"  Mean per sequence: {np.mean(unmarked_match_counts):.2f}\n")
    f.write(f"  Sequences with ≥1: {np.sum(unmarked_match_counts > 0)} ({100*np.mean(unmarked_match_counts > 0):.1f}%)\n")

print(f"\nSaved to: {output_dir}")

# =============================================================================
# 4. Background motifs in positive sequences
# =============================================================================
print("\n" + "="*60)
print("4. BACKGROUND MOTIFS IN POSITIVE SEQUENCES")
print("="*60)

# Extract motifs from NEGATIVE sequences (background pool)
neg_index = np.where(y_test[:, 0] == 0)[0]
X_neg = x_test[neg_index]
X_model_neg = test_model[neg_index]

N_neg = len(neg_index)
print(f"Negative sequences: {N_neg}")

# Compute GT masks for negative sequences
gt_info_neg = np.log2(4) + np.sum(X_model_neg * np.log2(X_model_neg + 1e-10), axis=1)
gt_mask_neg = gt_info_neg > 0.01

# Extract all motifs from negative sequences
background_motifs = set()
bg_motif_lengths = []

for i in range(N_neg):
    mask = gt_mask_neg[i]
    seq_onehot = X_neg[i]

    in_motif = False
    start = 0
    for j in range(L):
        if mask[j] and not in_motif:
            start = j
            in_motif = True
        elif not mask[j] and in_motif:
            motif_seq = onehot_to_seq(seq_onehot[start:j])
            background_motifs.add(motif_seq)
            bg_motif_lengths.append(j - start)
            in_motif = False
    if in_motif:
        motif_seq = onehot_to_seq(seq_onehot[start:L])
        background_motifs.add(motif_seq)
        bg_motif_lengths.append(L - start)

print(f"Total background motif instances: {len(bg_motif_lengths)}")
print(f"Unique background motif sequences: {len(background_motifs)}")

# Background-only = background - core
# (core motifs are in all_motifs from positive sequences)
background_only = background_motifs - all_motifs
print(f"Background-only motifs (excluding core): {len(background_only)}")
print(f"Overlap (in both core and background): {len(background_motifs & all_motifs)}")

# Group background-only motifs by length
bg_only_by_length = {}
for motif in background_only:
    length = len(motif)
    if length not in bg_only_by_length:
        bg_only_by_length[length] = set()
    bg_only_by_length[length].add(motif)

print(f"Background-only motif lengths: {sorted(bg_only_by_length.keys())}")

# Search for background-only motifs in POSITIVE sequences
print("\nSearching for background-only motifs in positive sequences...")
bg_match_counts = []

for i in range(N):
    seq_onehot = X[i]
    full_seq = onehot_to_seq(seq_onehot)

    matches_this_seq = 0

    for motif_len, motif_set in bg_only_by_length.items():
        for j in range(L - motif_len + 1):
            window_seq = full_seq[j:j+motif_len]
            if window_seq in motif_set:
                matches_this_seq += 1

    bg_match_counts.append(matches_this_seq)

bg_match_counts = np.array(bg_match_counts)

print(f"\nResults across {N} positive sequences:")
print(f"  Mean background-only matches per sequence: {np.mean(bg_match_counts):.2f}")
print(f"  Median: {np.median(bg_match_counts):.0f}")
print(f"  Max: {np.max(bg_match_counts)}")
print(f"  Sequences with ≥1 background-only match: {np.sum(bg_match_counts > 0)} ({100*np.mean(bg_match_counts > 0):.1f}%)")

# Update summary file
with open(summary_file, 'a') as f:
    f.write("\nBACKGROUND-ONLY MOTIFS IN POSITIVE SEQS\n")
    f.write(f"  Background motifs (from neg seqs): {len(background_motifs)}\n")
    f.write(f"  Background-only (excl. core): {len(background_only)}\n")
    f.write(f"  Mean matches per pos seq: {np.mean(bg_match_counts):.2f}\n")
    f.write(f"  Pos seqs with ≥1 match: {np.sum(bg_match_counts > 0)} ({100*np.mean(bg_match_counts > 0):.1f}%)\n")

print(f"\nUpdated: {summary_file}")

# =============================================================================
# 5. COMPREHENSIVE APPENDIX FIGURE AND TABLE
# =============================================================================
print("\n" + "="*60)
print("5. GENERATING APPENDIX FIGURE AND TABLE")
print("="*60)

# Count motifs per sequence for positive and negative
core_motifs_per_seq = []
for i in range(N):
    mask = gt_mask[i]
    # Count contiguous motif regions
    n_motifs = 0
    in_motif = False
    for j in range(L):
        if mask[j] and not in_motif:
            in_motif = True
        elif not mask[j] and in_motif:
            n_motifs += 1
            in_motif = False
    if in_motif:
        n_motifs += 1
    core_motifs_per_seq.append(n_motifs)
core_motifs_per_seq = np.array(core_motifs_per_seq)

bg_motifs_per_seq = []
for i in range(N_neg):
    mask = gt_mask_neg[i]
    n_motifs = 0
    in_motif = False
    for j in range(L):
        if mask[j] and not in_motif:
            in_motif = True
        elif not mask[j] and in_motif:
            n_motifs += 1
            in_motif = False
    if in_motif:
        n_motifs += 1
    bg_motifs_per_seq.append(n_motifs)
bg_motifs_per_seq = np.array(bg_motifs_per_seq)

# --- Comprehensive Figure (6 panels) ---
fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
plt.subplots_adjust(wspace=0.35, hspace=0.45, left=0.07, right=0.98, bottom=0.10, top=0.95)

# Panel A: Core motif length distribution (%)
ax = axes[0, 0]
main_lengths = [10, 11, 12]
main_counts = [counts[list(unique_lengths).index(l)] for l in main_lengths if l in unique_lengths]
main_pcts = [100 * c / len(motif_lengths) for c in main_counts]
bars = ax.bar(main_lengths, main_pcts, color='#4878A8', edgecolor='black', alpha=0.8, width=0.7)
ax.set_xlabel('Core motif length (bp)', fontsize=9)
ax.set_ylabel('Percentage (%)', fontsize=9)
ax.set_xticks(main_lengths)
ax.tick_params(labelsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, pct in zip(bars, main_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=7)

# Panel B: Background motif length distribution (%)
ax = axes[0, 1]
bg_unique_lengths, bg_len_counts = np.unique(bg_motif_lengths, return_counts=True)
# Group into bins for clarity
bg_pcts = 100 * bg_len_counts / len(bg_motif_lengths)
ax.bar(bg_unique_lengths, bg_pcts, color='#E85D4C', edgecolor='black', alpha=0.8, width=0.8)
ax.set_xlabel('Background motif length (bp)', fontsize=9)
ax.set_ylabel('Percentage (%)', fontsize=9)
ax.tick_params(labelsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel C: Core motifs per positive sequence (%)
ax = axes[0, 2]
bins = range(min(core_motifs_per_seq), max(core_motifs_per_seq)+2)
counts_hist, _ = np.histogram(core_motifs_per_seq, bins=bins)
pcts_hist = 100 * counts_hist / len(core_motifs_per_seq)
ax.bar(bins[:-1], pcts_hist, color='#4878A8', edgecolor='black', alpha=0.8, width=0.8, align='edge')
ax.axvline(np.mean(core_motifs_per_seq), color='red', linestyle='--', linewidth=1.5,
           label=f'Mean={np.mean(core_motifs_per_seq):.1f}')
ax.set_xlabel('Core motifs per pos. seq', fontsize=9)
ax.set_ylabel('Percentage (%)', fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel D: Background motifs per negative sequence (%)
ax = axes[1, 0]
bins = range(min(bg_motifs_per_seq), max(bg_motifs_per_seq)+2)
counts_hist, _ = np.histogram(bg_motifs_per_seq, bins=bins)
pcts_hist = 100 * counts_hist / len(bg_motifs_per_seq)
ax.bar(bins[:-1], pcts_hist, color='#E85D4C', edgecolor='black', alpha=0.8, width=0.8, align='edge')
ax.axvline(np.mean(bg_motifs_per_seq), color='darkred', linestyle='--', linewidth=1.5,
           label=f'Mean={np.mean(bg_motifs_per_seq):.1f}')
ax.set_xlabel('Background motifs per neg. seq', fontsize=9)
ax.set_ylabel('Percentage (%)', fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel E: Unmarked core-motif matches in positive seqs (%)
ax = axes[1, 1]
bins = range(0, max(unmarked_match_counts)+2)
counts_hist, _ = np.histogram(unmarked_match_counts, bins=bins)
pcts_hist = 100 * counts_hist / len(unmarked_match_counts)
ax.bar(bins[:-1], pcts_hist, color='#6AAF6A', edgecolor='black', alpha=0.8, width=0.8, align='edge')
ax.axvline(np.mean(unmarked_match_counts), color='darkgreen', linestyle='--', linewidth=1.5,
           label=f'Mean={np.mean(unmarked_match_counts):.2f}')
ax.set_xlabel('Unmarked core matches per pos. seq', fontsize=9)
ax.set_ylabel('Percentage (%)', fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel F: Background-only motif matches in positive seqs (%)
ax = axes[1, 2]
bins = range(0, max(bg_match_counts)+2)
counts_hist, _ = np.histogram(bg_match_counts, bins=bins)
pcts_hist = 100 * counts_hist / len(bg_match_counts)
ax.bar(bins[:-1], pcts_hist, color='#9467BD', edgecolor='black', alpha=0.8, width=0.8, align='edge')
ax.axvline(np.mean(bg_match_counts), color='purple', linestyle='--', linewidth=1.5,
           label=f'Mean={np.mean(bg_match_counts):.2f}')
ax.set_xlabel('Background-only matches per pos. seq', fontsize=9)
ax.set_ylabel('Percentage (%)', fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(fontsize=7, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig(os.path.join(output_dir, 'gt_analysis_appendix.pdf'),
            format='pdf', dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(output_dir, 'gt_analysis_appendix.svg'),
            format='svg', bbox_inches='tight')
plt.close()
print("Saved: gt_analysis_appendix.pdf/svg")

# --- Comprehensive Table (TSV) ---
table_file = os.path.join(output_dir, 'gt_analysis_table.tsv')
with open(table_file, 'w') as f:
    f.write("Category\tMetric\tValue\n")

    # Dataset
    f.write("Dataset\tPositive sequences\t{}\n".format(N))
    f.write("Dataset\tNegative sequences\t{}\n".format(N_neg))
    f.write("Dataset\tSequence length (bp)\t{}\n".format(L))

    # Core motifs
    f.write("Core motifs\tTotal instances\t{}\n".format(len(motif_lengths)))
    f.write("Core motifs\tUnique sequences\t{}\n".format(len(all_motifs)))
    f.write("Core motifs\tMean length (bp)\t{:.1f}\n".format(np.mean(motif_lengths)))
    f.write("Core motifs\tLength 10bp (%)\t{:.1f}\n".format(100*counts[list(unique_lengths).index(10)]/len(motif_lengths) if 10 in unique_lengths else 0))
    f.write("Core motifs\tLength 11bp (%)\t{:.1f}\n".format(100*counts[list(unique_lengths).index(11)]/len(motif_lengths) if 11 in unique_lengths else 0))
    f.write("Core motifs\tLength 12bp (%)\t{:.1f}\n".format(100*counts[list(unique_lengths).index(12)]/len(motif_lengths) if 12 in unique_lengths else 0))
    f.write("Core motifs\tMean motifs per pos. seq\t{:.1f}\n".format(np.mean(core_motifs_per_seq)))

    # Background motifs
    f.write("Background motifs\tTotal instances\t{}\n".format(len(bg_motif_lengths)))
    f.write("Background motifs\tUnique sequences\t{}\n".format(len(background_motifs)))
    f.write("Background motifs\tMean length (bp)\t{:.1f}\n".format(np.mean(bg_motif_lengths)))
    f.write("Background motifs\tMean motifs per neg. seq\t{:.1f}\n".format(np.mean(bg_motifs_per_seq)))
    f.write("Background motifs\tBackground-only (excl. core)\t{}\n".format(len(background_only)))
    f.write("Background motifs\tOverlap with core\t{}\n".format(len(background_motifs & all_motifs)))

    # Spurious matches
    f.write("Spurious matches\tUnmarked core matches (mean/seq)\t{:.2f}\n".format(np.mean(unmarked_match_counts)))
    f.write("Spurious matches\tPos seqs with ≥1 unmarked core\t{} ({:.1f}%)\n".format(
        np.sum(unmarked_match_counts > 0), 100*np.mean(unmarked_match_counts > 0)))
    f.write("Spurious matches\tBackground-only matches (mean/seq)\t{:.2f}\n".format(np.mean(bg_match_counts)))
    f.write("Spurious matches\tPos seqs with ≥1 background-only\t{} ({:.1f}%)\n".format(
        np.sum(bg_match_counts > 0), 100*np.mean(bg_match_counts > 0)))

print(f"Saved: {table_file}")

# --- Print table to console ---
print("\n" + "="*70)
print("APPENDIX TABLE: Ground Truth Analysis Summary")
print("="*70)
print(f"{'Category':<22} {'Metric':<35} {'Value':<15}")
print("-"*70)
print(f"{'Dataset':<22} {'Positive sequences':<35} {N:<15}")
print(f"{'':<22} {'Negative sequences':<35} {N_neg:<15}")
print(f"{'':<22} {'Sequence length (bp)':<35} {L:<15}")
print(f"{'Core motifs':<22} {'Total instances':<35} {len(motif_lengths):<15}")
print(f"{'':<22} {'Unique sequences':<35} {len(all_motifs):<15}")
print(f"{'':<22} {'Mean length (bp)':<35} {np.mean(motif_lengths):<15.1f}")
print(f"{'Background motifs':<22} {'Unique sequences':<35} {len(background_motifs):<15}")
print(f"{'':<22} {'Background-only (excl. core)':<35} {len(background_only):<15}")
print(f"{'Spurious matches':<22} {'Unmarked core (mean/seq)':<35} {np.mean(unmarked_match_counts):<15.2f}")
print(f"{'':<22} {'Background-only (mean/seq)':<35} {np.mean(bg_match_counts):<15.2f}")
print(f"{'':<22} {'Pos seqs with ≥1 bg-only':<35} {f'{100*np.mean(bg_match_counts > 0):.1f}%':<15}")
print("="*70)
