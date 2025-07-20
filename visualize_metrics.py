import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

# --------------------------------------------------------------------------- #
# 1. Configuration & Constants
# --------------------------------------------------------------------------- #

SEQ_LEN = 1000
CAUSAL_LEN = 60
# Center the causal region for clarity
CAUSAL_START = (SEQ_LEN - CAUSAL_LEN) // 2
CAUSAL_END = CAUSAL_START + CAUSAL_LEN

# Create the true mask for the causal region
TRUE_MASK = np.zeros(SEQ_LEN, dtype=bool)
TRUE_MASK[CAUSAL_START:CAUSAL_END] = True

# Hyperparameters from the experiment for annotation
GC_HPARAMS = [0.5, 0.55, 0.6, 0.65]
CONS_HPARAMS = [0.6, 0.7, 0.8]


# --------------------------------------------------------------------------- #
# 2. Metric Calculation Functions
# (Adapted from toy_slurm.py for this standalone script)
# --------------------------------------------------------------------------- #

def calculate_w_iou(attributions, true_mask, window_len):
    """Calculates the windowed Intersection-over-Union (Overlap)."""
    # Find the window with the highest sum of attributions
    window_sums = np.convolve(np.abs(attributions), np.ones(window_len), mode='valid')
    best_window_start = np.argmax(window_sums)
    
    # Create a mask for the predicted window
    pred_mask = np.zeros_like(true_mask, dtype=bool)
    pred_mask[best_window_start:best_window_start + window_len] = True
    
    # Calculate IoU
    intersection = np.sum(pred_mask & true_mask)
    union = np.sum(pred_mask | true_mask)
    
    iou = intersection / union if union > 0 else 0
    return iou, pred_mask

def calculate_saliency_auc(attributions, true_mask):
    """Calculates the Saliency AUC."""
    inside_scores = attributions[true_mask]
    outside_scores = attributions[~true_mask]
    
    # For performance on this illustrative script, we can downsample
    if len(outside_scores) > 2000:
        outside_scores = np.random.choice(outside_scores, 2000, replace=False)
        
    saliency_auc = (inside_scores[:, None] > outside_scores[None, :]).mean()
    return saliency_auc

def calculate_saliency_snr(attributions, true_mask):
    """Calculates the Saliency Signal-to-Noise Ratio (SNR)."""
    inside_scores = attributions[true_mask]
    
    sum_sq_inside = np.sum(inside_scores**2)
    sum_sq_total = np.sum(attributions**2)
    
    saliency_snr = sum_sq_inside / (sum_sq_total + 1e-9)
    return saliency_snr


# --------------------------------------------------------------------------- #
# 3. Attribution Curve Generation
# --------------------------------------------------------------------------- #

def generate_peak(center, std, height, length=SEQ_LEN):
    """Generates a Gaussian-like peak."""
    x = np.arange(length)
    return height * np.exp(-((x - center)**2) / (2 * std**2))

def generate_noise(level, length=SEQ_LEN):
    """Generates uniform random noise."""
    return (np.random.rand(length) - 0.5) * 2 * level


# --------------------------------------------------------------------------- #
# 4. Plotting
# --------------------------------------------------------------------------- #

def plot_scenario(ax, attributions, title_prefix, metric_name, show_pred_window=False):
    """Helper function to plot a single scenario."""
    
    # Calculate metric
    if metric_name == 'Overlap':
        score, pred_mask = calculate_w_iou(attributions, TRUE_MASK, CAUSAL_LEN)
    elif metric_name == 'Saliency AUC':
        score = calculate_saliency_auc(attributions, TRUE_MASK)
    elif metric_name == 'Saliency SNR':
        score = calculate_saliency_snr(attributions, TRUE_MASK)

    # Plotting
    ax.plot(attributions, color='#003366', lw=1.5, label='Attribution Score')
    
    # Shade the true causal region
    ax.axvspan(CAUSAL_START, CAUSAL_END - 1, color='green', alpha=0.3, label='True Causal Region', lw=0)

    # Shade the predicted window for Overlap
    if show_pred_window:
        pred_indices = np.where(pred_mask)[0]
        if len(pred_indices) > 0:
            pred_start, pred_end = pred_indices[0], pred_indices[-1]
            rect = patches.Rectangle((pred_start, ax.get_ylim()[0]), pred_end - pred_start, ax.get_ylim()[1] - ax.get_ylim()[0],
                                     linewidth=1.5, edgecolor='red', facecolor='red', alpha=0.2, linestyle='--', label='Predicted Window')
            ax.add_patch(rect)

    ax.set_title(f"{title_prefix}\n{metric_name} = {score:.2f}", fontsize=12)
    ax.set_xlabel("Sequence Position")
    ax.set_xlim(0, SEQ_LEN)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel("Attribution")
    
    handles, labels = ax.get_legend_handles_labels()
    return handles, labels


def main():
    """Main function to generate and save the plots."""
    np.random.seed(42)  # Set seed for reproducibility
    fig, axes = plt.subplots(3, 3, figsize=(20, 16), constrained_layout=True)
    fig.suptitle("Visualizing Interpretability Metrics for Attribution Maps", fontsize=20, weight='bold')

    # --- Row 1: Windowed IoU (Overlap) ---
    # High: Peak is perfectly centered in the causal region.
    attr_Overlap_high = generate_peak(center=CAUSAL_START + CAUSAL_LEN / 2, std=15, height=1) + generate_noise(0.05)
    # Medium: Peak is shifted, causing partial overlap.
    attr_Overlap_mid = generate_peak(center=CAUSAL_START - 20, std=15, height=1) + generate_noise(0.05)
    # Low: Peak is far outside the causal region.
    attr_Overlap_low = generate_peak(center=150, std=15, height=1) + generate_noise(0.05)

    plot_scenario(axes[0, 0], attr_Overlap_high, 'High Score (Peak inside region)', 'Overlap', show_pred_window=True)
    plot_scenario(axes[0, 1], attr_Overlap_mid, 'Medium Score (Peak partially overlaps)', 'Overlap', show_pred_window=True)
    plot_scenario(axes[0, 2], attr_Overlap_low, 'Low Score (Peak outside region)', 'Overlap', show_pred_window=True)

    # --- Row 2: Saliency AUC ---
    # High: Scores are clearly higher inside the region, but with some noise/overlap.
    attr_auc_high = generate_peak(center=CAUSAL_START + CAUSAL_LEN/2, std=40, height=0.5) + generate_noise(0.3)
    # Medium: Uncorrelated random noise across the sequence, leading to an AUC ~0.5.
    attr_auc_mid = generate_noise(0.6)
    # Low: Scores are systematically lower inside the causal region than outside.
    attr_auc_low = generate_noise(0.25) + 0.25
    attr_auc_low += generate_peak(center=200, std=50, height=0.8)
    attr_auc_low[TRUE_MASK] *= 0.4 # Suppress signal inside

    plot_scenario(axes[1, 0], attr_auc_high, 'High Score (Good separation)', 'Saliency AUC')
    plot_scenario(axes[1, 1], attr_auc_mid, 'Medium Score (Random-like)', 'Saliency AUC')
    plot_scenario(axes[1, 2], attr_auc_low, 'Low Score (Suppressed signal)', 'Saliency AUC')

    # --- Row 3: Saliency SNR ---
    # High: Very sharp, high-energy peak inside; minimal energy outside.
    attr_snr_high = generate_peak(center=CAUSAL_START + CAUSAL_LEN/2, std=10, height=1.0) + generate_noise(0.05)
    # Medium: Peak inside, but with significant energy from noise outside.
    attr_snr_mid = generate_peak(center=CAUSAL_START + CAUSAL_LEN/2, std=25, height=0.7) + generate_noise(0.3)
    # Low: Most energy is outside the causal region, with background noise.
    attr_snr_low = generate_peak(center=CAUSAL_START + CAUSAL_LEN/2, std=20, height=0.3) + generate_peak(center=800, std=50, height=0.8) + generate_noise(0.05)

    plot_scenario(axes[2, 0], attr_snr_high, 'High Score (Energy concentrated)', 'Saliency SNR')
    plot_scenario(axes[2, 1], attr_snr_mid, 'Medium Score (Energy diffuse)', 'Saliency SNR')
    h, l = plot_scenario(axes[2, 2], attr_snr_low, 'Low Score (Energy outside)', 'Saliency SNR')

    # Create a single legend for the entire figure
    fig.legend(h, l, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=12)

    # Save the figure
    output_path = "metrics_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

def generate_special_auc_plot():
    """Generates a separate plot for special AUC cases."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Special Cases for Saliency AUC ≈ 0.5", fontsize=16, weight='bold')

    # Case 1: Uncorrelated Random Noise
    attr_case1 = generate_noise(0.6)
    plot_scenario(axes[0], attr_case1, 'Case 1: Uncorrelated Random Noise', 'Saliency AUC')

    # Case 2: Confounding Signal Present Systematically
    # Here, the model has learned a feature that repeats across the sequence.
    # Because this feature is present equally inside and outside the causal
    # region, the AUC is close to 0.5.
    attr_case2 = generate_noise(0.05)
    num_peaks = 16
    # Distribute identical peaks across the sequence. One will land in the causal region.
    peak_positions = np.linspace(start=50, stop=950, num=num_peaks, dtype=int)
    for pos in peak_positions:
        attr_case2 += generate_peak(center=pos, std=10, height=1.0)
    
    h, l = plot_scenario(axes[1], attr_case2, 'Case 2: Systematic Confounding Signal', 'Saliency AUC')
    
    # Create a legend for the figure
    fig.legend(h, l, loc='upper right', bbox_to_anchor=(0.99, 0.95), fontsize=10)

    # Save the figure
    output_path = "auc_special_cases.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Special case plot saved to {output_path}")


# --------------------------------------------------------------------------- #
# 5. Effect Size Calculation & Plotting
# --------------------------------------------------------------------------- #

def cohens_d_to_accuracy(d):
    """Converts Cohen's d to the accuracy of an optimal classifier."""
    # Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
    return 50.0 * (1 + math.erf(d / (2 * math.sqrt(2))))

def calculate_gc_cohens_d(gc_pos, n=1000):
    """Calculates Cohen's d for the GC-content confounder."""
    mu_neg = 0.5
    var_neg = 0.5 * (1 - 0.5) / n
    mu_pos = gc_pos
    var_pos = gc_pos * (1 - gc_pos) / n
    if var_neg == 0 and var_pos == 0:
        return 0 if mu_pos == mu_neg else float('inf')
    pooled_std = math.sqrt((var_neg + var_pos) / 2)
    if pooled_std == 0:
        return 0 if mu_pos == mu_neg else float('inf')
    return (mu_pos - mu_neg) / pooled_std

def calculate_motif_information_content(conservation):
    """Calculates the per-base information content (KL divergence vs background)."""
    # At a motif position, what is the probability of emitting the master-base?
    p_master = conservation
    # What is the probability of emitting one of the 3 other bases?
    p_other = (1 - conservation) / 3.0
    
    # Background distribution is uniform (q=0.25)
    q = 0.25
    
    # IC = D_KL(P || Q) = sum over bases [ p(b) * log2(p(b)/q(b)) ]
    term1 = p_master * math.log2(p_master / q) if p_master > 0 else 0
    term2 = 3 * (p_other * math.log2(p_other / q)) if p_other > 0 else 0
    
    # Total IC is sum of information from master base and other 3 bases
    return term1 + term2

def generate_revised_effect_size_plot():
    """
    Generates a plot visualizing more intuitive metrics for signal vs. confounder strength.
    - Confounder: Optimal classifier accuracy based on GC%.
    - Signal: Per-base information content of the motif.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    fig.suptitle("A Priori Strength of Confounder vs. Signal", fontsize=18, weight='bold')

    # --- Panel 1: GC Content Confounder Strength ---
    ax = axes[0]
    gc_range = np.linspace(0.5, 0.7, 200)
    # Calculate Cohen's d then convert to accuracy
    acc_values = [cohens_d_to_accuracy(calculate_gc_cohens_d(g)) for g in gc_range]
    
    ax.plot(gc_range, acc_values, color='firebrick', lw=2.5)
    
    # Annotate the specific hparams used in the experiment
    hparam_acc_values = [cohens_d_to_accuracy(calculate_gc_cohens_d(g)) for g in GC_HPARAMS]
    ax.scatter(GC_HPARAMS, hparam_acc_values, color='black', s=80, zorder=5, label="Experiment Hyperparameters")

    for gc, acc in zip(GC_HPARAMS, hparam_acc_values):
        ax.text(gc + 0.003, acc - 4, f"{acc:.1f}%", fontsize=10, verticalalignment='top')
    
    ax.set_title("Confounder Strength", fontsize=14)
    ax.set_xlabel("Positive-Class GC Content (`gc_pos`)", fontsize=12)
    ax.set_ylabel("Optimal Classifier Accuracy (%)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(45, 105)

    # --- Panel 2: Causal Motif Signal Clarity ---
    ax = axes[1]
    cons_range = np.linspace(0.25, 1.0, 200) # Start at 0.25 (random)
    ic_values = [calculate_motif_information_content(c) for c in cons_range]
    
    ax.plot(cons_range, ic_values, color='darkblue', lw=2.5)
    
    # Annotate the specific hparams used in the experiment
    hparam_ic_values = [calculate_motif_information_content(c) for c in CONS_HPARAMS]
    ax.scatter(CONS_HPARAMS, hparam_ic_values, color='black', s=80, zorder=5, label="Experiment Hyperparameters")

    for cons, ic in zip(CONS_HPARAMS, hparam_ic_values):
        ax.text(cons, ic - 0.08, f"{ic:.2f} bits", fontsize=10, horizontalalignment='center')

    ax.set_title("Signal Clarity", fontsize=14)
    ax.set_xlabel("Motif Conservation (`conservation`)", fontsize=12)
    ax.set_ylabel("Information Content (bits/bp)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(-0.1, 2.1)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])

    # Save the figure
    output_path = "revised_effect_size_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Revised effect size plot saved to {output_path}")


if __name__ == "__main__":
    main()
    generate_revised_effect_size_plot()

# --------------------------------------------------------------------------- #
# 6. Complex Multi-Block Dataset Analysis
# --------------------------------------------------------------------------- #

def generate_complex_dataset_analysis():
    """
    Generates analysis plots for the complex multi-block dataset from merged_experiment.py.
    This dataset has:
    - Positives: 3-4 blocks + promoter, GC = 0.5 + gc_gap
    - Negatives: Mixed population (20% decoy, 20% promoter-only, 60% background), GC ≈ 0.5
    """
    
    # Hyperparameters from the merged experiment
    GC_GAP_HPARAMS = np.linspace(0.0, 0.2, 3).tolist()  # [0.0, 0.1, 0.2]
    CONS_HPARAMS = np.linspace(0.55, 0.95, 3).tolist()  # [0.55, 0.75, 0.95]
    
    # Constants from the data generation
    BLOCK_LEN_MEAN = 55
    PROMOTER_LEN = 6 + 17 + 6  # TTGACA + spacer + TATAAT = 29bp
    SEQ_LEN = 1000
    TARGET_SIGNAL_FRAC = 0.20  # 20% of sequence is signal
    
    # Calculate expected number of blocks (3-4, typically 3.5)
    n_blocks_mean = (TARGET_SIGNAL_FRAC * SEQ_LEN) / BLOCK_LEN_MEAN  # ≈ 3.6
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    fig.suptitle("Complex Multi-Block Dataset: Confounder vs Signal Analysis", fontsize=18, weight='bold')
    
    # --- Panel 1: GC Content Confounder Strength ---
    ax = axes[0, 0]
    
    # For each gc_gap, calculate the effective GC difference
    gc_gaps = np.linspace(0.0, 0.25, 200)
    
    # Positives have GC = 0.5 + gc_gap (with some variance)
    # Negatives are mixed but average around GC = 0.5
    # The effective separation depends on the gc_gap
    
    acc_values = []
    for gap in gc_gaps:
        # Positive class: mean = 0.5 + gap, std ≈ 0.04 (from normal sampling)
        # Negative class: mean = 0.5, std ≈ 0.04
        # For a simple approximation, we can use the means
        gc_pos_mean = 0.5 + gap
        gc_neg_mean = 0.5
        
        # Cohen's d calculation for the GC content difference
        d = calculate_gc_cohens_d(gc_pos_mean, n=SEQ_LEN)
        acc = cohens_d_to_accuracy(d)
        acc_values.append(acc)
    
    ax.plot(gc_gaps, acc_values, color='firebrick', lw=2.5)
    
    # Annotate experiment hyperparameters
    hparam_acc_values = []
    for gap in GC_GAP_HPARAMS:
        gc_pos_mean = 0.5 + gap
        d = calculate_gc_cohens_d(gc_pos_mean, n=SEQ_LEN)
        acc = cohens_d_to_accuracy(d)
        hparam_acc_values.append(acc)
    
    ax.scatter(GC_GAP_HPARAMS, hparam_acc_values, color='black', s=80, zorder=5)
    for gap, acc in zip(GC_GAP_HPARAMS, hparam_acc_values):
        ax.text(gap + 0.005, acc - 2, f"{acc:.1f}%", fontsize=10)
    
    ax.set_title("Confounder Strength (GC Content)", fontsize=14)
    ax.set_xlabel("GC Gap Parameter", fontsize=12)
    ax.set_ylabel("Optimal GC-based Classifier Accuracy (%)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(45, 105)
    
    # --- Panel 2: Signal Strength Components ---
    ax = axes[0, 1]
    
    cons_range = np.linspace(0.25, 1.0, 200)
    
    # Calculate total information content from multiple sources
    total_ic_values = []
    motif_ic_values = []
    promoter_ic_values = []
    
    for cons in cons_range:
        # Information from conserved blocks (3-4 blocks)
        motif_ic_per_bp = calculate_motif_information_content(cons)
        motif_total_ic = motif_ic_per_bp * BLOCK_LEN_MEAN * n_blocks_mean
        
        # Information from promoter (assuming high conservation for core hexamers)
        # TTGACA and TATAAT are highly conserved, spacer is not
        promoter_ic = 2 * 6 * calculate_motif_information_content(0.95)  # Core hexamers
        promoter_ic += 17 * 0  # Spacer has no information
        
        total_ic = motif_total_ic + promoter_ic
        
        motif_ic_values.append(motif_ic_per_bp)
        promoter_ic_values.append(promoter_ic / SEQ_LEN)  # Normalize by seq length
        total_ic_values.append(total_ic / SEQ_LEN)  # Bits per position
    
    ax.plot(cons_range, motif_ic_values, color='darkblue', lw=2.5, label='Motif IC (per bp)')
    
    # Annotate experiment hyperparameters
    hparam_ic_values = [calculate_motif_information_content(c) for c in CONS_HPARAMS]
    ax.scatter(CONS_HPARAMS, hparam_ic_values, color='black', s=80, zorder=5)
    
    for cons, ic in zip(CONS_HPARAMS, hparam_ic_values):
        ax.text(cons, ic - 0.08, f"{ic:.2f} bits", fontsize=10, horizontalalignment='center')
    
    ax.set_title("Signal Clarity (Per-Position Information)", fontsize=14)
    ax.set_xlabel("Motif Conservation", fontsize=12)
    ax.set_ylabel("Information Content (bits/bp)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(-0.1, 2.1)
    ax.legend()
    
    # --- Panel 3: Total Signal Budget ---
    ax = axes[1, 0]
    
    # Show how total signal (in bits) varies with conservation
    total_signal_bits = []
    for cons in cons_range:
        motif_ic_per_bp = calculate_motif_information_content(cons)
        motif_total_bits = motif_ic_per_bp * BLOCK_LEN_MEAN * n_blocks_mean
        promoter_bits = 2 * 6 * calculate_motif_information_content(0.95)
        total_bits = motif_total_bits + promoter_bits
        total_signal_bits.append(total_bits)
    
    ax.plot(cons_range, total_signal_bits, color='darkgreen', lw=2.5)
    
    # Annotate experiment hyperparameters
    hparam_total_bits = []
    for cons in CONS_HPARAMS:
        motif_ic_per_bp = calculate_motif_information_content(cons)
        motif_total_bits = motif_ic_per_bp * BLOCK_LEN_MEAN * n_blocks_mean
        promoter_bits = 2 * 6 * calculate_motif_information_content(0.95)
        total_bits = motif_total_bits + promoter_bits
        hparam_total_bits.append(total_bits)
    
    ax.scatter(CONS_HPARAMS, hparam_total_bits, color='black', s=80, zorder=5)
    
    for cons, bits in zip(CONS_HPARAMS, hparam_total_bits):
        ax.text(cons, bits + 5, f"{bits:.0f} bits", fontsize=10, horizontalalignment='center')
    
    ax.set_title("Total Signal Information Budget", fontsize=14)
    ax.set_xlabel("Motif Conservation", fontsize=12)
    ax.set_ylabel("Total Information (bits per sequence)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # --- Panel 4: Dataset Complexity Summary ---
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create a text summary of the dataset
    summary_text = """
Complex Multi-Block Dataset Structure:

Positive Examples (50%):
• Background: GC = 0.5 + gc_gap (±0.04)
• Signal Elements:
  - 3-4 conserved blocks (~55bp each)
  - 1 σ70-like promoter (TTGACA-17bp-TATAAT)
• Total signal fraction: ~20% of sequence

Negative Examples (50%):
• 20% Decoy negatives:
  - 1-2 conserved blocks
  - Background GC ≈ 0.5 (±0.04)
• 20% Promoter-only negatives:
  - Only promoter, no blocks
  - Background GC ≈ 0.5 (±0.04)
• 60% Pure background:
  - No signal elements
  - Background GC ≈ 0.5 (±0.04)

Key Insights:
1. Confounder: GC content difference between
   positive (0.5+gc_gap) and negative (0.5) classes
   
2. True Signal: Conservation pattern in blocks
   plus promoter structure
   
3. Challenge: Decoy negatives contain partial
   signal, making the task more realistic
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Save the figure
    output_path = "complex_dataset_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Complex dataset analysis saved to {output_path}")


# Update main execution
if __name__ == "__main__":
    main()
    generate_revised_effect_size_plot()
    generate_complex_dataset_analysis() 