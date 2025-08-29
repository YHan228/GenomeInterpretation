import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from matplotlib.colors import TwoSlopeNorm
import scipy.special
from scipy.special import logsumexp

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


# --------------------------------------------------------------------------- #
# 6. Information-Theoretic Visualization
# --------------------------------------------------------------------------- #

def _kl_divergence(p, q):
    """Calculates KL divergence D_KL(P || Q) for discrete distributions."""
    p = np.asarray(p, dtype=float) + 1e-12
    q = np.asarray(q, dtype=float) + 1e-12
    return np.sum(p * np.log2(p / q))

def jsd(p, q):
    """Calculates Jensen-Shannon Divergence (JSD) in bits."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)

def get_bg_distribution(gc_content):
    """Returns the nucleotide distribution [A, C, G, T] for a given GC content."""
    return np.array([
        (1 - gc_content) / 2,  # A
        gc_content / 2,        # C
        gc_content / 2,        # G
        (1 - gc_content) / 2   # T
    ])

def _bg_Q(gc):
    # A,C,G,T
    return np.array([(1-gc)/2, gc/2, gc/2, (1-gc)/2], dtype=float)

def jsd_chunk_bits(m, conservation, gc_pos, gc_neg=0.5, b0=2, N=5000, seed=1):
    """
    JSD between P1^{joint} and P0^{joint} for an m-bp motif chunk.
    P1 per-site: with prob c emit canonical base b0, else sample background Q_pos.
    P0 per-site: background Q_neg.  Returns bits in [0,1].
    """
    rng = np.random.default_rng(seed)
    Qpos = _bg_Q(gc_pos); Qneg = _bg_Q(gc_neg)
    Ppos = (1.0 - conservation) * Qpos
    Ppos[b0] += conservation
    Pneg = Qneg

    # sample chunks as integer codes in {0,1,2,3}
    pos = rng.choice(4, size=(N, m), p=Ppos)
    neg = rng.choice(4, size=(N, m), p=Pneg)

    # log-likelihoods under joint (sum of per-site logs, natural logs)
    lp1_pos = np.sum(np.log(Ppos[pos]), axis=1)
    lp0_pos = np.sum(np.log(Pneg[pos]), axis=1)
    lp1_neg = np.sum(np.log(Ppos[neg]), axis=1)
    lp0_neg = np.sum(np.log(Pneg[neg]), axis=1)

    # JSD = 0.5*E_pos[ log2( 2*p1 / (p1+p0) ) ] + 0.5*E_neg[ log2( 2*p0 / (p1+p0) ) ]
    ln2 = np.log(2.0)
    term_pos = (ln2 + lp1_pos - logsumexp(np.vstack([lp1_pos, lp0_pos]), axis=0)) / ln2
    term_neg = (ln2 + lp0_neg - logsumexp(np.vstack([lp1_neg, lp0_neg]), axis=0)) / ln2
    return 0.5 * (term_pos.mean() + term_neg.mean())

def jsd_gc_confounder_bits(L, mu1, mu0):
    """Calculates exact discrete JSD between Binomial(L, mu1) and Binomial(L, mu0)."""
    k = np.arange(L + 1)
    logC = (scipy.special.gammaln(L + 1) - scipy.special.gammaln(k + 1)
            - scipy.special.gammaln(L - k + 1))
    def pmf(mu):
        # Use np.log1p for numerical stability when mu is small
        p = np.exp(logC + k * np.log(mu) + (L - k) * np.log1p(-mu))
        return p / p.sum()
    p1, p0 = pmf(mu1), pmf(mu0)
    return jsd(p1, p0)

def calculate_information_strengths(conservation, gc_pos,
                                    L=SEQ_LEN, motif_bp=CAUSAL_LEN,
                                    realised_frac=None, b0=2, N_mc=5000):
    """
    Calculates total signal and confounder information in bits using JSD.
    """
    # Confounder: keep exact binomial JSD you already implemented
    info_confounder = jsd_gc_confounder_bits(L, mu1=gc_pos, mu0=0.5)

    # Signal (vanilla): chunk-level JSD for the whole causal chunk (m = motif_bp)
    if realised_frac is None:
        info_signal = jsd_chunk_bits(motif_bp, conservation, gc_pos, gc_neg=0.5, b0=b0, N=N_mc)
    else:
        # Complex: approximate by scaling the Monte-Carlo sample count with expected covered bp
        # Draw chunks of length round(realised_frac*L) for signal estimate
        m_eff = max(1, int(round(realised_frac * L)))
        info_signal = jsd_chunk_bits(m_eff, conservation, gc_pos, gc_neg=0.5, b0=b0, N=N_mc)
    return info_signal, info_confounder

def generate_information_heatmap():
    # Grid: avoid exactly 0.5 to prevent singularities
    gc_values   = np.linspace(0.502, 0.70, 50)
    cons_values = np.linspace(0.50, 1.00, 50)

    Isig  = np.zeros((len(cons_values), len(gc_values)))
    Iconf = np.zeros_like(Isig)

    for i, cons in enumerate(cons_values):
        for j, gc in enumerate(gc_values):
            Isig[i, j], Iconf[i, j] = calculate_information_strengths(
                conservation=cons, gc_pos=gc, L=SEQ_LEN, motif_bp=CAUSAL_LEN, N_mc=3000
            )

    # --- Mask near-zero confounder to avoid infinite ratios ---
    eps_bits = 1e-3  # bits threshold for masking
    mask = (Iconf < eps_bits)
    ratio = Isig / np.maximum(Iconf, eps_bits)  # safe ratio for display
    log_ratio = np.log10(ratio)                 # center at 0 == parity

    # --- Figure with three panels: Isig, Iconf, log10 ratio ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    fig.suptitle("Information-Theoretic View (total bits and log-ratio)", fontsize=16, weight='bold')

    extent = [gc_values[0], gc_values[-1], cons_values[0], cons_values[-1]]

    # Panel A: signal bits
    im0 = axes[0].imshow(Isig, origin='lower', aspect='auto', cmap='viridis',
                         extent=extent)
    axes[0].set_title("Signal (motif) bits — chunk‑level JSD")
    axes[0].set_xlabel("gc_pos"); axes[0].set_ylabel("conservation")
    cbar0 = fig.colorbar(im0, ax=axes[0]); cbar0.set_label("bits")

    # Panel B: confounder bits
    im1 = axes[1].imshow(np.ma.masked_where(mask, Iconf), origin='lower',
                         aspect='auto', cmap='viridis', extent=extent)
    axes[1].set_title("Confounder (GC%) bits")
    axes[1].set_xlabel("gc_pos")
    cbar1 = fig.colorbar(im1, ax=axes[1]); cbar1.set_label("bits")
    # Hatch masked region
    if mask.any():
        axes[1].contour(gc_values, cons_values, mask, levels=[0.5],
                        colors='k', linestyles=':', linewidths=1)

    # Panel C: log10 ratio, centered at 0
    from matplotlib.colors import TwoSlopeNorm
    # Robust range: ignore top 1% outliers even after masking
    finite_lr = log_ratio[~mask]
    vmin = np.percentile(finite_lr, 1)
    vmax = np.percentile(finite_lr, 99)
    vcenter = 0.0
    if vmax <= vcenter: vmax = vcenter + 1e-6
    if vmin >= vcenter: vmin = vcenter - 1e-6

    im2 = axes[2].imshow(np.ma.masked_where(mask, log_ratio), origin='lower',
                         aspect='auto', cmap='coolwarm',
                         norm=TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax),
                         extent=extent)
    axes[2].set_title("log10(Signal/Confounder)")
    axes[2].set_xlabel("gc_pos")
    cbar2 = fig.colorbar(im2, ax=axes[2])
    # Nice ticks: show powers of 10 and their meaning
    ticks = np.linspace(np.floor(vmin), np.ceil(vmax), 5)
    cbar2.set_ticks(ticks)
    cbar2.set_ticklabels([f"{10**t:.1f}×" for t in ticks])

    # Mark experimental hparams
    for ax in axes:
        for gc in GC_HPARAMS:
            if gc < gc_values[0]: continue
            for cons in CONS_HPARAMS:
                ax.scatter(gc, cons, marker='x', color='yellow', s=50, lw=1.5, zorder=5)

    out = "information_theoretic_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Information-theoretic heatmaps saved to {out}")

def run_acceptance_checks():
    """Runs a series of assertions to check the behavior of the new functions."""
    print("\nRunning acceptance checks for information-theoretic functions...")
    
    # Check 1: Confounder bits increase with sequence length
    assert jsd_gc_confounder_bits(2000, 0.6, 0.5) > jsd_gc_confounder_bits(1000, 0.6, 0.5)
    
    # Check 2: Confounder bits increase with GC gap
    assert jsd_gc_confounder_bits(1000, 0.62, 0.5) > jsd_gc_confounder_bits(1000, 0.58, 0.5)
    
    # Check 3: Chunk-level signal bits are in [0,1] and increase with conservation
    assert 0 <= jsd_chunk_bits(60, 0.2, 0.6) <= 1
    assert jsd_chunk_bits(60, 0.8, 0.6) > jsd_chunk_bits(60, 0.2, 0.6)

    # Check 4: Approach saturation as m grows / c→1
    v1 = jsd_chunk_bits(60, 0.99, 0.6, N=2000)
    v2 = jsd_chunk_bits(120, 0.99, 0.6, N=2000)
    assert v2 >= v1 and v2 <= 1.0 + 1e-6

    # Check 5: Vanish near c≈0
    assert jsd_chunk_bits(60, 1e-6, 0.6) < 1e-3
    
    print("All acceptance checks passed.\n")
    

if __name__ == "__main__":
    main()
    generate_revised_effect_size_plot()
    #run_acceptance_checks()
    generate_information_heatmap()