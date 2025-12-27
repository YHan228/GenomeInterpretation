# Predictive Multiplicity (H1) Analysis

## Overview

This analysis demonstrates **Predictive Multiplicity** - the phenomenon where different valid feature selection methods identify different feature sets that achieve similar predictive performance.

Two scripts are available:
- `multiplicity_h1_slim.py` - **Recommended for thesis**: Clean CPSS comparison (L1-logistic vs RF)
- `multiplicity_h1.py` - Comprehensive version with additional methods (Boruta, CPI, etc.)

## Slim Version (`multiplicity_h1_slim.py`)

Uses **Complementary-Pairs Stability Selection (CPSS)** with two base learners:

| Method | Model | Selection Mechanism |
|--------|-------|---------------------|
| CPSS-Logistic | L1-regularized logistic | Natural sparsity from L1 penalty |
| CPSS-RF | Random Forest | Top-K features by MDI (importance) |

### What is CPSS?

CPSS repeatedly splits data into complementary halves (like 2-fold CV), fits a model on each half, and records which features are selected. The **selection probability** π̂ measures how often each feature is selected across all half-samples.

- High π̂ (≥ τ): Feature is consistently selected → stable
- Low π̂: Feature selection is sensitive to data perturbation → unstable

This directly demonstrates **H1 (multiplicity exists)**: even the same method on different data splits selects different features.

### Usage

```bash
# Via SLURM (all 13 phenotypes)
sbatch slurm_scripts/run_sporulation_h1_slim.sh

# Single phenotype
python sporulation/code/multiplicity_h1_slim.py \
  --phenotype "Spore formation" \
  --cpss_pairs 100 \
  --cpss_tau 0.7
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cpss_pairs` | 100 | Number of complementary pairs (200 fits total) |
| `--cpss_tau` | 0.7 | Stability threshold (π̂ ≥ τ → stable) |
| `--cpss_q` | 100 | Target selection size for L1-logistic |
| `--rf_top_k` | 100 | Top-K features per RF fit |
| `--rf_trees` | 600 | Trees per RF |
| `--rf_max_depth` | 30 | Max tree depth |

### Outputs

All figures saved in PNG, PDF, and SVG.

| File | Description |
|------|-------------|
| `lasso_selection_probs` | Top-N selection probabilities (LASSO) |
| `lasso_hist` | Histogram of all π̂ values (LASSO) |
| `lasso_size_vs_tau` | |S_τ| vs threshold τ (LASSO) |
| `rf_selection_probs` | Top-N selection probabilities (RF) |
| `rf_hist` | Histogram of all π̂ values (RF) |
| `rf_size_vs_tau` | |S_τ| vs threshold τ (RF) |
| `rank_comparison_lasso_vs_rf` | Rank scatter + Spearman ρ |
| `overlap_network` | MDS visualization of bag similarities |
| `jaccard_violins` | Within/cross-method Jaccard distributions |
| `stable_set_overlap` | Venn-style bar: LASSO-only / both / RF-only |
| `h1_summary.json` | All statistics |
| `stable_genes.csv` | Features stable in either/both methods |

### Interpreting Results

#### 1. Selection Probability (π̂)

The fraction of half-samples in which a feature was selected.

| π̂ Range | Interpretation |
|---------|----------------|
| 0.9 - 1.0 | **Highly stable**: Selected almost always, strong signal |
| 0.7 - 0.9 | **Stable**: Consistently selected (above default τ=0.7) |
| 0.3 - 0.7 | **Unstable**: Selection depends on data split |
| 0.0 - 0.3 | **Rarely selected**: Weak or no signal |

**What to look for:**
- Histogram shape: Bimodal (peaks at 0 and 1) = clear signal/noise separation
- Histogram shape: Uniform or single peak in middle = high multiplicity, no clear stable set

#### 2. Nogueira Stability Index

Measures agreement across all selection sets, accounting for set sizes.

| Value | Interpretation |
|-------|----------------|
| > 0.8 | **High stability**: Selections are consistent |
| 0.5 - 0.8 | **Moderate**: Some consistency, notable variation |
| 0.2 - 0.5 | **Low**: Substantial multiplicity |
| < 0.2 | **Very low**: Selections are nearly random |

**Formula intuition:** 1 = perfect agreement (all splits select identical features), 0 = no more agreement than random chance.

#### 3. Jaccard Similarity

Pairwise overlap between selection sets: |A ∩ B| / |A ∪ B|

| Comparison | Typical Range | Interpretation |
|------------|---------------|----------------|
| **Within-method** | 0.3 - 0.7 | How much do splits of the same method agree? |
| **Cross-method** | 0.1 - 0.4 | How much do LASSO and RF agree? |

**What indicates multiplicity:**
- Within-method Jaccard < 0.5 → same method, different splits → different features
- Cross-method Jaccard < within-method → methods select different feature sets
- Cross-method ≈ 0 → almost no overlap between methods

**Expected under random selection** (for reference):
- If each bag selects K features from P total: E[Jaccard] ≈ K / (2P - K)
- With K=100, P=5000: E[Jaccard] ≈ 0.01
- Observed >> expected → selections are non-random but still show multiplicity

#### 4. Stable Set Overlap (Venn diagram)

| Category | Interpretation |
|----------|----------------|
| **Both** | Core features: stable in both LASSO and RF |
| **LASSO-only** | Linear-specific: captured by sparse linear model |
| **RF-only** | Nonlinear-specific: captured by tree ensemble |

**What to report:**
- If "Both" >> "LASSO-only" + "RF-only": Methods largely agree on stable features
- If "LASSO-only" ≈ "RF-only" >> "Both": Strong cross-method multiplicity
- Ratio: Both / (LASSO-only + RF-only + Both) = agreement fraction

#### 5. Rank Comparison (Spearman ρ)

Correlation between LASSO and RF rankings of features by π̂.

| ρ Value | Interpretation |
|---------|----------------|
| > 0.7 | **Strong agreement**: Methods rank features similarly |
| 0.4 - 0.7 | **Moderate**: Some agreement in top features |
| 0.0 - 0.4 | **Weak**: Methods prioritize different features |
| < 0 | **Negative**: Methods actively disagree |

#### Summary: What Demonstrates H1?

**H1 (multiplicity exists) is demonstrated when:**

1. **Within-method Jaccard < 0.5** — Same algorithm on different splits selects different features
2. **Cross-method Jaccard < within-method** — Different algorithms select different features
3. **Nogueira stability < 0.7** — Selection is sensitive to data perturbation
4. **Stable set has method-specific members** — LASSO-only and RF-only sets are non-empty
5. **Spearman ρ < 0.7** — Rankings differ between methods

**Example interpretation:**
> "CPSS-Logistic achieved Nogueira stability of 0.45 with mean within-method Jaccard of 0.38, indicating substantial selection variability across data splits. Cross-method Jaccard (0.22) was lower than within-method, and only 15 of 45 stable features were shared between methods (33%), demonstrating that both within-method and cross-method multiplicity exist."

---

## Full Version (`multiplicity_h1.py`)

Includes additional methods beyond CPSS:
- **Boruta**: All-relevant feature selection
- **Permutation VI**: Validation-based importance
- **CPI**: Conditional Permutation Importance

Use this for comprehensive analysis; the slim version is sufficient for demonstrating H1.

---

## SLURM Configuration

| Script | Time | Memory | Purpose |
|--------|------|--------|---------|
| `run_sporulation_h1_slim.sh` | 1 day | 200GB | Slim version |
| `run_sporulation_h1.sh` | 2 days | 300GB | Full version |

Both use 50 CPUs and array jobs for 13 phenotypes.

## Caching

Results are cached in `.cache/` subdirectory:
- `cpss_lasso.npz`: LASSO selection matrix and π̂
- `cpss_rf.npz`: RF selection matrix and π̂

Delete cache files to force recomputation.
