# Rashomon CNN Analysis for Motif Discovery

## Overview

We extend the Rashomon set framework to convolutional neural network (CNN) architectures to identify sequence motifs predictive of bacterial sporulation phenotype. Unlike gene-based Rashomon analysis that identifies necessary and common genes, CNN-based analysis examines whether consistent sequence patterns (motifs) emerge across models in the ε-Rashomon set.

## Methods

### Model Architecture

We employ a minimal single-layer CNN architecture to ensure interpretable filter weights that can be directly converted to position weight matrices (PWMs):

```
Input: One-hot encoded DNA sequence (4 × L)
    ↓
Conv1D(filters=48, kernel_size=k, padding='valid')
    ↓
BatchNorm1D
    ↓
Activation (ReLU or Exponential)
    ↓
GlobalMaxPool1D
    ↓
Dense(1, sigmoid)
    ↓
Output: Binary classification probability
```

The exponential activation function is included as an alternative to ReLU because it more closely mimics the biophysical interpretation of PWM scores, where the probability of transcription factor binding increases exponentially with sequence match quality.

### Multi-Architecture Analysis

To capture motifs of varying lengths, we train models with five different kernel sizes:

| Kernel Size | Biological Rationale |
|-------------|---------------------|
| k=6 | Short regulatory elements (TATA boxes, -10/-35 promoter elements) |
| k=9 | Typical transcription factor binding sites |
| k=12 | Extended binding motifs |
| k=15 | Longer structured motifs |
| k=21 | Complex regulatory elements (riboswitches, extended operators) |

For each kernel size and activation function combination (10 configurations total), we train 40 models with different random initializations, yielding 400 total models.

### Training Protocol

**Data Preparation:**
- Genomic sequences truncated to 100,000 bp for computational efficiency
- One-hot encoding with vectorized NumPy operations
- Train/validation split: 82%/18%
- Positive class: Sporulating bacteria; Negative class: Non-sporulating bacteria

**Optimization:**
- Loss function: Binary cross-entropy
- Optimizer: AdamW with learning rate 0.001, weight decay 1×10⁻⁵
- Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=2)
- Gradient clipping: max norm 1.0
- Mixed-precision training (FP16) with gradient scaling
- Maximum epochs: 15
- Early stopping: patience=5 based on validation balanced accuracy

**Evaluation Metric:**
- Balanced accuracy to account for class imbalance

### Rashomon Set Construction

The ε-Rashomon set consists of all trained models whose validation balanced accuracy falls within ε of the best-performing model:

$$\mathcal{R}_\epsilon = \{m : \text{acc}(m) \geq \text{acc}^* - \epsilon\}$$

We use ε = 0.05 (5 percentage points) as the threshold, consistent with our gene-based Rashomon analysis.

### Filter Weight Extraction

For each model in the Rashomon set, we extract the learned convolutional filters. Filter importance is computed as the mean absolute activation across all validation samples:

$$\text{importance}_f = \frac{1}{N} \sum_{i=1}^{N} \max_j |a_{i,f,j}|$$

where $a_{i,f,j}$ is the activation of filter $f$ at position $j$ for sample $i$.

### PWM Conversion

Convolutional filter weights are converted to position weight matrices (PWMs) using softmax normalization:

$$\text{PWM}_{p,b} = \frac{\exp(w_{p,b})}{\sum_{b'} \exp(w_{p,b'})}$$

where $w_{p,b}$ is the filter weight at position $p$ for base $b \in \{A, C, G, T\}$.

### Information Content

Information content (IC) per position is calculated as:

$$\text{IC}_p = 2 + \sum_{b} \text{PWM}_{p,b} \log_2(\text{PWM}_{p,b})$$

yielding values in [0, 2] bits, where 2 bits indicates perfect conservation.

### Motif Clustering

Filters from all Rashomon models are clustered to identify recurring motifs. We use IC-weighted Pearson correlation with reverse complement matching:

1. **IC-Weighted Similarity:** For PWMs $P_i$ and $P_j$:
   $$\text{sim}(P_i, P_j) = \text{corr}_w(P_i, P_j)$$
   where weights are the average IC at each position: $w_p = (\text{IC}_p^{(i)} + \text{IC}_p^{(j)})/2$

2. **Reverse Complement Matching:** For each pair, we compute similarity to both forward and reverse complement orientations, taking the maximum:
   $$\text{sim}^*(P_i, P_j) = \max(\text{sim}(P_i, P_j), \text{sim}(P_i, P_j^{RC}))$$

3. **Hierarchical Clustering:** Average-linkage clustering on the distance matrix $(1 - \text{sim}^*)$, cut at threshold $t$ to form clusters.

### Motif Frequency Analysis

For each motif cluster, we compute:

- **Frequency:** Fraction of Rashomon models containing at least one filter in the cluster
- **Necessary motifs:** Clusters appearing in 100% of Rashomon models
- **Common motifs:** Clusters appearing in ≥50% of Rashomon models

This parallels the gene-level analysis where necessary genes appear in all Rashomon models and common genes appear in the majority.

### Consensus Sequence

Representative consensus sequences are derived from the highest-importance PWM in each cluster:

$$\text{consensus}_p = \arg\max_b \text{PWM}_{p,b}$$

Uppercase letters indicate positions with probability ≥0.5; lowercase otherwise.

## Implementation

The analysis is implemented in Python using:
- PyTorch for neural network training
- CUDA mixed-precision training for efficiency
- SciPy for hierarchical clustering
- NumPy for vectorized sequence encoding

Computation is parallelized across GPU nodes using SLURM array jobs, with each kernel size and activation combination running on a separate T4 GPU.

## Results Summary

Analysis of the Spore formation phenotype reveals:

1. **Model Performance:** Best validation balanced accuracy ranges from 0.75-0.82 depending on kernel size, with ReLU generally outperforming exponential activation.

2. **Rashomon Set Size:** 6-39 models per configuration fall within the ε-Rashomon set.

3. **Complete Filter Independence:** Re-clustering analysis across similarity thresholds (0.4-0.8) reveals that **no two filters from different models are similar enough to cluster together**, even at the most permissive threshold (0.4). The number of clusters equals the total number of filters:

   | Configuration | Models | Total Filters | Clusters (any threshold) | Max Frequency |
   |--------------|--------|---------------|-------------------------|---------------|
   | k12_exp | 6 | 288 | 288 | 16.7% (1/6) |
   | k12_relu | 37 | 1776 | 1776 | 2.7% (1/37) |
   | k15_exp | 25 | 1200 | 1200 | 4.0% (1/25) |
   | k15_relu | 39 | 1872 | 1872 | 2.6% (1/39) |

4. **No Consistent Motifs:** Across all kernel sizes and similarity thresholds tested (0.4-0.8), no motifs reach the "common" (≥50%) or "necessary" (100%) threshold. Maximum frequencies exactly equal 1/n_models, indicating each motif appears in exactly one model.

## Critical Finding: Filters Learn No Discriminative Motifs

Diagnostic analysis of the learned filters reveals a fundamental issue: **filters have near-zero information content**.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Average IC | 0.006 bits | Near-zero (max = 2.0 bits) |
| PWM values | ~0.23-0.29 | Nearly uniform (expected 0.25) |
| Cosine similarity (cross-model) | 0.99+ | All PWMs identical (all uniform) |
| IC-weighted similarity | ~0.09 | Low because IC ≈ 0 |

This means filters are NOT learning sequence motifs—they output nearly uniform probability distributions over bases. The clustering approach is mathematically sound; there is simply nothing meaningful to cluster.

**Where does the predictive signal come from?**
If filters don't learn motifs, the CNN must rely on:
1. **Aggregate statistics**: The GlobalMaxPool aggregates across ~100,000 positions; even tiny deviations from uniform may encode genome-wide composition
2. **Dense layer weights**: The final layer combines 48 near-uniform filter outputs, potentially capturing subtle correlations
3. **Count-based features**: Filter activations may implicitly count k-mer frequencies without learning sharp PWMs

## Ablation Study Results

To validate the hypothesis, we conducted ablation experiments comparing CNN variants with explicit k-mer frequency baselines:

| Model Variant | Balanced Accuracy | Interpretation |
|--------------|-------------------|----------------|
| Full CNN | 0.66 ± 0.06 | Baseline |
| Frozen Random Filters + Trained Dense | 0.50 | Random guessing |
| No Dense Layer (mean pooling → sigmoid) | 0.50 | Random guessing |
| **4-mer Logistic Regression** | **0.689** | Explicit k-mer counting |
| **6-mer Logistic Regression** | **0.713** | Explicit k-mer counting |

**Critical Finding: K-mer baselines outperform the CNN.**

**Key Findings:**
1. **Filters DO matter**: Random frozen filters yield chance performance, so learned filters capture meaningful signal
2. **Dense layer is essential**: Without the dense layer's learned combination weights, the model fails
3. **Filters learn diffuse patterns**: Despite being necessary, filters have IC ≈ 0.006 bits (near-uniform)
4. **K-mer counting is superior**: Simple 6-mer frequency + logistic regression achieves 0.71 vs CNN's 0.66

**Reconciliation**: The CNN attempts to learn k-mer-like statistics implicitly through its filters, but does so inefficiently compared to explicit k-mer counting. The near-uniform filter weights (IC ≈ 0) suggest the CNN is averaging over many positions rather than detecting specific patterns. Explicit k-mer features capture genome composition more directly and effectively.

This confirms that sporulation phenotype is predicted by **genome-wide compositional statistics** (k-mer frequencies), not by discrete sequence motifs. The CNN's filters are essentially noisy approximations of k-mer counters.

## Interpretation

The complete absence of filter similarity across Rashomon CNN models—where no two filters from different models cluster together even at 0.4 similarity threshold—is a strong negative result with important implications:

1. **Extreme Functional Redundancy:** The sequence space contains a vast number of equally predictive patterns. Each random initialization leads the optimization to a completely different region of filter space, yet achieves comparable performance. This suggests the predictive signal is highly distributed and non-specific.

2. **No Discrete Binding Sites:** Unlike transcription factor binding where specific motifs (e.g., TATA box) are evolutionarily conserved and consistently learned, sporulation prediction does not rely on identifiable sequence motifs. The filters appear to capture statistical regularities (e.g., k-mer frequencies, local GC content variations) rather than discrete biological signals.

3. **Global vs. Local Features:** Gene-based Rashomon analysis identifies specific genes as necessary or common because genes represent functional units with biological meaning. In contrast, short sequence windows (6-21 bp) may not capture the relevant biological units, which could be:
   - Longer regulatory elements spanning hundreds of base pairs
   - Gene-level features (presence/absence, copy number)
   - Genome-wide compositional biases

4. **Implications for Interpretability:** Single-layer CNNs, while mathematically interpretable (filters → PWMs), do not yield biologically interpretable features for this phenotype. The Rashomon framework reveals this limitation: if consistent patterns existed, they would be discovered across multiple random initializations.

### Comparison with Gene-Level Analysis

| Aspect | Gene-Based Rashomon | CNN-Based Rashomon |
|--------|--------------------|--------------------|
| Feature unit | Gene (biological unit) | 6-21 bp window (arbitrary) |
| Necessary features | Yes (specific genes) | No |
| Common features | Yes (≥50% frequency) | No |
| Interpretation | Biologically meaningful | No consistent patterns |
| Conclusion | Specific genes required | Distributed, non-specific signal |

This contrast strongly suggests that sporulation phenotype prediction relies on gene-level features rather than short sequence motifs, consistent with the biological understanding that sporulation is controlled by complex regulatory networks involving specific genes (e.g., *spoIIA*, *spoIIE*, *sigF*) rather than simple sequence patterns.
