# Saliency-Aware Neural Network Training: Workflow Summary

This document describes the computational workflow for benchmarking saliency-aware CNN training, implemented across two experimental settings: synthetic DNA sequences (`toy_single_arch.py`) and real genomic phenotype prediction (`optuna_phenotype.py`).

---

## Part 1: Common Workflow Concepts

### 1.1 Two-Phase Hyperparameter Optimization

Both experiments follow a two-phase strategy:

| Phase | Goal | What is tuned | What is fixed |
|-------|------|---------------|---------------|
| **Phase 1 (Standard)** | Find best CNN architecture | Architecture + optimizer | Training regime (clean inputs) |
| **Phase 2 (Robust)** | Find best robustness hyperparameters | Regime type, epsilon/flip_fraction, scheduling | Architecture from Phase 1 |

**Rationale**: Decoupling architecture search from robustness tuning reduces the search space and ensures fair comparison of training regimes.

---

### 1.2 CNN Architecture (Shared Design)

Both experiments use a 1D CNN with the following pattern:

```
[Conv1 (large kernel) + BN + Activation + Dropout + Pool] ->
[Conv2..N (small kernels) + BN + ReLU + Dropout + Pool]* ->
[Global Pool -> FC -> Logits]
```

**Key interpretability-favoring choices**:
- Large first-layer kernel (`k1`) to capture motif-scale patterns
- Exponential activation (`exp`) option in first layer for sharper feature detection
- Aggressive early pooling to reduce sequence length while preserving positional information

---

### 1.3 Training Regimes

| Regime | Description | Key Parameter |
|--------|-------------|---------------|
| **Standard** | Clean inputs, BCE/CE loss | - |
| **Random Smoothing** | Replace one-hot with Dirichlet samples | `epsilon` in [0.001, 0.5] |
| **Gaussian Smoothing** | Add N(0, sigma^2) noise to inputs | `sigma2` in [1e-5, 0.2] |
| **HotFlip (Iterative)** | Gradient-based single-flip per iteration | `max_flip_fraction` in [0.001, 0.3] |
| **Direct HotFlip** | Single gradient, apply top-k flips at once | `max_flip_fraction` in [0.001, 0.5] |

**Flip Scheduling** (HotFlip variants):
- `linear`: Ramp from 1 flip to `max_flips` linearly over epochs
- `cosine`: Smooth cosine ramp (slower start/end)
- `none`: Apply full `max_flips` from epoch 1

---

### 1.4 Optimization Setup

| Component | Configuration |
|-----------|---------------|
| **Optimizer** | AdamW with tunable LR and weight decay |
| **Scheduler** | ReduceLROnPlateau (factor=0.5, patience varies) |
| **Mixed Precision** | AMP with GradScaler |
| **Gradient Clipping** | Tunable threshold (typically 0.5-10.0) |
| **Early Stopping** | Patience-based on validation loss |

---

### 1.5 Optuna HPO Configuration

| Setting | Value |
|---------|-------|
| **Sampler** | TPE (Tree-structured Parzen Estimator) |
| **Pruner** | Hyperband (min_resource=5, reduction_factor=3) |
| **Objective** | Weighted combination of saliency metric + accuracy |
| **Parallelization** | Multiple SLURM workers share study via MySQL/PostgreSQL |

---

### 1.6 Saliency Evaluation

**Method**: Integrated Gradients (IG) with gradient correction

**Gradient Correction**: After computing raw IG attributions, subtract the mean across nucleotides at each position:
```
corrected_attr[i,j] = raw_attr[i,j] - mean_over_i(raw_attr[i,j])
```
This ensures attributions sum to zero at each position, appropriate for one-hot encoded DNA.

**Common Metrics**:

| Metric | Definition | Formula |
|--------|------------|---------|
| **Saliency AUC** | Probability that a randomly chosen inside position has higher attribution than a randomly chosen outside position | `mean(attr_inside[:, None] > attr_outside[None, :])` |
| **Saliency SNR** | Fraction of squared attribution "energy" within ground truth region | `sum(attr_inside²) / sum(attr_total²)` |
| **Saliency AUPR** (Koo) | Area under Precision-Recall curve treating motif positions as binary labels | `sklearn.auc(recall, precision)` where labels derived from information content |
| **wIoU** | Intersection-over-Union between top-k attribution window and ground truth | `|pred ∩ gt| / |pred ∪ gt|` |

**Koo's Saliency AUPR** (used in `koo/` experiments):
1. Compute ground truth information content per position: `I[j] = log2(4) + Σ_i PWM[i,j] * log2(PWM[i,j])`
2. Create binary labels: `label[j] = 1 if I[j] > 0.01 else 0`
3. Use summed attributions as scores: `score[j] = Σ_i attr[i,j]`
4. Compute PR-AUC: `auc(recall, precision)` from `precision_recall_curve(label, score)`

---

## Part 2: Synthetic Dataset Experiment

**Script**: `toy_single_arch.py`

### 2.1 Synthetic Data Generation

| Parameter | Value |
|-----------|-------|
| Sequence length (`SEQ_LEN`) | 1000 bp |
| Motif length (`CHUNK_LEN`) | 60 bp |
| Dataset size (`N_TOTAL`) | 10,000 sequences (balanced: 5000 positive, 5000 negative) |
| Positive class | Background + planted motif at random position |
| Negative class | Pure background (no motif) |

**Dataset Grid**:

| Parameter | Values | Count |
|-----------|--------|-------|
| `GC_HPARAMS` | 0.50, 0.53, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75, 0.775, 0.80 | 13 |
| `CONS_HPARAMS` | 0.60, 0.65, 0.70, 0.75, 0.80 | 5 |
| **Total** | 13 × 5 = **65 datasets** | |

**Confounding Design**: Negative class always has GC = 0.50, creating a spurious GC-content shortcut.

```
┌─────────────────────────────────────────────────────────────────┐
│  POSITIVE CLASS (y=1)                                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Background (GC = gc_pos, e.g. 0.70)                      │   │
│  │  ════════════════╔══════════╗════════════════════════    │   │
│  │                  ║  MOTIF   ║  (60 bp, conserved)        │   │
│  │                  ╚══════════╝                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│  Signal: Motif presence   |   Shortcut: High GC background      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  NEGATIVE CLASS (y=0)                                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Background (GC = 0.50, FIXED)                            │   │
│  │  ────────────────────────────────────────────────────    │   │
│  │                  (no motif)                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│  Shortcut: Low GC background distinguishes from positive        │
└─────────────────────────────────────────────────────────────────┘
```

A naive model can achieve high accuracy by learning GC content alone, ignoring the motif.

---

#### 2.1.1 Background Sequence Generation

Background sequences are generated by i.i.d. sampling nucleotides with specified GC content:

```
FUNCTION sample_background(length, gc):
    p_A = p_T = (1 - gc) / 2
    p_C = p_G = gc / 2
    RETURN random_choice(['A','C','G','T'], size=length, p=[p_A, p_C, p_G, p_T])
```

---

#### 2.1.2 Master Motif Creation

Each dataset has a single **master motif** (60 bp) generated once with a fixed seed for consistency:

```
FUNCTION create_master_motif(gc_pos):
    SET random seed to 42  # Fixed for reproducibility
    master_chunk = sample_background(CHUNK_LEN=60, gc=gc_pos)
    RESTORE random state
    RETURN master_chunk
```

The master motif is a random sequence matching the positive-class GC content. All positive examples contain mutations of this same master motif.

---

#### 2.1.3 Motif Mutation

Individual motif instances are created by mutating the master motif with controlled **conservation** level:

```
FUNCTION mutate(chunk, conservation, gc_target):
    n_to_mutate = int(len(chunk) * (1.0 - conservation))
    positions = random_sample(range(len(chunk)), n_to_mutate, replace=False)

    # Mutation distribution matches target GC
    p = [p_A, p_C, p_G, p_T]  where p_C = p_G = gc_target/2, p_A = p_T = (1-gc_target)/2

    FOR each position in positions:
        original_base = chunk[position]
        # Sample new base from remaining 3 bases, weighted by GC target
        temp_p = p.copy()
        temp_p[original_base] = 0
        temp_p = normalize(temp_p)
        chunk[position] = random_choice(['A','C','G','T'], p=temp_p)

    RETURN mutated_chunk
```

**Conservation levels**: 0.60, 0.65, 0.70, 0.75, 0.80
- Conservation 0.80 → 20% of positions mutated (12/60 bp)
- Conservation 0.60 → 40% of positions mutated (24/60 bp)

---

#### 2.1.4 Complete Dataset Generation Algorithm

```
FUNCTION generate_dataset(gc_pos, conservation):
    GC_NEG = 0.50  # Fixed for negative class (confounder)

    # Create master motif (same for all positive examples in this dataset)
    master_chunk = create_master_motif(gc_pos)

    X, y, masks = [], [], []

    # --- Generate Positive Examples ---
    FOR i = 1 to POS_N (5000):
        # 1. Generate background with positive-class GC
        bg = sample_background(SEQ_LEN=1000, gc=gc_pos)

        # 2. Create mutated motif instance
        chunk = mutate(master_chunk, conservation, gc_target=gc_pos)

        # 3. Embed motif at random position
        start = random_int(0, SEQ_LEN - CHUNK_LEN)
        bg[start : start+60] = chunk

        # 4. Store one-hot encoding, label, and ground truth mask
        X.append(one_hot(bg))        # Shape: (4, 1000)
        y.append(1)
        mask = zeros(1000, dtype=bool)
        mask[start : start+60] = True
        masks.append(mask)

    # --- Generate Negative Examples ---
    FOR i = 1 to NEG_N (5000):
        # Pure background with FIXED GC=0.50 (creates shortcut)
        bg = sample_background(SEQ_LEN=1000, gc=GC_NEG)

        X.append(one_hot(bg))
        y.append(0)
        masks.append(zeros(1000, dtype=bool))  # No motif

    RETURN Dataset(X, y, masks)
```

---

#### 2.1.5 One-Hot Encoding

```
FUNCTION one_hot(seq):
    # seq: array of chars ['A','C','G','T',...]
    # output: (4, seq_len) float32 array
    arr = zeros((4, len(seq)))
    FOR i, base in enumerate(seq):
        arr[base_to_index[base], i] = 1.0
    RETURN arr

# base_to_index: {'A':0, 'C':1, 'G':2, 'T':3}
```

---

#### 2.1.6 Dataset Caching

Generated datasets are cached to disk (`.pt` files) to avoid regeneration:
- Cache path: `synthetic/data/gc{gc:.3f}_cons{cons:.2f}.pt`
- Contains: `(X_tensor, y_tensor, masks_array)`

---

#### 2.1.7 Adversarial Flip Vulnerability Analysis

**Question**: At what (GC, conservation) combinations do adversarial flips destroy the motif vs. suppress the GC confounder?

##### Attack Cost Formulas

| Target | Formula | Range |
|--------|---------|-------|
| **GC Signal** | `n_flips = \|GC_pos - 0.50\| × 1000` | 0–300 flips |
| **Motif Signal** | `n_flips = d × cons × 60` | 18–24 flips (d=0.5) |

Where `d` = destruction threshold (fraction of conserved positions to flip).

##### Assumptions

1. **GC attack**: Each flip optimally converts G/C↔A/T. Gradient-based attacks achieve ~90% optimality for distributed signals.

2. **Motif attack**: To destroy the motif, flip `d` fraction of the `cons × 60` conserved positions. At d=0.5: need to flip 18–24 positions.

3. **Attack targeting**: Adversarial flips go where gradients are largest, i.e., toward whatever signal the model currently relies on.

##### The Paradox

The calculation shows motif is "easy to destroy" (18–24 flips), while GC requires up to 300 flips. Yet robust training IMPROVES interpretability. Why?

##### Resolution: Model-Dependent Targeting

Adversarial flips target **the signal the model currently uses**, not the most vulnerable signal:

```
Timeline of Robust Training:

1. EARLY EPOCHS
   └─ Model learns GC shortcut (simpler, global signal)
   └─ Flips target GC-affecting positions (spread over 1000bp)
   └─ Motif region receives only ~6% of flips by chance
   └─ Result: Motif remains intact

2. MIDDLE EPOCHS
   └─ GC signal suppressed
   └─ Model shifts to motif as backup signal
   └─ Gradients begin pointing to motif region

3. LATE EPOCHS
   └─ Model has learned motif-based classification
   └─ Good interpretability achieved
```

##### Quantitative Example

At GC=0.70, cons=0.80, flip_fraction=0.10 (100 flips/batch):

| If model uses... | Where flips go | Motif damage |
|-----------------|----------------|--------------|
| **GC shortcut** | Spread over 1000bp | ~6 flips hit motif (25% of threshold) |
| **Motif** (hypothetical) | Concentrated in 60bp | 100 flips hit motif (417% of threshold) |

**Key Insight**: The motif survives BECAUSE the model initially learns the GC shortcut, directing flips away from the motif region.

##### Vulnerability Ratio Table

| GC | Conservation | n_GC | n_motif | Ratio | Interpretation |
|----|--------------|------|---------|-------|----------------|
| 0.50 | 0.60–0.80 | 0 | 18–24 | 0 | No GC signal, flips would target motif |
| 0.55 | 0.60–0.80 | 50 | 18–24 | 2.1–2.8 | Similar difficulty |
| 0.70 | 0.60–0.80 | 200 | 18–24 | 8–11 | Motif much more vulnerable (if targeted) |
| 0.80 | 0.60–0.80 | 300 | 18–24 | 12–17 | Motif extremely vulnerable (if targeted) |

##### Implications for Training

1. **Low GC datasets** (GC ≈ 0.50): No GC shortcut exists → model must use motif → flips would destroy motif → robust training may hurt
2. **High GC datasets** (GC > 0.55): Strong GC shortcut → model uses GC first → flips suppress GC → motif survives → robust training helps
3. **Higher conservation**: Slightly more robust motif (24 vs 18 flips), but effect is secondary to GC

##### Uniform Random Flipping Analysis

The targeted attack analysis assumes all flips are directed at a specific signal. Under **uniform random flipping** (where each position is equally likely to be flipped), motif destruction probability follows a hypergeometric distribution.

**Setup**:
- Population size: `N = SEQ_LEN = 1000` (total positions)
- Success states: `K = cons × 60` (conserved motif positions)
- Sample size: `n = flip_fraction × 1000` (number of flips)
- Threshold: `t = d × K` (flips needed to destroy, typically d=0.5)

**Distribution**: Let `X` = number of flips landing in conserved motif positions

```
X ~ Hypergeometric(N=1000, K=cons×60, n=flip_fraction×1000)

Expected value: E[X] = n × K / N = flip_fraction × cons × 60

Variance: Var(X) = n × (K/N) × (1-K/N) × (N-n)/(N-1)
```

**Probability of Motif Destruction**:

```
P(motif destroyed) = P(X >= t) = 1 - F_X(t-1)

where F_X is the hypergeometric CDF
```

**Numerical Results** (d=0.5):

| Conservation | Conserved positions | Threshold | flip_fraction for P=0.5 |
|--------------|---------------------|-----------|-------------------------|
| 0.60 | 36 | 18 | ~0.49 |
| 0.70 | 42 | 21 | ~0.49 |
| 0.80 | 48 | 24 | ~0.49 |

At `flip_fraction = 0.10` (100 flips), typical training range:
- Expected flips in motif: `E[X] = 0.10 × cons × 60 ≈ 4.8` (at cons=0.80)
- Probability of destruction: `P(X >= 24) ≈ 0` (negligible)

**Key Comparison**:

| Scenario | Flips needed for destruction | At flip_fraction=0.10 |
|----------|------------------------------|----------------------|
| **Targeted attack** (all flips to motif) | 18-24 | Motif destroyed |
| **Uniform random** | ~490 (for P=0.5) | P(destroyed) ≈ 0 |

This explains why robust training is effective: even if targeted attacks *could* destroy the motif easily, the actual flip distribution depends on what the model learned, and when it uses the GC shortcut, flips are distributed across all 1000 positions—approximately uniform from the motif's perspective.

**Visualization**: See `synthetic/docu/flip_vulnerability_comprehensive.png`

---

### 2.2 Model Architecture

The model is a 3-layer 1D CNN (`TinyCNN`) designed for motif detection in DNA sequences.

**Input**: One-hot encoded sequence, shape `(batch, 4, 1000)` where channels are A/C/G/T.

**Architecture**:

```
Input (batch, 4, 1000)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Conv Block 1 (Motif Scanner)                        │
│   Conv1d(4 → c1, kernel=k1, padding=(k1-1)//2)     │
│   BatchNorm1d(c1)                                   │
│   Activation (exp | relu | gelu | softplus)         │
│   Dropout(drop_conv1)                               │
│   MaxPool1d(pool_w)  → shape: (batch, c1, 1000/pool_w) │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Conv Block 2                                        │
│   Conv1d(c1 → c2, kernel=k2, padding=(k2-1)//2)    │
│   BatchNorm1d(c2)                                   │
│   ReLU                                              │
│   Dropout(drop_conv2)                               │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Conv Block 3                                        │
│   Conv1d(c2 → c3, kernel=k3, padding=(k3-1)//2)    │
│   BatchNorm1d(c3)                                   │
│   ReLU                                              │
│   Dropout(drop_conv3)                               │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Classification Head                                 │
│   AdaptiveMaxPool1d(1)  → shape: (batch, c3, 1)    │
│   Flatten              → shape: (batch, c3)        │
│   Dropout(drop_fc)                                  │
│   Linear(c3 → 1)       → shape: (batch, 1)         │
└─────────────────────────────────────────────────────┘
    │
    ▼
Output: logit (batch,)
```

**Design Rationale**:
- **Large first kernel** (`k1`): Captures motif-scale patterns (up to 41 bp)
- **Exponential activation**: Produces sharper, more interpretable filter responses
- **Aggressive early pooling** (`pool_w`): Reduces sequence to ~10-20 positions after conv1, forcing localized motif detection
- **BatchNorm after each conv**: Stabilizes training, especially with exponential activation
- **Progressive dropout**: Regularization increases through layers

---

### 2.3 Architecture Search Space

| Parameter | Range | Scale | Description |
|-----------|-------|-------|-------------|
| `k1` | 7-41 (odd) | linear | First conv kernel (motif width) |
| `k2`, `k3` | 3-11 (odd) | linear | Later conv kernels |
| `c1` | 16-128 | log | First conv channels |
| `c2` | 32-256 | log | Second conv channels |
| `c3` | 64-384 | log | Third conv channels |
| `pool_w` | 1-100 | log | Pooling width after conv1 |
| `act1` | {exp, relu, gelu, softplus} | categorical | First layer activation |
| `drop_conv1/2/3` | 0.0-0.4 | linear | Conv block dropout rates |
| `drop_fc` | 0.2-0.7 | linear | FC layer dropout |

---

### 2.4 Training Configuration

#### Fixed Settings

| Component | Configuration |
|-----------|---------------|
| **Loss function** | `BCEWithLogitsLoss` (binary cross-entropy with logits) |
| **Optimizer** | AdamW |
| **LR Scheduler** | `ReduceLROnPlateau(factor=0.5, patience=8, mode='min')` |
| **Mixed precision** | PyTorch AMP (`autocast` + `GradScaler`) |
| **Gradient clipping** | `clip_grad_norm_` with tunable max_norm |

#### Optimizer Search Space

| Parameter | Range | Scale |
|-----------|-------|-------|
| `lr` | 1e-4 – 3e-3 | log |
| `weight_decay` | 1e-8 – 1e-2 | log |
| `train_batch_size` | 256–1024 | step=128 |
| `grad_clip` | 0.5–10.0 | linear |

#### Early Stopping

| Phase | Max Epochs | Patience | Min Delta |
|-------|------------|----------|-----------|
| HPO trials | 40 | 15 | 1e-4 |
| Final refit | 50 | 15 | 1e-4 |

Early stopping monitors validation loss. Training halts if no improvement exceeding `min_delta` occurs for `patience` consecutive epochs.

---

### 2.5 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Saliency AUC** | Primary objective (see Section 1.6) |
| **Overlap** | IoU between top-60bp attribution window and ground truth motif |
| **Saliency SNR** | Signal-to-noise ratio of attributions |
| **PGD Success Rate** | Fraction of samples where PGD finds adversarial baseline |

**Effect Size Monitoring** (during training):
- **GC effect**: |Δ logit| when neutralizing background GC
- **Motif effect**: |Δ logit| when ablating motif region
- **Effect ratio**: motif_effect / gc_effect (higher = less reliance on confounder)

---

### 2.6 Phase 1: Standard HPO

```bash
sbatch --array=1-N slurm_scripts/run_singlearch_standard.sh
```

**Objective**: `0.9 x SaliencyAUC + 0.1 x ValAccuracy` (averaged over dataset panel)

**Dataset Evaluation Modes**:
- `--eval-mode panel`: 21 representative datasets (7 GC x 3 conservation)
- `--eval-mode all`: All 65 datasets
- `--eval-mode progressive`: Growing sample with trial number

---

### 2.7 Phase 2: Robust HPO

```bash
sbatch --array=0-64 slurm_scripts/run_singlearch_robust.sh
```

**Per-dataset studies**: Array index maps to (GC, conservation) pair

**Fixed from Phase 1**: Architecture (`--robust-arch-from-summary`)

**Tuned**:
- `regime`: {random_smoothing, gaussian_smoothing, hotflip, direct_hotflip}
- `schedule`: {on, off} (for HotFlip variants)
- Regime-specific epsilon/flip_fraction

---

### 2.8 CLI Reference (Synthetic)

| Argument | Description |
|----------|-------------|
| `--tune` | Run Optuna HPO |
| `--mode {standard,robust}` | HPO phase |
| `--study-name NAME` | Optuna study identifier |
| `--n-trials N` | Trials per worker |
| `--max-epochs N` | Max epochs per trial |
| `--eval-mode {panel,all,progressive}` | Dataset coverage |
| `--robust-arch-from-summary PATH` | Fix architecture from standard summary |
| `--dataset-gc VALUE` | GC for per-dataset robust study |
| `--dataset-cons VALUE` | Conservation for per-dataset robust study |

---

## Part 3: Phenotype (Real Genomic Data) Experiment

**Script**: `phenotype/code/optuna_phenotype.py`

### 3.1 Data Characteristics

| Property | Value |
|----------|-------|
| Input | Whole-genome FASTA files |
| Sequence length | 1,000,000 bp windows (randomly sampled) |
| Labels | Binary phenotype from metadata Excel |
| Default phenotype | "Spore formation" |
| Train/Val/Test | Pre-split directories |

**Data Loading**:
- Sequences cached in memory across Optuna trials
- Random window sampling during training
- Deterministic sampling for validation (reproducibility)

---

### 3.2 Architecture Search Space

| Parameter | Range | Scale | Notes |
|-----------|-------|-------|-------|
| `k1` | 51-401 (odd) | linear | Much larger for genomic scale |
| `c1` | 32-128 (or 64-256 extended) | log | |
| `stride1` | 5-50 | log | Aggressive downsampling |
| `pool1_k` | 10-128 | log | |
| `pool1_s` | 5-50 | log | |
| `n_blocks` | 1-3 | categorical | Depth of small-kernel blocks |
| `k_small` | 3-13 (odd) | linear | |
| `c2` | 64-192 (or 128-384) | log | |
| `c3` | 128-384 (or 256-768) | log | |
| `fc_hidden` | 128-512 (or 256-1024) | log | |
| `act1` | {exp} | fixed | Kept at `exp` for interpretability |
| `global_pool` | {avg, max} | categorical | |

**Extended Capacity Mode** (`--extended-capacity`):
- Wider channel ranges
- Prefer deeper stacks (2-3 blocks)
- Larger FC hidden dimensions

---

### 3.2.1 Optimizer Search Space

| Parameter | Range | Scale | Notes |
|-----------|-------|-------|-------|
| `lr` | 1e-5 – 3e-3 | log | Learning rate |
| `weight_decay` | 1e-8 – 1e-2 | log | AdamW regularization |
| `grad_clip` | 0.5 – 10.0 | linear | Gradient clipping threshold |
| `batch_size` | {4, 8, 12, 16, 20, 24, 32} | categorical | Training batch size |

---

### 3.2.2 Robust Search Space

| Parameter | Range | Scale | Notes |
|-----------|-------|-------|-------|
| `max_flip_fraction` | 1e-6 – 0.1 | log | Fraction of positions to flip per sequence |
| `schedule` | {cosine, linear, none} | categorical | Flip count ramp over epochs |

**Scheduling Modes**:
- `cosine`: Smooth ramp from 1 flip to max, slower start/end
- `linear`: Linear ramp from 1 flip to max over epochs
- `none`: Apply full max_flip_fraction from epoch 1

---

### 3.2.3 Selected Architecture (Spore Formation)

Best architecture from Phase 1 HPO (`sporo_full_std_v2_cont_exp_sporulation`):

| Parameter | Tuned Value | Derived Value |
|-----------|-------------|---------------|
| `k1_idx` | 134 | k1 = 269 (2×134+1) |
| `c1_cont` | 32.3 | c1 = 32 |
| `stride1_cont` | 29.3 | stride1 = 29 |
| `pool1_k_cont` | 67.1 | pool1_k = 67 |
| `pool1_s_cont` | 20.8 | pool1_s = 21 |
| `n_blocks` | 2 | 2 small-kernel blocks |
| `k_small_idx` | 1 | k_small = 3 |
| `c2_cont` | 68.2 | c2 = 64 |
| `c3_cont` | 235.0 | c3 = 224 |
| `use_pool2` | False | No pooling after block 2 |
| `use_pool3` | True | Pooling after block 3 |
| `drop1` | 0.004 | Minimal first-layer dropout |
| `drop2` | 0.325 | |
| `drop3` | 0.087 | |
| `drop_fc` | 0.402 | |
| `fc_hidden_cont` | 277.2 | fc_hidden = 288 |
| `act1` | exp | Exponential activation |
| `global_pool` | avg | Average pooling |
| `lr` | 8.2e-4 | |
| `weight_decay` | 1.4e-6 | |
| `grad_clip` | 1.32 | |
| `batch_size` | 32 | |

**Best Validation Balanced Accuracy**: 0.9817

---

### 3.2.4 Best Robustness Parameters (Spore Formation)

Best robust model from Phase 2 HPO (`50epochs_linear_spore_formation`):

| Parameter | Value |
|-----------|-------|
| `max_flip_fraction` (ε) | 7.83e-4 |
| `schedule` | linear |
| `max_epochs` | 50 |

**Evaluation Metrics**:

| Metric | Value |
|--------|-------|
| Saliency AUC | 0.5225 |
| Accuracy | 0.9784 |
| SaSNR | 0.1356 |
| SaSNR (expected) | 0.1269 |
| Δ SaSNR (median) | 0.0100 |
| Δ SaSNR CI | [0.0096, 0.0105] |
| Wilcoxon p (SaSNR > expected) | 1.1e-218 |
| n (samples) | 2500 |

**Objective**: `0.9 × SaAUC + 0.1 × Accuracy = 0.568`

**Interpretation**: The robust model achieves statistically significant improvement in saliency signal-to-noise ratio (SaSNR > expected with p < 1e-200), indicating attributions are more concentrated on ground-truth sporulation loci compared to random baseline expectations.

---

### 3.3 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Balanced Accuracy** | Primary objective (handles class imbalance) |
| **Saliency AUC** | IG-based attribution quality |
| **SaSNR** | Saliency signal-to-noise ratio |
| **SaSNR_expected** | Expected SaSNR under null (random baseline) |
| **Wilcoxon p-value** | Statistical significance of SaSNR > expected |
| **Delta  Median CI** | Confidence interval for (SaSNR - expected) difference |

**Objective for Robust Phase**: `0.9 x SaliencyAUC + 0.1 x Accuracy`

---

### 3.4 Phase 1: Standard HPO

```bash
sbatch slurm_scripts/run_optuna_phenotypes_std.sh
```

**Key Differences from Synthetic**:
- Single phenotype dataset (not a grid)
- Balanced accuracy as objective (not weighted saliency+accuracy)
- Class-weighted CrossEntropyLoss for imbalanced labels
- Best model saved per phenotype under `phenotype/bacillales_model/<phenotype_slug>/`

---

### 3.5 Phase 2: Robust HPO

```bash
sbatch slurm_scripts/run_optuna_phenotypes_robust.sh
```

**Enabled via**: `--robust-epsilon-only`

**Fixed from Phase 1**:
- Architecture loaded from `SporulationModel(summary_path=...)`
- Optimizer settings (LR, weight decay, batch size)

**Tuned**:
- `max_flip_fraction` in [1e-6, 0.1] (log scale, via `--eps-min`, `--eps-max`)

**Scheduling**: Cosine ramp of flip count over epochs (configurable via `--robust-schedule`)

**Evaluation Pipeline**:
- Calls external `phenotype/code/evaluation.py` subprocess
- Computes full saliency metrics on test set
- Saves best model to `phenotype/robust_model/<study_name>/`

---

### 3.6 CLI Reference (Phenotype)

| Argument | Description |
|----------|-------------|
| `--tune` | Run Optuna HPO |
| `--study-name NAME` | Study identifier |
| `--phenotype-col COL` | Metadata column for labels (default: "Spore formation") |
| `--metadata-xlsx PATH` | Excel file with phenotype annotations |
| `--seq-len N` | Window length (default: 1,000,000) |
| `--epoch-budget N` | Training samples per epoch |
| `--extended-capacity` | Use wider architecture search ranges |
| `--robust-epsilon-only` | Enable robust-phase epsilon tuning |
| `--eps-min`, `--eps-max` | Flip fraction search bounds |
| `--robust-schedule {cosine,linear,none}` | Flip scheduling strategy |
| `--summary-path PATH` | Standard-phase summary for architecture loading |

---

## Summary Comparison

| Aspect | Synthetic | Phenotype |
|--------|-----------|-----------|
| **Data** | Generated (1000 bp, planted motif) | Real genomes (1M bp windows) |
| **Dataset Grid** | 65 (GC x conservation) | Single phenotype |
| **Confounder** | GC content shortcut | Unknown biological confounders |
| **Primary Metric** | Saliency AUC | Balanced Accuracy |
| **Ground Truth** | Exact motif positions | Approximate (promoter regions) |
| **Phase 2 Scope** | Tune regime + epsilon per dataset | Tune epsilon only (fixed regime) |
| **Evaluation** | In-script | External subprocess |

---

## Part 4: Reproducibility

### 4.1 Software Environment

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.12.9 | Runtime |
| PyTorch | 2.3.1+cu121 | Deep learning framework |
| CUDA | 12.1 | GPU acceleration |
| NumPy | 1.26.4 | Numerical operations |
| Optuna | 4.4.0 | Hyperparameter optimization |
| Captum | 0.8.0 | Attribution methods (Integrated Gradients) |
| Matplotlib | 3.9.2 | Visualization |
| SciPy | 1.14.1 | Statistical functions |
| MySQL Connector | 9.1.0 | Optuna distributed storage |
| Logomaker | 0.8 | Sequence logo visualization |
| Seaborn | 0.13.2 | Statistical plots |

---

### 4.2 Data Split (Synthetic Experiments)

| Split | Fraction | Count | Purpose |
|-------|----------|-------|---------|
| Train | 70% | 7,000 | Model training |
| Validation | 15% | 1,500 | Early stopping, LR scheduling |
| Test | 15% | 1,500 | Final evaluation |

**Splitting mechanism**: `torch.utils.data.random_split` with seeded generator:

```python
generator = torch.Generator().manual_seed(0)  # Fixed seed
train_ds, val_ds, test_ds = random_split(
    full_dataset,
    [7000, 1500, 1500],
    generator=generator
)
```

This ensures identical train/val/test partitions across all runs for the same (GC, conservation) dataset.

---

### 4.3 Dataset Caching

Generated datasets are cached to `.npz` files to avoid recomputation:

```
synthetic/data/gc{gc:.3f}_cons{cons:.2f}.npz
```

Each cache file contains:
- `X`: One-hot encoded sequences, shape `(10000, 4, 1000)`, dtype `float32`
- `y`: Binary labels, shape `(10000,)`, dtype `int64`
- `masks`: Ground truth motif masks, shape `(10000, 1000)`, dtype `bool`

**Cache invalidation**: Delete `.npz` files to regenerate with different parameters.

---

### 4.4 Random Seed Handling

| Component | Seed Strategy |
|-----------|---------------|
| Master motif | Fixed seed 42 per dataset |
| Dataset split | Fixed seed 0 (generator) |
| HPO trials | Trial index as seed |
| Refit runs | Seed 0-4 for 5-seed average |

**Per-trial seeding** (in HPO):
```python
seed = trial.number
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
```

---

### 4.5 Numerical Reproducibility Notes

1. **cuDNN non-determinism**: By default, cuDNN may use non-deterministic algorithms for performance. For exact reproducibility:
   ```python
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```
   (Not enforced by default due to ~10-20% performance cost)

2. **Mixed precision**: AMP introduces minor floating-point variations across runs. Results are reproducible within numerical tolerance (~1e-4).

3. **Multi-GPU**: Current implementation is single-GPU. Multi-GPU training may introduce additional non-determinism from parallel reduction operations.
