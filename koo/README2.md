# Implementation Notes: Koo & Ploenzke (2021) Task 3

This document tracks the discrepancies between the published paper, the original repository code, and our current implementation.

For common concepts (Integrated Gradients, gradient correction, saliency metrics), see `synthetic/docu/WORKFLOW.md`.

---

## Reference

**Koo & Ploenzke (2021)** — *Improving Representations of Genomic Sequence Motifs in Convolutional Networks with Exponential Activations*

Task 3 focuses on comparing CNN architectures (CNN-local vs CNN-dist) with different activation functions (ReLU vs Exponential) for synthetic sequence classification.

---

## 1. Published Description (Paper)

### Architectures

| Model | Architecture |
|-------|--------------|
| **CNN-local** | conv(24, k=19) → maxpool(50) → dense(96) |
| **CNN-dist** | conv(24, k=7) → conv(32, k=9)+pool(3) → conv(48, k=6)+pool(4) → conv(64, k=4)+pool(3) → dense(96) |

### Training Protocol

- **Epochs**: 100
- **Learning rate**: starts at 0.001
- **LR decay**: factor 0.3 if monitored metric doesn't improve for 5 epochs
- **Model selection**: report test metrics using best-validation model parameters
- **Trials**: 10 runs with different random initializations

---

## 2. Original Repository Code (Observed)

### Architectures

| Model | Repo Implementation | Difference from Paper |
|-------|---------------------|----------------------|
| **CNN-local** | conv(24, k=19) → pool(50) → **conv(48, k=3) → pool(2)** → dense(96) | Extra conv layer added |
| **CNN-dist** | conv(24, **k=19**) → conv(32, k=7)+pool(**4**) → conv(48, **k=7**)+pool(**4**) → conv(64, **k=3**)+pool(3) → dense(96) | Different kernel sizes and pool sizes |

### Key Mismatches (Paper vs Original Repo)

| Aspect | Paper | Repo Code |
|--------|-------|-----------|
| CNN-local architecture | 1 conv layer | 2 conv layers (adds conv(48,k=3)+pool(2)) |
| CNN-dist first kernel | k=7 | k=19 |
| CNN-dist kernel schedule | 7/9/6/4 | 19/7/7/3 |
| CNN-dist pool schedule | 3/4/3 | 4/4/3 |
| LR decay factor | 0.3 | 0.2 |
| Best-validation checkpoint | Yes | No (saves final epoch) |

---

## 3. Our Implementation

PyTorch re-implementation with robustness training added. We implement **three** architectures:

### Architectures (`koo/code/model_zoo/torch_models.py`)

#### CNN-Local (paper & repo match)
```
Input (batch, seq_len, 4) → permute to (batch, 4, seq_len)
Conv1d(4→24, k=19, pad=9) + BN + act + Dropout(0.1) + MaxPool(50)
Conv1d(24→48, k=3, pad=1) + BN + ReLU + Dropout(0.2) + MaxPool(2)
Flatten → Linear(96→96) + BN + ReLU + Dropout(0.5)
Linear(96→1) + Sigmoid
```

#### CNN-Dist (paper version, k=7 first layer)
```
Input (batch, seq_len, 4) → permute to (batch, 4, seq_len)
Conv1d(4→24, k=7, pad=3) + BN + act + Dropout(0.1)
Conv1d(24→32, k=9, pad=4) + BN + ReLU + Dropout(0.2) + MaxPool(3)
Conv1d(32→48, k=6, pad=0) + BN + ReLU + Dropout(0.3) + MaxPool(4)
Conv1d(48→64, k=4, pad=0) + BN + ReLU + Dropout(0.4) + MaxPool(3)
Flatten → Linear(256→96) + BN + ReLU + Dropout(0.5)
Linear(96→1) + Sigmoid
```

#### CNN-Local-Deep (repo's mislabeled "cnn-dist", k=19 first layer)
```
Input (batch, seq_len, 4) → permute to (batch, 4, seq_len)
Conv1d(4→24, k=19, pad=9) + BN + act + Dropout(0.1)
Conv1d(24→32, k=7, pad=3) + BN + ReLU + Dropout(0.2) + MaxPool(4)
Conv1d(32→48, k=7, pad=0) + BN + ReLU + Dropout(0.3) + MaxPool(4)
Conv1d(48→64, k=3, pad=0) + BN + ReLU + Dropout(0.4) + MaxPool(3)
Flatten → Linear(192→96) + BN + ReLU + Dropout(0.5)
Linear(96→1) + Sigmoid
```

**Note**: The original repo's "cnn-dist" used k=19 in the first layer, making it functionally a deeper CNN-Local (captures local motifs). We renamed it to CNN-Local-Deep and added the paper's true CNN-Dist (k=7 first layer, captures distributed motifs).

- First layer activation configurable: `relu` or `exponential`
- Progressive dropout: 0.1 → 0.2 → 0.3 → 0.4 → 0.5

### Training Protocol

| Parameter | Standard | Robust |
|-----------|----------|--------|
| Optimizer | Adam (lr=0.001, wd=1e-6) | Same |
| Loss | BCELoss | BCELoss |
| Batch size | 100 | 100 |
| Max epochs | 100 | 100 |
| LR scheduler | ReduceLROnPlateau(0.2) | Same |
| LR patience | 5 | 8 |
| Early stopping patience | 20 | 30 |
| Trials | 10 | 10 |

**Model Selection**: Saves final epoch weights (NOT best-validation checkpoint, matching repo).

---

## 4. Dataset

Pre-generated HDF5 file: `koo/data/synthetic_code_dataset.h5`

| Key | Shape | Description |
|-----|-------|-------------|
| `X_train/valid/test` | (N, 4, 200) | One-hot sequences |
| `Y_train/valid/test` | (N, 1) | Binary labels |
| `model_train/valid/test` | (N, 4, 200) | Ground truth PWMs |

**Sequence length**: 200 bp (vs 1000 bp in our synthetic experiment)

### Ground Truth Definition

Each positive sample has a Position Weight Matrix (PWM). Motif positions defined by information content:

```
I[j] = log2(4) + Σ_i PWM[i,j] * log2(PWM[i,j] + 1e-10)
label[j] = 1 if I[j] > 0.01 else 0
```

---

## 5. Robustness Training

### Direct HotFlip

Single gradient pass, apply top-k flips. See WORKFLOW.md Section 1.3 for general concept.

**Koo-specific parameters**:
- Flip fractions: 0.01, 0.05, 0.1, 0.15, 0.2
- At seq_len=200: k_flips = 2, 10, 20, 30, 40
- No warmup (applied from epoch 1)
- No scheduling (fixed flip fraction throughout training)

### Experimental Grid

| Model | Activation | Training | Flip Fractions |
|-------|------------|----------|----------------|
| cnn-local | relu | standard | - |
| cnn-local | relu | robust | 0.01, 0.05, 0.1, 0.15, 0.2 |
| cnn-local | exponential | standard | - |
| cnn-local | exponential | robust | 0.01, 0.05, 0.1, 0.15, 0.2 |
| cnn-dist | relu | standard | - |
| cnn-dist | relu | robust | 0.01, 0.05, 0.1, 0.15, 0.2 |
| cnn-dist | exponential | standard | - |
| cnn-dist | exponential | robust | 0.01, 0.05, 0.1, 0.15, 0.2 |

**Total**: 2 × 2 × 6 × 10 = **240 models**

---

## 6. Attribution Computation

Uses Integrated Gradients with gradient correction (see WORKFLOW.md Section 1.6).

### Baseline Methods

| Method | Description | Used For |
|--------|-------------|----------|
| **PGD** | Find adversarial input that flips prediction | All models |
| **Shuffle** | Average of 10 per-channel shuffled versions | Standard only |

### PGD Parameters

- `epsilon = 0.1` (L∞ bound)
- `step_size = 0.01`
- `num_iter = 20`
- Fallback: zero baseline if PGD fails

### Evaluation Set

500 positive test samples with ground truth PWMs.

### Output

```
koo/results/task3_robust_5/scores/
├── cnn-local_relu_standard.pickle          # PGD baseline
├── cnn-local_relu_standard_shuffle.pickle  # Shuffle baseline
├── cnn-local_relu_robust_0.01.pickle       # Robust models (PGD only)
...
```

Shape: `(10, 500, 4, 200)` = (trials, samples, nucleotides, positions)

---

## 7. Summary

| Aspect | Paper | Original Repo | Our Implementation |
|--------|-------|---------------|-------------------|
| Framework | TensorFlow | TensorFlow | **PyTorch** |
| CNN-Local | 1 conv | 2 conv | 2 conv (repo) |
| CNN-Dist | k=7/9/6/4 | k=19/7/7/3 (mislabeled) | **Both**: paper (k=7) + repo (k=19 as CNN-Local-Deep) |
| LR decay | 0.3 | 0.2 | 0.2 (repo) |
| Best-val checkpoint | Yes | No | No (repo) |
| Robustness | N/A | N/A | **Direct HotFlip** |
| IG baseline | N/A | Zero/shuffle | **PGD adversarial** |
| Gradient correction | N/A | N/A | **Yes** |

---

## 8. File Structure

```
koo/
├── code/
│   ├── task3_step1_train_model.py          # Training
│   ├── task3_step2_attribution_scores.py   # IG computation
│   ├── task3_step3_plot_attr_score_comparisons.py
│   ├── task3_step4_plot_attr_logo_comparisons.py
│   ├── helper.py                           # Data loading
│   ├── robust_helper.py                    # HotFlip, PGD, IG
│   └── model_zoo/torch_models.py           # Architectures
├── data/synthetic_code_dataset.h5
└── results/task3_robust_5/
```

---

## 9. Running the Pipeline

```bash
# Step 1: Train models (GPU, ~2 days)
sbatch slurm_scripts/run_task3_step1_training.sh

# Step 2: Compute attributions (GPU, ~1 day)
sbatch slurm_scripts/run_task3_step2_attribution.sh

# Step 3-4: Evaluate, plot metrics, and generate logos (CPU, ~1 hour)
sbatch slurm_scripts/run_task3_step3_plots.sh
```
