# Experiment Design, HPO, and Implementation Details (Updated 2025-08-11)

This document summarizes the experiment design and the new Optuna-based hyperparameter optimization (HPO) path in `toy_slurm.py`.

## Recent Changes (2025-08-11)

- Added Optuna HPO with multi-objective tuning for Validation Accuracy and Saliency AUC (NSGA-II), with optional scalar objective for TPE.
- Pruning via validation loss across regimes; early stopping behavior unchanged.
- Post-hoc accuracy constraint: filter completed trials with Acc ≥ 0.90 and select by highest Saliency AUC.
- Parallel/distributed HPO via shared storage (SQLite/PostgreSQL/MySQL). n_jobs=1 per worker; scale via SLURM arrays.
- CLI flags: `--tune`, `--study-name`, `--storage`, `--sampler`, `--pruner`, `--n-trials`, `--max-epochs`, `--num-seeds-per-trial`, `--mode`, `--save-trial-artifacts`, `--artifacts-dir`, `--reval-trial`, `--reval-seeds`, `--smoke-test`.
- Model `TinyCNN` parameterized: kernel sizes `k1,k2,k3` (odd), channels `c1,c2,c3`, pool window `pool_w`, dropouts `drop_conv1..3`, `drop_fc`, and first-block activation `act1` in {exp, softplus, gelu, relu}. Forward API unchanged.
- Results exported to `optuna_results/{study}/trials.csv` and `summary.json` (counts, Pareto, Acc≥0.90 top trials).
- Re-evaluation utility to retrain a chosen trial across K seeds and report mean±SE for Accuracy and Saliency AUC.

### SLURM HPO Examples

Single-worker smoke test (SQLite):

```bash
export OPTUNA_STORAGE="sqlite:////scratch/$USER/optuna.db"
python toy_slurm.py --tune --study-name tinycnn_dev --n-trials 10 --mode standard --output_dir ./hpo_out
```

20-way parallel with PostgreSQL storage:

```bash
export OPTUNA_STORAGE="postgresql+psycopg2://USER:PASS@DBHOST:5432/optuna"
sbatch --array=1-20 --gres=gpu:1 -p gpu --mem=20G -t 2:00:00 \
  --wrap='RUN_HPO=1 STUDY_NAME=tinycnn_gc055 N_TRIALS=400 HPO_MODE=standard \
          python toy_slurm.py --tune --study-name tinycnn_gc055 --n-trials 400 --mode standard --output_dir ./hpo_out'
```

## High-level overview of the experimental design

This script implements a controlled synthetic experiment to evaluate the effect of adversarial training on model interpretability in the presence of a feature confounder. The key components are:

#### a. Synthetic Data Generation:
- **Task**: Binary classification of 1 kbp DNA sequences.
- **Negatives**: Random background with a GC content of 50%.
- **Positives**: A conserved 60-bp "causal motif" is embedded into a background with a higher GC content.
- **Control**: The strength of the confounder is controlled by the `gc_pos` parameter, and the strength of the true signal is controlled by the `conservation` parameter.

#### b. Quantifying Signal and Confounder Strength:
To provide a rigorous, *a priori* understanding of the experimental conditions, we use two standardized metrics:

1.  **Confounder Strength (Optimal Classifier Accuracy)**: We measure the strength of the GC-content confounder by calculating the theoretical maximum accuracy that a simple classifier could achieve by only using the GC% feature. This is derived from the Cohen's d statistic, which measures the separation between the positive (`μ_pos`, `σ_pos`) and negative (`μ_neg`, `σ_neg`) GC-content distributions.
    $$ d = \frac{|\mu_{pos} - \mu_{neg}|}{\sqrt{(\sigma_{pos}^2 + \sigma_{neg}^2)/2}} $$
    This is then converted to accuracy via the normal cumulative distribution function `Φ`:
    $$ \text{Accuracy} = \Phi(d/2) $$
    An accuracy of 50% indicates no confounding, while an accuracy of 100% indicates a perfect, deterministic confounder.

2.  **Signal Clarity (Information Content)**: We measure the clarity or learnability of the causal motif using Information Content (IC), a standard metric from bioinformatics. It is the Kullback-Leibler (KL) divergence from the background base distribution (`Q`, uniform) to the signal's base distribution (`P`, determined by `conservation`). The per-base IC is calculated as:
    $$ IC(c) = \sum_{b \in \{A,C,G,T\}} P(b|c) \cdot \log_2\frac{P(b|c)}{Q(b)} $$
    A higher IC (in bits) means the motif pattern is less noisy and more distinct from the random background, making it easier for a model to learn.

#### b. Adversarial Attack (HotFlip-style):
- Adversarially-trained models are generated using an iterative, gradient-based attack inspired by HotFlip.
- For each sequence in a batch, the process is as follows:
    1. The gradient of the loss is computed with respect to the one-hot encoded input sequence.
    2. This gradient is used to calculate a saliency score for flipping each nucleotide at each position to one of the other three bases.
    3. The single best flip (i.e., the position and new base that most increases the loss) is identified.
    4. The sequence is modified with this single flip.
    5. This process is repeated for *k* iterations, where *k* is the number of nucleotides to flip (k = epsilon * sequence length). This iterative design creates progressively more challenging adversarial examples.
- The model is then trained on this final, "attacked" batch.
   
#### c. Interpretability Metrics:
- We use Integrated Gradients with a two-stage baseline (PGD-based attack, falling back to compositional) to generate attribution maps. The quality of these maps and model robustness are quantified using four metrics:
    1. **Windowed IoU (wIoU)**: Measures how well the attribution map *locates* the true causal motif. It is the standard Intersection-over-Union between the true motif's 60-bp mask and the predicted 60-bp mask (defined as the window with the highest total attribution score). A score of 1 indicates a perfect match.
    2. **SaliencyAUC**: Measures the *purity* of attributions. It is the probability that a randomly chosen position inside the motif has a higher attribution score than a randomly chosen position outside. A score of 1 means all attributions are correctly concentrated within the motif.
       - **Metric Refinement (Effective Root-Skill)**: To better capture the scientific goal of interpretability, we transform the raw SaliencyAUC (`p`) into a more robust metric. We first calculate `Root-Skill`, defined as `Φ = sign(S) · √|S|` where `S = 2p-1`. This concave transformation heavily rewards improvements that escape randomness (e.g., AUC 0.5 → 0.7).
       - To ensure that only meaningful improvements (i.e., gains in the positive predictive range) are rewarded, we define an **Effective Root-Skill** as `max(0, Φ)`. The final reported metric, **`ΔEffectiveRootSkill`**, is the difference between the effective skills of the robust and standard models. This correctly assigns an improvement score of 0 for changes like AUC 0.4 → 0.5, as there is no gain in the positive skill domain. For scientific rigor, we also report the linear `ΔSkill` alongside `ΔEffectiveRootSkill`.
    3. **SaliencySNR (Signal-to-Noise Ratio)**: Measures the *cleanness* of attributions, defined as an R-squared-like metric. It is the fraction of attribution "energy" (sum of squared scores) within the true causal motif relative to the total energy across the entire sequence. Calculated as `sum(inside_scores^2) / sum(all_scores^2)`, it ranges from 0 (no signal in motif) to 1 (all signal in motif).
    4. **PGD Success Rate**: Measures model robustness. It is the fraction of times a 20-step PGD attack successfully finds an adversarial example within a small epsilon-ball that flips the model's prediction from positive to negative. A lower score indicates a more robust model.

#### d. Model Architecture (`TinyCNN`):
- Three Conv1d blocks with batch norm, configurable activation for block 1 (exp/softplus/gelu/relu), localist max-pooling with tunable window `pool_w`, and a final FC layer. Channels and kernel sizes are tunable; padding preserves length before pooling.

---

# Performance & Optimization

The evaluation pipeline in `toy_slurm.py` has been heavily optimized to reduce runtime from ~5-10 minutes per model to ~30-60 seconds, enabling faster and more extensive experimentation.

### Key Optimizations:
- **Batched PGD Attacks**: `find_adversarial_baseline_pgd_batch()` processes samples in parallel, providing a >10x speedup for the primary evaluation bottleneck.
- **PGD Result Caching**: A model fingerprint-based cache eliminates redundant PGD computations for models within the same experiment seed, saving significant time.
- **Batched Integrated Gradients**: IG attributions are now computed in batches, yielding a >5x speedup.
- **Efficient DataLoading**: Utilizes `pin_memory`, `persistent_workers`, and `prefetch_factor` for a non-blocking data pipeline.
- **Torch Compile**: The script attempts to use `torch.compile()` for a ~10-20% speedup on model forward/backward passes, with clear logging on success or failure.

### GPU Utilization Optimizations (2025-01):
To address low GPU utilization (9% → 60-90% expected), the following optimizations were added:

- **Vectorized HotFlip** (`generate_hotflip_examples_optimized`): Eliminates sequential loops and CPU-GPU synchronization. Uses batched tensor operations throughout the iterative attack process.
- **Optimized Direct HotFlip** (`generate_direct_hotflip_examples_optimized`): Fully vectorized implementation that applies all k flips using advanced indexing instead of loops.
- **GPU-Optimized PGD** (`find_adversarial_baseline_pgd_batch_optimized`): Minimizes synchronization by deferring all `.item()` calls to the end. Tracks success states on GPU using masks.
- **Increased Batch Sizes**: Evaluation batch size increased to 1024, PGD batch size to 50, IG batch size to 25 for better GPU saturation.
- **GPU Monitoring**: Added `log_gpu_stats()` to track utilization and memory usage during training, helping identify remaining bottlenecks.
- **Optional Prefetch DataLoader**: `GPUPrefetchDataLoader` class provides asynchronous data transfers using CUDA streams (can be enabled if data loading becomes a bottleneck).

These optimizations maintain identical algorithmic behavior while dramatically improving GPU efficiency, especially for the iterative HotFlip attack which was the primary bottleneck. 