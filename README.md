# GenomeInterpretation

Robustness-Driven Feature Attribution in Genomic Deep Learning with Integrated Gradients

## Overview

This project develops tools and modeling standards to enhance the interpretability of phenotype-genome models, with the goal of automating biomarker discovery from genome sequences.

We train CNN models on one-hot encoded DNA sequences and apply gradient-based interpretation methods (e.g., Integrated Gradients). This repository contains experiments to validate and improve these interpretation approaches in biological contexts.

## Project Structure

```
GenomeInterpretation/
├── synthetic/          # Synthetic data experiments
├── koo/                # Explainability/interpretation research
├── sporulation/        # Sporulation phenotype analysis
├── phenotype/          # General phenotype prediction
├── slurm_scripts/      # SLURM job submission scripts
└── toy_single_arch.py  # Main synthetic experiment entry point
```

## Components

### 1. Synthetic Data Experiments

Controlled experiments with synthetic sequences to validate interpretation methods.

| Script | Description |
|--------|-------------|
| `toy_single_arch.py` | Main entry point for synthetic experiments |
| `synthetic/code/gc_baseline_sweep.py` | GC-content baseline sanity check |

**Documentation:** `synthetic/docu/`

### 2. Explainability Research (koo/)

Task-based pipeline for model interpretation and robustness analysis.

| Script | Description |
|--------|-------------|
| `koo/code/task3_step1.py` | Step 1 of interpretation pipeline |
| `koo/code/task3_step2.py` | Step 2 of interpretation pipeline |
| `koo/code/task3_step3.py` | Step 3 of interpretation pipeline |
| `koo/code/task3_step4.py` | Step 4 of interpretation pipeline |
| `koo/code/gc_baseline_classifier.py` | GC-content baseline classifier |

**Documentation:** `koo/README2.md`

### 3. Sporulation Analysis

Sporulation phenotype prediction and multi-model analysis.

| Script | Description |
|--------|-------------|
| `sporulation/code/bugphyzz_visual.py` | BugPhyzz data visualization |
| `sporulation/code/codon_msa.py` | Codon-level MSA analysis |
| `sporulation/code/figure_four_panel.py` | Multi-panel figure generation |
| `sporulation/code/rashomon_*.py` | Rashomon set analysis (multiple variants) |
| `sporulation/code/run_streme.py` | STREME motif discovery |

**Documentation:** `sporulation/docu/`

### 4. Phenotype Prediction (phenotype/)

General-purpose phenotype prediction framework. All scripts in `phenotype/code/` are actively used.

Key scripts include training, evaluation, GFF processing, and genome feature analysis.

## Environment

```bash
conda env create -f environment.yml
conda activate genome
```

Requires: Python 3.12, PyTorch 2.3, CUDA 12.1

## Contact

yichen.han AT campus.lmu.de
