"""Shared utilities for phenotype-aware processing and evaluation.

Provides:
- canonical gene name normalization
- phenotype name helpers (slug, directory name)
- metadata loading helpers
- ground-truth gene set derivation from clustered multiplicity outputs
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Phenotype columns as recorded in the metadata Excel (microbe.cards table S1.xlsx)
PHENOTYPE_COLUMNS: List[str] = [
    "Motility",
    "Gram staining",
    "Aerophilicity",
    "Extreme environment tolerance",
    "Biofilm formation",
    "Animal pathogenicity",
    "Biosafety level",
    "Health association",
    "Host association",
    "Plant pathogenicity",
    "Spore formation",
    "Hemolysis",
    "Cell shape",
]

# Ground-truth inference defaults
GROUND_TRUTH_FREQ_THRESHOLD = 0.60  # >60% of runs in both LASSO and RF
DEFAULT_CLUSTER_MIN_PREV = 0.02
DEFAULT_CLUSTER_METRIC = "ochiai"
DEFAULT_CLUSTER_THRESHOLD = 0.7  # similarity threshold used when building clusters (distance = 1 - threshold)
CLUSTER_BASE_PATH = Path("/vol/projects/BIFO/genomenet/yichen")

# File naming template mirrors sporulation/code/multiplicity_h1.py
CLUSTER_FILENAME_TEMPLATE = (
    "gene_clusters_all_samples_minprev{min_prev:.3f}_metric-{metric}_mode-abs_link-average_thr-{distance:.2f}.npz"
)
CLUSTER_FILENAME_FALLBACK = (
    "gene_clusters_all_samples_minprev{min_prev:.3f}_mode-abs_link-average_thr-{distance:.2f}.npz"
)


# ---------------------------------------------------------------------------
# Phenotype helpers
# ---------------------------------------------------------------------------

def phenotype_to_slug(name: str) -> str:
    """Return a lowercase slug suitable for column names (e.g., 'Spore formation' -> 'spore_formation')."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower())
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_")


def phenotype_to_dirname(name: str) -> str:
    """Return the canonical directory name used in results folders (replace spaces with underscores)."""
    return name.strip().replace(" ", "_")


# ---------------------------------------------------------------------------
# Gene name normalization
# ---------------------------------------------------------------------------

def canonical_gene_name(name: Optional[str]) -> Optional[str]:
    """Normalize a gene name for reliable matching across datasets."""
    if name is None:
        return None
    s = str(name).strip()
    if not s or s == "." or s.lower() == "none":
        return None
    s = s.lower()
    s = re.sub(r"_[0-9]+$", "", s)  # drop trailing locus tag suffixes
    s = re.sub(r"^sig([a-z])$", r"sigma_\1", s)  # unify sigma naming
    if not re.match(r"^[a-z0-9_\-]+$", s):
        return None
    return s


def extract_canonical_gene_tokens(value: object) -> Set[str]:
    """Return canonical gene-like tokens parsed from a free-form value."""
    tokens: Set[str] = set()
    if value is None:
        return tokens
    s = str(value)
    if not s:
        return tokens
    # Replace common separators with spaces, then split
    cleaned = re.sub(r"[;,/|]", " ", s)
    parts = re.split(r"\s+", cleaned)
    for part in parts:
        canon = canonical_gene_name(part)
        if canon:
            tokens.add(canon)
    return tokens


# ---------------------------------------------------------------------------
# Metadata value normalization
# ---------------------------------------------------------------------------

TRUE_VALUES = {"true", "t", "yes", "y", "1", "1.0"}
FALSE_VALUES = {"false", "f", "no", "n", "0", "0.0"}

_FASTA_SUFFIXES = (".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz")


def normalize_metadata_value(val: object) -> Optional[bool]:
    """Convert heterogeneous truthy/falsey metadata entries to booleans."""
    if pd.isna(val):
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    s = str(val).strip().lower()
    if s in TRUE_VALUES:
        return True
    if s in FALSE_VALUES:
        return False
    return None


def metadata_series_to_bool(series: pd.Series) -> pd.Series:
    """Convert a pandas Series of metadata phenotype values to nullable boolean."""
    return series.apply(normalize_metadata_value).astype("boolean")


# ---------------------------------------------------------------------------
# Phenotype label mapping helpers (multi-class aware)
# ---------------------------------------------------------------------------


def normalize_label_value(val: object) -> str:
    """Normalize phenotype label to a lowercase string with canonical booleans."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    low = s.lower()
    if low in TRUE_VALUES:
        return "true"
    if low in FALSE_VALUES:
        return "false"
    return low


def _maybe_reduce_three_class_to_binary(
    df: pd.DataFrame,
    classes: List[str],
    phenotype: str,
    key_col: str,
    train_dirs: Optional[Iterable[str]],
    min_training_genomes: int,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop a minority class when a 3-way phenotype lacks training genomes."""
    if len(classes) != 3:
        return df, classes

    candidate_dirs: List[Path] = []
    if train_dirs:
        for raw in train_dirs:
            if not raw:
                continue
            path = Path(raw)
            if path.exists():
                candidate_dirs.append(path)
    else:
        default_dir = DATA_ROOT / "train"
        if default_dir.exists():
            candidate_dirs.append(default_dir)

    if not candidate_dirs:
        return df, classes

    label_lookup = df.set_index(key_col)["_val_norm"].to_dict()
    counts = {cls: 0 for cls in classes}
    found_any = False

    for directory in candidate_dirs:
        try:
            entries = list(directory.iterdir())
        except Exception:
            continue
        for entry in entries:
            if not entry.is_file():
                continue
            name = entry.name.strip().lower()
            if not name.endswith(_FASTA_SUFFIXES):
                continue
            label = label_lookup.get(name)
            if label is None:
                continue
            counts[label] += 1
            found_any = True

    if not found_any:
        return df, classes

    threshold = int(min_training_genomes)
    below = [cls for cls, cnt in counts.items() if cnt < threshold]
    if len(below) != 1:
        return df, classes

    drop_class = below[0]
    kept_classes = [cls for cls in classes if cls != drop_class]
    if len(kept_classes) < 2:
        return df, classes

    print(
        f"[phenotype-utils] Reducing phenotype '{phenotype}' to binary by dropping class '{drop_class}' "
        f"(training genomes={counts[drop_class]} < {threshold}).",
        flush=True,
    )
    reduced_df = df[df["_val_norm"] != drop_class].copy()
    return reduced_df, sorted(kept_classes)


def build_labels_map_and_classes(
    metadata_df: pd.DataFrame,
    phenotype_col: str,
    file_col: str = "Fasta file",
    train_dirs: Optional[Iterable[str]] = None,
    min_training_genomes: int = 25,
) -> Tuple[Dict[str, int], List[str]]:
    """Return (labels_map, classes) for a phenotype column.

    labels_map normalizes FASTA filenames to lowercase basenames and maps them to
    integer class ids (0..C-1). Classes are returned as a sorted list of normalized
    label strings (e.g., ['false', 'true'] or multiple categorical values).

    For phenotypes with exactly three categories, this will drop a minority class
    when the training directories provide fewer than ``min_training_genomes``
    labeled FASTA files for that category, reducing the task to binary.
    """
    if file_col not in metadata_df.columns:
        raise ValueError(f"Expected column '{file_col}' in metadata DataFrame")
    if phenotype_col not in metadata_df.columns:
        raise ValueError(f"Expected phenotype column '{phenotype_col}' in metadata DataFrame")

    df = metadata_df.copy()
    if "Fasta file_norm" not in df.columns and file_col == "Fasta file":
        df["Fasta file_norm"] = df[file_col].map(lambda x: Path(str(x)).name.strip().lower())
        key_col = "Fasta file_norm"
    else:
        key_col = file_col

    df["_val_norm"] = df[phenotype_col].map(normalize_label_value)
    df = df[[key_col, "_val_norm"]].dropna()
    df[key_col] = df[key_col].map(lambda x: str(x).strip().lower())
    df["_val_norm"] = df["_val_norm"].map(str)
    df = df[df["_val_norm"].str.len() > 0]
    df = df.drop_duplicates(subset=[key_col], keep="last")

    classes = sorted(df["_val_norm"].unique().tolist())
    df, classes = _maybe_reduce_three_class_to_binary(
        df,
        classes,
        phenotype_col,
        key_col,
        train_dirs,
        min_training_genomes,
    )
    class_to_id = {c: i for i, c in enumerate(classes)}

    labels_map = {
        key: class_to_id[val]
        for key, val in zip(df[key_col], df["_val_norm"])
        if val in class_to_id
    }
    return labels_map, classes


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def read_metadata_table(xlsx_path: Path) -> pd.DataFrame:
    """Load the metadata Excel table and attach normalized FASTA basenames."""
    df = pd.read_excel(xlsx_path)
    if "Fasta file" not in df.columns:
        raise ValueError("Expected column 'Fasta file' in metadata Excel sheet")

    def _norm_basename(val: object) -> str:
        s = str(val) if not pd.isna(val) else ""
        return Path(s).name.strip().lower()

    df["Fasta file_norm"] = df["Fasta file"].map(_norm_basename)
    return df


# ---------------------------------------------------------------------------
# Ground-truth gene sets from clustered multiplicity outputs
# ---------------------------------------------------------------------------

def _resolve_cluster_npz(min_prev: float, metric: str, cluster_threshold: float) -> Path:
    distance = max(0.0, min(1.0, 1.0 - float(cluster_threshold)))
    primary = CLUSTER_BASE_PATH / CLUSTER_FILENAME_TEMPLATE.format(
        min_prev=min_prev,
        metric=metric,
        distance=distance,
    )
    if primary.exists():
        return primary
    fallback = CLUSTER_BASE_PATH / CLUSTER_FILENAME_FALLBACK.format(
        min_prev=min_prev,
        distance=distance,
    )
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Cluster NPZ not found for min_prev={min_prev}, metric={metric}, threshold={cluster_threshold}. "
        f"Looked for {primary.name} and {fallback.name} under {CLUSTER_BASE_PATH}"
    )


@lru_cache(maxsize=None)
def _load_cluster_mapping(
    min_prev: float = DEFAULT_CLUSTER_MIN_PREV,
    metric: str = DEFAULT_CLUSTER_METRIC,
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> Tuple[Dict[str, int], int]:
    """Return (gene -> cluster_id, n_clusters) from the precomputed cluster NPZ."""
    npz_path = _resolve_cluster_npz(min_prev=min_prev, metric=metric, cluster_threshold=cluster_threshold)
    data = np.load(npz_path, allow_pickle=False)
    genes = data["genes"].astype(str)
    cluster_ids = data["cluster_ids"].astype(int)
    unique_ids = sorted(set(int(cid) for cid in cluster_ids))
    id_remap = {old: idx for idx, old in enumerate(unique_ids)}
    mapping = {str(g): id_remap[int(cid)] for g, cid in zip(genes, cluster_ids)}
    return mapping, len(unique_ids)


def _resolve_cache_dir(phenotype: str) -> Optional[Path]:
    dirname = phenotype_to_dirname(phenotype)
    candidates = [
        Path("sporulation/results/clustering") / dirname / ".cache",
        Path("sporulation/results/h1_multiplicity") / dirname / ".cache",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_selection_matrix(path: Path, key: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    if "sel_matrix" not in data or "genes" not in data:
        raise KeyError(f"Missing required arrays in {path.name}")
    genes = data["genes"].astype(str)
    sel = data["sel_matrix"].astype(bool)
    return genes, sel


def _compute_cluster_frequencies(sel: np.ndarray, clusters: np.ndarray, n_clusters: int) -> np.ndarray:
    if sel.size == 0:
        return np.zeros(n_clusters, dtype=float)
    n_runs = sel.shape[0]
    cluster_sel = np.zeros((n_runs, n_clusters), dtype=bool)
    for feat_idx, cluster in enumerate(clusters):
        cluster_sel[:, cluster] |= sel[:, feat_idx]
    return cluster_sel.mean(axis=0)


@lru_cache(maxsize=None)
def load_ground_truth_gene_set(
    phenotype: str,
    freq_threshold: float = GROUND_TRUTH_FREQ_THRESHOLD,
    min_prev: float = DEFAULT_CLUSTER_MIN_PREV,
    metric: str = DEFAULT_CLUSTER_METRIC,
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> Set[str]:
    """Return the set of canonical gene names forming the ground-truth mask for a phenotype."""
    cache_dir = _resolve_cache_dir(phenotype)
    if cache_dir is None:
        print(f"[GroundTruth] Warning: missing cache directory for phenotype '{phenotype}'.")
        return set()

    cpss_path = cache_dir / "cpss_cache.npz"
    rf_path = cache_dir / "rf_halves_cache.npz"
    if not cpss_path.exists() or not rf_path.exists():
        print(f"[GroundTruth] Warning: missing selection caches for phenotype '{phenotype}' under {cache_dir}.")
        return set()

    try:
        genes_lasso, sel_lasso = _load_selection_matrix(cpss_path, "sel_matrix")
        genes_rf, sel_rf = _load_selection_matrix(rf_path, "sel_matrix")
    except Exception as exc:
        print(f"[GroundTruth] Warning: failed to load selection matrices for '{phenotype}': {exc}")
        return set()

    # Ensure gene orders align between LASSO and RF caches; if not, align by name.
    if len(genes_lasso) != len(genes_rf) or not np.array_equal(genes_lasso, genes_rf):
        # Build alignment map from gene name to index
        index_map = {g: i for i, g in enumerate(genes_rf)}
        aligned_sel_rf = np.zeros_like(sel_lasso, dtype=bool)
        for feat_idx, gene in enumerate(genes_lasso):
            j = index_map.get(gene)
            if j is None:
                continue
            aligned_sel_rf[:, feat_idx] = sel_rf[:, j]
        sel_rf = aligned_sel_rf
        genes = genes_lasso
    else:
        genes = genes_lasso
    if sel_rf.shape[1] != len(genes):
        # Dimensions mismatch after alignment; safest is to truncate to common length
        min_p = min(sel_rf.shape[1], len(genes))
        sel_rf = sel_rf[:, :min_p]
        sel_lasso = sel_lasso[:, :min_p]
        genes = genes[:min_p]

    mapping, base_clusters = _load_cluster_mapping(min_prev=min_prev, metric=metric, cluster_threshold=cluster_threshold)
    clusters = np.empty(len(genes), dtype=int)
    next_cluster = base_clusters
    for i, gene in enumerate(genes):
        cid = mapping.get(gene)
        if cid is None:
            cid = next_cluster
            next_cluster += 1
        clusters[i] = cid
    n_clusters = int(next_cluster)

    freq_lasso = _compute_cluster_frequencies(sel_lasso, clusters, n_clusters)
    freq_rf = _compute_cluster_frequencies(sel_rf, clusters, n_clusters)
    if freq_lasso.size == 0 or freq_rf.size == 0:
        print(f"[GroundTruth] Warning: empty selection matrices for phenotype '{phenotype}'.")
        return set()

    mask = (freq_lasso > float(freq_threshold)) & (freq_rf > float(freq_threshold))
    if not mask.any():
        print(f"[GroundTruth] Warning: no clusters exceeded frequency threshold for '{phenotype}'.")
        return set()

    gt: Set[str] = set()
    for gene, cluster in zip(genes, clusters):
        if mask[cluster]:
            canon = canonical_gene_name(gene)
            if canon:
                gt.add(canon)
    return gt


def load_ground_truth_gene_sets(
    phenotypes: Iterable[str],
    freq_threshold: float = GROUND_TRUTH_FREQ_THRESHOLD,
    min_prev: float = DEFAULT_CLUSTER_MIN_PREV,
    metric: str = DEFAULT_CLUSTER_METRIC,
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> Dict[str, Set[str]]:
    """Load ground-truth gene sets for multiple phenotypes."""
    result: Dict[str, Set[str]] = {}
    for phenotype in phenotypes:
        result[phenotype] = load_ground_truth_gene_set(
            phenotype,
            freq_threshold=freq_threshold,
            min_prev=min_prev,
            metric=metric,
            cluster_threshold=cluster_threshold,
        )
    return result
# Primary data root (migrated from repo-local `sporulation/data`)
DATA_ROOT = Path("/vol/projects/BIFO/genomenet/yichen/phenotype/data")
