"""
Analyze phenotype-specific signal in processed GFFs (phenotype-agnostic).

Inputs: a directory of processed GFF tables (parquet/csv/feather/tsv) with columns
  seqid, start, end, strand, locus_tag, gene, Name, product, inference,
  protein_id, sources, gff_filename, fasta_filename, Phylum, Class, Order,
  Family, Genus, Species, and an optional phenotype metadata column. A
  ground-truth mask column is expected as gt_<phenotype_slug>, where
  phenotype_slug is derived from --phenotype (see phenotype_utils.phenotype_to_slug).

This script supports binary, categorical, and numeric phenotypes:
  - binary: computes contrast between positive and negative genomes (effect sizes and AUROC).
  - categorical (>2): summarizes metrics per category and reports variance explained.
  - numeric: reports Pearson/Spearman correlations with metrics.

Key analyses (per genome, where genome ≡ gff_filename):
  1) Fraction of masked loci among all loci (p_mask_loci).
  2) Coverage by bp: union length of mask intervals / genome length; also coding coverage.
  3) Window sampling: draw W windows of size S; compute mask bp fraction per window.
  4) Nearest-neighbor distances between masked locus midpoints (per contig), pooled.
  5) Phenotype association with the above metrics (contrast/categorical summary/correlation).
  6) Ground-truth gene/cluster enrichment stratified by phenotype.

Outputs (CSV unless stated otherwise):
  - per_genome_metrics.csv
  - window_samples.csv (W rows)
  - window_summary.csv (mean, sd, quantiles)
  - nnd_distances.csv (all NNDs; columns: gff_filename, seqid, distance_bp)
  - phenotype_contrast.csv (association summary; name kept for backward-compat)
  - ground_truth_gene_enrichment.csv (selected ground-truth genes)
  - ground_truth_cluster_enrichment.csv (ground-truth clusters)

Notes:
  * Assumes each row ≈ a coding locus; if both gene/CDS rows were retained, we can deduplicate
    to one row per (gff_filename, locus_tag) preferring records with protein_id (optional).
  * Genome length is estimated as sum over contigs of max(end) (typical for GFF-derived lengths).
  * Only standard libs + pandas/numpy are used (no SciPy/sklearn).
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:  # optional aesthetics
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None

try:
    from phenotype_utils import (
        phenotype_to_slug,
        metadata_series_to_bool,
        DATA_ROOT,
        GROUND_TRUTH_FREQ_THRESHOLD,
        DEFAULT_CLUSTER_MIN_PREV,
        DEFAULT_CLUSTER_METRIC,
        DEFAULT_CLUSTER_THRESHOLD,
        _load_cluster_mapping,
        _load_selection_matrix,
        _compute_cluster_frequencies,
        _resolve_cache_dir,
    )
except ImportError:  # pragma: no cover - package-style import fallback
    from .phenotype_utils import (  # type: ignore
        phenotype_to_slug,
        metadata_series_to_bool,
        DATA_ROOT,
        GROUND_TRUTH_FREQ_THRESHOLD,
        DEFAULT_CLUSTER_MIN_PREV,
        DEFAULT_CLUSTER_METRIC,
        DEFAULT_CLUSTER_THRESHOLD,
        _load_cluster_mapping,
        _load_selection_matrix,
        _compute_cluster_frequencies,
        _resolve_cache_dir,
    )

# --------------------------- I/O utilities ---------------------------

def _get_parquet_columns(path: str) -> Optional[List[str]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
        pf = pq.ParquetFile(path)
        return [f.name for f in pf.schema_arrow]
    except Exception:
        return None


def read_all_tables(input_dir: str, required_columns: Optional[List[str]] = None, progress_every: int = 500) -> pd.DataFrame:
    """Read all parquet/csv tables from a directory (non-recursive), selecting only
    required columns when possible for speed/memory. Returns a single concatenated DataFrame.
    """
    paths: List[str] = []
    for ext in ("*.parquet", "*.pq", "*.feather", "*.csv", "*.tsv"):
        paths.extend(glob.glob(os.path.join(input_dir, ext)))
    if not paths:
        raise FileNotFoundError(f"No data files found in {input_dir}")

    paths = sorted(paths)
    print(f"Found {len(paths)} files in {input_dir}", flush=True)

    dfs: List[pd.DataFrame] = []
    desired = set(required_columns or BASE_REQUIRED_COLUMNS)
    for idx, p in enumerate(paths, start=1):
        try:
            if p.endswith((".parquet", ".pq")):
                cols = None
                schema_cols = _get_parquet_columns(p)
                if schema_cols is not None:
                    inter = [c for c in schema_cols if c in desired]
                    cols = inter if inter else None
                df = pd.read_parquet(p, columns=cols)
            elif p.endswith(".feather"):
                # Feather does not support column projection without prior schema fetch; read all then reduce
                df = pd.read_feather(p)
                keep = [c for c in df.columns if c in desired]
                if keep:
                    df = df[keep]
            elif p.endswith(".csv"):
                df = pd.read_csv(p, usecols=lambda c: (c in desired))
            elif p.endswith(".tsv"):
                df = pd.read_csv(p, sep="\t", usecols=lambda c: (c in desired))
            else:
                continue
            dfs.append(df)
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}", flush=True)
            continue

        if (idx % progress_every == 0) or (idx == len(paths)):
            nrows = sum(len(x) for x in dfs)
            print(f"Loaded {idx}/{len(paths)} files, cumulative rows={nrows}", flush=True)

    if not dfs:
        raise FileNotFoundError("Found files, but none loadable.")

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Concatenated: shape={df_all.shape}", flush=True)
    return df_all


def apply_test_glob(df: pd.DataFrame, pattern: Optional[str]) -> pd.DataFrame:
    if not pattern:
        return df
    mask = df["gff_filename"].astype(str).str.contains(
        glob_to_regex(pattern), regex=True, na=False
    )
    return df.loc[mask].copy()


def glob_to_regex(pat: str) -> str:
    # very small helper to reuse glob-like mask on a Series via regex
    regex = re.escape(pat)
    regex = regex.replace(r"\*", ".*").replace(r"\?", ".")
    return f"^{regex}$"


def _list_fasta_basenames(fasta_dir: Optional[str]) -> Set[str]:
    """Return lowercase basenames of FASTA files under a directory. Empty set if missing/None."""
    names: Set[str] = set()
    if not fasta_dir:
        return names
    try:
        if not os.path.isdir(fasta_dir):
            print(f"[Exclude] FASTA dir not found: {fasta_dir}; skipping exclusion.", flush=True)
            return names
        for ext in (".fasta", ".fa", ".fna"):
            for fn in os.listdir(fasta_dir):
                if fn.lower().endswith(ext):
                    names.add(fn.strip())
    except Exception as e:
        print(f"[Exclude] Warning: failed to list FASTA dir {fasta_dir}: {e}", flush=True)
    return set(x.lower() for x in names)


def exclude_fastas_in_dir(df: pd.DataFrame, fasta_dir: Optional[str]) -> pd.DataFrame:
    """Drop rows whose 'fasta_filename' is present in the given directory.

    Intended to prevent leakage from test genomes when deriving rules/metrics.
    """
    if (fasta_dir is None) or (str(fasta_dir).strip() == ""):
        return df
    exclude = _list_fasta_basenames(fasta_dir)
    if not exclude:
        return df
    d = df.copy()
    before = len(d)
    # normalize to lowercase basenames for robust matching
    ff = d["fasta_filename"].astype("string").str.strip().str.lower()
    mask = ~ff.isin(exclude)
    d = d.loc[mask].copy()
    dropped = before - len(d)
    print(f"[Exclude] Excluding rows for {len(exclude)} FASTA files under {os.path.abspath(fasta_dir)}. Dropped rows={dropped}, remaining={len(d)}.", flush=True)
    return d


def include_fastas_in_dir(df: pd.DataFrame, fasta_dir: Optional[str]) -> pd.DataFrame:
    """Keep only rows whose 'fasta_filename' is present in the given directory.

    Used to restrict plotting to the test set when retention is applied.
    """
    if (fasta_dir is None) or (str(fasta_dir).strip() == ""):
        return df
    include = _list_fasta_basenames(fasta_dir)
    if not include:
        return df
    d = df.copy()
    before = len(d)
    ff = d["fasta_filename"].astype("string").str.strip().str.lower()
    mask = ff.isin(include)
    d = d.loc[mask].copy()
    dropped = before - len(d)
    print(f"[Include] Restricting to rows for {len(include)} FASTA files under {os.path.abspath(fasta_dir)}. Kept rows={len(d)} (dropped {dropped}).", flush=True)
    return d


# --------------------------- Preprocessing ---------------------------

# Base columns always expected in processed tables
BASE_REQUIRED_COLUMNS = [
    "seqid",
    "start",
    "end",
    "strand",
    "locus_tag",
    "gene",
    "Name",
    "product",
    "inference",
    "protein_id",
    "sources",
    "canonical_gene_names",
    "gff_filename",
    "fasta_filename",
    "Phylum",
    "Class",
    "Order",
    "Family",
    "Genus",
    "Species",
    "contig_len",
    "row_type",
]


def build_required_columns(mask_column: str, phenotype_col: Optional[str] = None) -> List[str]:
    cols = list(BASE_REQUIRED_COLUMNS)
    if mask_column and mask_column not in cols:
        cols.append(mask_column)
    # Ensure the selected phenotype metadata column is loaded
    if phenotype_col and phenotype_col not in cols:
        cols.append(phenotype_col)
    return cols


def validate_schema(df: pd.DataFrame, mask_column: str, phenotype_col: Optional[str] = None) -> None:
    required = build_required_columns(mask_column, phenotype_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def coerce_types(df: pd.DataFrame, mask_column: str, phenotype_col: Optional[str] = None) -> pd.DataFrame:
    # Core integer coordinates
    for c in ("start", "end", "contig_len"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    # Mask flag as boolean
    if mask_column in df.columns:
        df[mask_column] = _coerce_series_to_boolean(df[mask_column])
    # Strings
    for c in [
        "seqid", "strand", "locus_tag", "gene", "Name", "product", "inference",
        "protein_id", "sources", "canonical_gene_names", "gff_filename", "fasta_filename",
        "Phylum", "Class", "Order", "Family", "Genus", "Species", "row_type",
    ]:
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df


def _coerce_series_to_boolean(s: pd.Series) -> pd.Series:
    """Robustly coerce a Series with mixed encodings to pandas nullable boolean.

    Accepts: True/False, 1/0, "1"/"0", "true"/"false", "yes"/"no", "y"/"n", case-insensitive.
    Returns a Series of dtype 'boolean' with pd.NA for unrecognized values.
    """
    if is_bool_dtype(s):
        return s.astype("boolean")
    # Numeric: treat 1 -> True, 0 -> False, others -> NA
    if is_numeric_dtype(s):
        sn = pd.to_numeric(s, errors="coerce")
        true_mask = sn == 1
        false_mask = sn == 0
        out = pd.Series(pd.NA, index=s.index, dtype="boolean")
        out[true_mask] = True
        out[false_mask] = False
        return out
    # String/object: normalize and map
    ss = s.astype("string").str.strip().str.lower()
    true_vals = {"true", "t", "yes", "y", "1", "1.0"}
    false_vals = {"false", "f", "no", "n", "0", "0.0"}
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out[ss.isin(true_vals)] = True
    out[ss.isin(false_vals)] = False
    return out
def canonical_gene_name(name: Optional[str]) -> Optional[str]:
    """Normalize a gene name for robust matching.

    Mirrors logic used in codon_msa.canonical_gene_name.
    """
    if name is None:
        return None
    s = str(name).strip()
    if not s or s == "." or s.lower() == "none":
        return None
    s = s.lower()
    s = re.sub(r"_[0-9]+$", "", s)  # drop locus_tag-like suffixes
    s = re.sub(r"^sig([a-z])$", r"sigma_\1", s)  # unify sigma notations
    if not re.match(r"^[a-z0-9_\-]+$", s):
        return None
    return s


############################ Phenotype handling ############################

def detect_phenotype_kind(series: pd.Series, phenotype_type: str = "auto") -> Tuple[str, pd.Series]:
    """Detect phenotype kind and return (kind, normalized_series).

    kind in {"binary", "categorical", "numeric"}.
    - binary: returns boolean nullable series
    - categorical: returns string series
    - numeric: returns float series
    """
    s = series.copy()
    if phenotype_type not in {"auto", "binary", "categorical", "numeric"}:
        phenotype_type = "auto"

    if phenotype_type == "binary":
        return "binary", _coerce_series_to_boolean(s)
    if phenotype_type == "numeric":
        return "numeric", pd.to_numeric(s, errors="coerce").astype(float)
    if phenotype_type == "categorical":
        return "categorical", s.astype("string")

    # auto detection
    s_bool = _coerce_series_to_boolean(s)
    uniq_bool = set(x for x in s_bool.dropna().unique().tolist())
    if uniq_bool.issubset({True, False}) and len(uniq_bool) in (1, 2) and len(s_bool.dropna()) > 0:
        return "binary", s_bool

    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().sum() >= max(3, int(0.5 * len(s))):
        return "numeric", s_num.astype(float)

    return "categorical", s.astype("string")


def build_per_genome_labels(
    df: pd.DataFrame,
    phenotype_col: Optional[str],
    phenotype_type: str = "auto",
    base_label: str = "Phenotype",
    bins_for_numeric: int = 4,
) -> pd.DataFrame:
    """Return per-genome taxonomy and phenotype labels.

    Columns: gff_filename, Order, phenotype_kind, phenotype_value, PhenotypeGroup, PhenotypeBin (optional)
    """
    rows: List[Dict[str, object]] = []
    if phenotype_col and phenotype_col in df.columns:
        kind, norm = detect_phenotype_kind(df[phenotype_col], phenotype_type)
    else:
        kind, norm = "categorical", pd.Series(pd.NA, index=df.index)

    tmp = df[["gff_filename", "Order"]].copy()
    tmp["_pheno_norm"] = norm

    for gff, sub in tmp.groupby("gff_filename", sort=False):
        order_val = first_nonnull(sub["Order"]) if "Order" in sub.columns else pd.NA
        if kind == "binary":
            v = first_nonnull(sub["_pheno_norm"].astype("boolean"))
            group = pd.NA if pd.isna(v) else (f"{base_label}+" if bool(v) else f"{base_label}-")
            rows.append({
                "gff_filename": gff,
                "Order": order_val,
                "phenotype_kind": kind,
                "phenotype_value": v,
                "phenotype_binary": v,
                "PhenotypeGroup": group,
            })
        elif kind == "numeric":
            v = pd.to_numeric(sub["_pheno_norm"], errors="coerce").mean()
            rows.append({
                "gff_filename": gff,
                "Order": order_val,
                "phenotype_kind": kind,
                "phenotype_value": float(v) if not np.isnan(v) else pd.NA,
            })
    else:
            v = first_nonnull(sub["_pheno_norm"].astype("string"))
            rows.append({
                "gff_filename": gff,
                "Order": order_val,
                "phenotype_kind": kind,
                "phenotype_value": v,
                "PhenotypeGroup": v,
            })

    pg = pd.DataFrame(rows)
    if not pg.empty and pg["phenotype_kind"].iloc[0] == "numeric":
        s = pd.to_numeric(pg["phenotype_value"], errors="coerce")
        try:
            bins = pd.qcut(s, q=bins_for_numeric, duplicates="drop")
            pg["PhenotypeBin"] = bins.astype("string")
        except Exception:
            pg["PhenotypeBin"] = pd.Series(["All"] * len(pg))
    return pg



def deduplicate_loci(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse potential duplicates to one row per (gff_filename, locus_key).

    locus_key fallback priority:
      1) locus_tag
      2) protein_id
      3) gene
      4) Name
      5) seqid:start:end:strand (coordinate signature)

    Selection preference within a group:
      has protein_id → longer length → first.
    """
    df = df.copy()
    df["length"] = (df["end"] - df["start"] + 1).astype("Int64")
    # Build coordinate signature
    coord_sig = (
        df["seqid"].astype(str)
        + ":" + df["start"].astype("Int64").astype(str)
        + ":" + df["end"].astype("Int64").astype(str)
        + ":" + df["strand"].astype(str)
    )
    df["locus_key"] = df["locus_tag"].astype("string")
    df.loc[df["locus_key"].isna() | (df["locus_key"].str.len() == 0), "locus_key"] = df["protein_id"].astype("string")
    df.loc[df["locus_key"].isna() | (df["locus_key"].str.len() == 0), "locus_key"] = df["gene"].astype("string")
    df.loc[df["locus_key"].isna() | (df["locus_key"].str.len() == 0), "locus_key"] = df["Name"].astype("string")
    df.loc[df["locus_key"].isna() | (df["locus_key"].str.len() == 0), "locus_key"] = coord_sig.astype("string")

    def pick(group: pd.DataFrame) -> pd.Series:
        g = group.copy()
        g["_has_prot"] = g["protein_id"].notna()
        g = g.sort_values(["_has_prot", "length"], ascending=[False, False])
        return g.iloc[0]

    picked = df.groupby(["gff_filename", "locus_key"], as_index=False, sort=False).apply(pick)
    if isinstance(picked.index, pd.MultiIndex):
        picked = picked.reset_index(drop=True)
    return picked.drop(columns=[c for c in ("_has_prot",) if c in picked.columns])


# --------------------------- Interval helpers ---------------------------

def merge_intervals(starts: np.ndarray, ends: np.ndarray) -> List[Tuple[int, int]]:
    """Merge half-closed genomic intervals [start, end] inclusive.
    Inputs are 1-based inclusive coordinates. Returns a list of merged intervals.
    """
    arr = np.column_stack((starts, ends))
    arr = arr[~np.any(pd.isna(arr), axis=1)]
    if arr.size == 0:
        return []
    arr = arr[np.argsort(arr[:, 0], kind="mergesort")]
    merged: List[Tuple[int, int]] = []
    cs, ce = int(arr[0, 0]), int(arr[0, 1])
    for s, e in arr[1:]:
        s, e = int(s), int(e)
        if s <= ce + 1:  # overlapping or touching
            if e > ce:
                ce = e
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged


def intervals_total_length(ivals: List[Tuple[int, int]]) -> int:
    return sum(e - s + 1 for s, e in ivals)


def overlap_length(window: Tuple[int, int], ivals: List[Tuple[int, int]]) -> int:
    ws, we = window
    total = 0
    # binary search over sorted intervals
    # simple linear scan is fine given typical counts, but we keep it efficient
    for s, e in ivals:
        if e < ws:
            continue
        if s > we:
            break
        total += max(0, min(we, e) - max(ws, s) + 1)
    return total


# --------------------------- Metrics ---------------------------

@dataclass
class GenomeIntervals:
    contig_lengths: Dict[str, int]  # seqid -> contig length (max end)
    all_union: Dict[str, List[Tuple[int, int]]]  # seqid -> merged intervals for all loci
    mask_union: Dict[str, List[Tuple[int, int]]]  # seqid -> merged intervals for mask loci


def build_genome_intervals(df_g: pd.DataFrame) -> GenomeIntervals:
    # Robust contig lengths: max numeric end per contig; treat missing as 0
    ends_numeric = pd.to_numeric(df_g["end"], errors="coerce")
    contig_len_series = ends_numeric.groupby(df_g["seqid"]).max().fillna(0).astype(int)
    contig_lengths: Dict[str, int] = contig_len_series.to_dict()
    all_union: Dict[str, List[Tuple[int, int]]] = {}
    mask_union: Dict[str, List[Tuple[int, int]]] = {}
    for seqid, sub in df_g.groupby("seqid", sort=False):
        # Filter out rows with missing coordinates to avoid int conversion errors
        valid_coords = sub["start"].notna() & sub["end"].notna()
        starts_all = sub.loc[valid_coords, "start"].to_numpy(dtype=np.int64)
        ends_all = sub.loc[valid_coords, "end"].to_numpy(dtype=np.int64)
        all_union[seqid] = merge_intervals(starts_all, ends_all)
        m = sub["mask_flag"].fillna(False).to_numpy()
        if m.any():
            sel = valid_coords & m
            starts_sp = sub.loc[sel, "start"].to_numpy(dtype=np.int64)
            ends_sp = sub.loc[sel, "end"].to_numpy(dtype=np.int64)
            mask_union[seqid] = merge_intervals(starts_sp, ends_sp)
        else:
            mask_union[seqid] = []
    return GenomeIntervals(contig_lengths, all_union, mask_union)


def per_genome_metrics(
    df: pd.DataFrame,
    progress_every: int = 500,
) -> Tuple[pd.DataFrame, Dict[str, GenomeIntervals]]:
    rows = []
    genome_cache: Dict[str, GenomeIntervals] = {}

    groups = list(df.groupby("gff_filename", sort=False))
    total = len(groups)
    print(f"Computing per-genome metrics for {total} genomes...", flush=True)
    for idx, (gff, df_g) in enumerate(groups, start=1):
        gi = build_genome_intervals(df_g)
        genome_cache[gff] = gi

        L_genome = int(sum(gi.contig_lengths.values()))
        U_mask = int(sum(intervals_total_length(v) for v in gi.mask_union.values()))
        U_all = int(sum(intervals_total_length(v) for v in gi.all_union.values()))

        n_all = int(len(df_g))
        n_mask = int(df_g["mask_flag"].fillna(False).sum())
        p_g = n_mask / n_all if n_all else 0.0

        cov_genome = U_mask / L_genome if L_genome else 0.0
        cov_coding = U_mask / U_all if U_all else 0.0

        # NND median per genome
        nnd_values = nearest_neighbor_distances(df_g)
        median_nnd = float(np.median(nnd_values)) if len(nnd_values) else np.nan

        # Annotation quality (product field)
        prod = df_g["product"].astype("string") if "product" in df_g.columns else pd.Series(dtype="string")
        prod_lc = prod.str.lower()
        is_null = prod.isna() | (prod.str.len() == 0) | (prod == "None") | (prod == ".")
        is_hyp = prod_lc.fillna("").str.contains(r"\bhypothetical\b", regex=True)

        rows.append({
            "gff_filename": gff,
            "Species": first_nonnull(df_g["Species"]),
            "Phylum": first_nonnull(df_g["Phylum"]),
            "Class": first_nonnull(df_g["Class"]),
            "Order": first_nonnull(df_g["Order"]),
            "Family": first_nonnull(df_g["Family"]),
            "Genus": first_nonnull(df_g["Genus"]),
            "n_all_loci": n_all,
            "n_mask_loci": n_mask,
            "p_mask_loci": p_g,
            "U_mask_bp": U_mask,
            "U_all_bp": U_all,
            "L_genome_bp": L_genome,
            "cov_genome": cov_genome,
            "cov_coding": cov_coding,
            "median_nnd_bp": median_nnd,
            "pct_product_null": float(is_null.mean()) if len(prod) else np.nan,
            "pct_product_hypothetical": float(is_hyp.mean()) if len(prod) else np.nan,
        })

        if (idx % progress_every == 0) or (idx == total):
            print(f"  per-genome metrics: {idx}/{total} done", flush=True)

    return pd.DataFrame(rows), genome_cache


def first_nonnull(s: pd.Series):
    for x in s:
        if pd.notna(x):
            return x
    return pd.NA


def nearest_neighbor_distances(df_g: pd.DataFrame) -> List[int]:
    dists: List[int] = []
    sp = df_g[df_g["mask_flag"].fillna(False)]
    if sp.empty:
        return dists
    for seqid, sub in sp.groupby("seqid", sort=False):
        valid = sub["start"].notna() & sub["end"].notna()
        if not valid.any():
            continue
        starts = sub.loc[valid, "start"].to_numpy(dtype=np.int64)
        ends = sub.loc[valid, "end"].to_numpy(dtype=np.int64)
        mids = ((starts + ends) // 2)
        if len(mids) < 2:
            continue
        mids.sort()
        # nearest neighbor distance for each point
        left = np.empty_like(mids)
        right = np.empty_like(mids)
        left[0] = np.iinfo(mids.dtype).max
        left[1:] = mids[1:] - mids[:-1]
        right[-1] = np.iinfo(mids.dtype).max
        right[:-1] = mids[1:] - mids[:-1]
        nn = np.minimum(left, right)
        dists.extend(nn[1:-1].tolist() + [nn[0], nn[-1]])
    return dists


def phenotype_association(per_genome: pd.DataFrame, base_label: str) -> pd.DataFrame:
    """Compute association between phenotype and genome-level metrics.

    - binary: effect sizes and AUROC
    - categorical: per-group means and variance explained (eta^2)
    - numeric: Pearson and Spearman correlations
    """
    metrics = [
        ("p_mask_loci", "Fraction masked loci"),
        ("cov_genome", "Mask bp / genome bp"),
        ("cov_coding", "Mask bp / coding bp"),
        ("n_mask_loci", "# masked loci"),
        ("median_nnd_bp", "Median NND (bp)"),
    ]
    kind = per_genome["phenotype_kind"].iloc[0] if "phenotype_kind" in per_genome.columns else "categorical"
    rows: List[Dict[str, object]] = []

    if kind == "binary":
        mask_true = per_genome["phenotype_binary"].astype("boolean") == True
        mask_false = per_genome["phenotype_binary"].astype("boolean") == False
        for col, label in metrics:
            x = per_genome.loc[mask_true, col].to_numpy(dtype=float)
            y = per_genome.loc[mask_false, col].to_numpy(dtype=float)
            rows.append({
                "metric": col,
                "label": label,
                "mean_TRUE": np.nanmean(x),
                "mean_FALSE": np.nanmean(y),
                "cohens_d": cohens_d(x, y),
                "cliffs_delta": cliffs_delta(x, y),
                "auroc_TRUE_gt_FALSE": auc_mann_whitney(x, y),
                "n_TRUE": int(np.sum(~np.isnan(x))),
                "n_FALSE": int(np.sum(~np.isnan(y))),
            })
        return pd.DataFrame(rows)

    if kind == "numeric":
        s = pd.to_numeric(per_genome["phenotype_value"], errors="coerce").to_numpy(dtype=float)
        for col, label in metrics:
            y = per_genome[col].to_numpy(dtype=float)
            mask = ~np.isnan(s) & ~np.isnan(y)
            if mask.sum() >= 2:
                sx = s[mask]
                sy = y[mask]
                sxm = sx - sx.mean()
                sym = sy - sy.mean()
                denom = (np.sqrt((sxm ** 2).sum()) * np.sqrt((sym ** 2).sum()))
                pearson = float((sxm * sym).sum() / denom) if denom != 0 else 0.0
                rx = pd.Series(sx).rank(method="average").to_numpy()
                ry = pd.Series(sy).rank(method="average").to_numpy()
                rxm = rx - rx.mean()
                rym = ry - ry.mean()
                denom_r = (np.sqrt((rxm ** 2).sum()) * np.sqrt((rym ** 2).sum()))
                spearman = float((rxm * rym).sum() / denom_r) if denom_r != 0 else 0.0
                n = int(mask.sum())
            else:
                pearson = float("nan")
                spearman = float("nan")
                n = int(mask.sum())
            rows.append({
                "metric": col,
                "label": label,
                "pearson_r": pearson,
                "spearman_rho": spearman,
                "n": n,
            })
        return pd.DataFrame(rows)

    # categorical (>2): per-group means and eta^2 (variance explained)
    groups = per_genome["PhenotypeGroup"].astype("string")
    for col, label in metrics:
        y = per_genome[col].to_numpy(dtype=float)
        mask_valid = (~pd.isna(groups)) & (~np.isnan(y))
        if mask_valid.sum() == 0:
            rows.append({"metric": col, "label": label})
            continue
        yv = y[mask_valid]
        gv = groups[mask_valid]
        overall_mean = float(np.mean(yv)) if len(yv) else float("nan")
        ss_between = 0.0
        ss_total = float(((yv - overall_mean) ** 2).sum()) if len(yv) else float("nan")
        means: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for g, idx in pd.Series(gv).groupby(gv).groups.items():
            vals = yv[list(idx)]
            means[str(g)] = float(np.mean(vals)) if len(vals) else float("nan")
            counts[str(g)] = int(len(vals))
        for g, n in counts.items():
            if n:
                ss_between += n * (means[g] - overall_mean) ** 2
        eta_sq = float(ss_between / ss_total) if ss_total and not np.isnan(ss_total) else float("nan")
        row: Dict[str, object] = {"metric": col, "label": label, "eta_squared": eta_sq, "overall_mean": overall_mean}
        for i, (g, m) in enumerate(list(means.items())[:10]):
            row[f"mean[{g}]"] = m
            row[f"n[{g}]"] = counts[g]
        rows.append(row)
    return pd.DataFrame(rows)


    
# --------------------------- Plotting ---------------------------

def _init_plotting() -> None:
    if sns is not None:
        sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _savefig(fig: plt.Figure, out_base: str, pdf: Optional[PdfPages] = None) -> None:
    fig.tight_layout()
    fig.savefig(out_base + ".png", bbox_inches="tight")
    try:
        fig.savefig(out_base + ".pdf", bbox_inches="tight")
    except Exception:
        pass
    if pdf is not None:
        try:
            pdf.savefig(fig, bbox_inches="tight")
        except Exception:
            pass
    plt.close(fig)


def _phenotype_labels(s: pd.Series, base_label: str) -> pd.Series:
    """Map boolean/nullable series to dynamic +/− labels for the chosen phenotype.

    True -> f"{base_label}+", False -> f"{base_label}-", NA -> 'Unknown'.
    """
    s_bool = s.astype("boolean")
    pos = f"{base_label}+"
    neg = f"{base_label}-"
    return s_bool.map({True: pos, False: neg}).fillna("Unknown")


def plot_per_genome_metrics(per_genome: pd.DataFrame, outdir: str, pdf: Optional[PdfPages] = None, base_label: str = "Phenotype") -> None:
    _init_plotting()
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    df = per_genome.copy()
    kind = df["phenotype_kind"].iloc[0] if "phenotype_kind" in df.columns else "categorical"

    metrics = [
        ("p_mask_loci", "Fraction masked loci"),
        ("cov_genome", "Mask bp / genome bp"),
        ("cov_coding", "Mask bp / coding bp"),
        ("n_mask_loci", "# masked loci"),
        ("median_nnd_bp", "Median NND (bp)"),
    ]
    if kind in ("binary", "categorical"):
        group_col = "PhenotypeGroup"
        for col, label in metrics:
            fig, ax = plt.subplots(figsize=(5, 4))
            if sns is not None:
                sns.violinplot(data=df, x=group_col, y=col, inner=None, cut=0, linewidth=0.8, ax=ax)
                sns.boxplot(data=df, x=group_col, y=col, whis=1.5, width=0.25, showcaps=True,
                            boxprops={"facecolor": "white"}, ax=ax)
                sns.stripplot(data=df, x=group_col, y=col, color="k", alpha=0.35, size=2, jitter=0.15, ax=ax)
            else:
                df.boxplot(column=col, by=group_col, ax=ax)
                ax.get_figure().suptitle("")
            ax.set_xlabel("")
            ax.set_ylabel(label)
            ax.set_title(label)
            _savefig(fig, os.path.join(outdir, "plots", f"metric_{col}"), pdf=pdf)
    else:
        xlab = base_label
        for col, label in metrics:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(df["phenotype_value"], df[col], s=8, alpha=0.5, color="#1f77b4")
            ax.set_xlabel(xlab)
            ax.set_ylabel(label)
            ax.set_title(f"{label} vs {xlab}")
            _savefig(fig, os.path.join(outdir, "plots", f"metric_vs_{col}"), pdf=pdf)


def plot_windows(win: pd.DataFrame, per_genome: pd.DataFrame, outdir: str, pdf: Optional[PdfPages] = None, base_label: str = "Phenotype") -> None:
    if win.empty:
        return
    _init_plotting()
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    kind = per_genome["phenotype_kind"].iloc[0] if "phenotype_kind" in per_genome.columns else "categorical"
    if kind == "numeric":
        ph = per_genome[["gff_filename", "PhenotypeBin"]].copy()
        ph = ph.rename(columns={"PhenotypeBin": "Group"})
    elif kind == "binary":
        ph = per_genome[["gff_filename", "PhenotypeGroup"]].copy().rename(columns={"PhenotypeGroup": "Group"})
    else:
        ph = per_genome[["gff_filename", "PhenotypeGroup"]].copy().rename(columns={"PhenotypeGroup": "Group"})
    w = win.merge(ph, on="gff_filename", how="left")

    fig, ax = plt.subplots(figsize=(6, 4))
    if sns is not None:
        sns.kdeplot(data=w, x="mask_frac", hue="Group", fill=True, common_norm=False, alpha=0.3, ax=ax)
    else:
        for k, sub in w.groupby("Group"):
            ax.hist(sub["mask_frac"], bins=50, alpha=0.4, label=str(k), density=True)
        ax.legend()
    ax.set_xlabel("Window mask bp fraction")
    ax.set_ylabel("Density")
    ax.set_title("Window mask fraction distribution")
    _savefig(fig, os.path.join(outdir, "plots", "windows_mask_frac_density"), pdf=pdf)

    # Add zero vs non-zero barplot by phenotype to highlight mass at zero
    w["is_zero"] = (w["mask_frac"] <= 0).astype(int)
    counts_wide = (
        w.groupby(["Group", "is_zero"]).size().unstack(fill_value=0)
    )
    if not counts_wide.empty:
        counts_wide = counts_wide.rename(columns={1: "zero", 0: "nonzero"})
        counts_wide = counts_wide.reindex(columns=["zero", "nonzero"], fill_value=0)
        props = counts_wide.div(counts_wide.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    else:
        props = pd.DataFrame(columns=["zero", "nonzero"])

    if not props.empty:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for ph, row_vals in props.sort_index().iterrows():
            label = str(ph)
            vals = row_vals[["zero", "nonzero"]].to_list()
            bars = ax2.bar([f"{label}\nzero", f"{label}\nnonzero"], vals, alpha=0.8)
            for bar in bars:
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height * 100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax2.set_ylabel("Proportion of windows")
        ax2.set_title("Zero vs non-zero window mask fraction")
        ax2.set_ylim(0, min(1.05, max(1.0, ax2.get_ylim()[1])))
        _savefig(fig2, os.path.join(outdir, "plots", "windows_zero_nonzero_bar"), pdf=pdf)


def plot_nnd(nnd_df: pd.DataFrame, per_genome: pd.DataFrame, outdir: str, pdf: Optional[PdfPages] = None, base_label: str = "Phenotype") -> None:
    if nnd_df.empty:
        return
    _init_plotting()
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    kind = per_genome["phenotype_kind"].iloc[0] if "phenotype_kind" in per_genome.columns else "categorical"
    if kind == "numeric":
        ph = per_genome[["gff_filename", "PhenotypeBin"]].copy()
        ph = ph.rename(columns={"PhenotypeBin": "Group"})
    elif kind == "binary":
        ph = per_genome[["gff_filename", "PhenotypeGroup"]].copy().rename(columns={"PhenotypeGroup": "Group"})
    else:
        ph = per_genome[["gff_filename", "PhenotypeGroup"]].copy().rename(columns={"PhenotypeGroup": "Group"})
    d = nnd_df.merge(ph, on="gff_filename", how="left")

    fig, ax = plt.subplots(figsize=(6, 4))
    if sns is not None:
        sns.ecdfplot(data=d, x="distance_bp", hue="Group", ax=ax)
    else:
        # Matplotlib fallback: cumulative hist
        for k, sub in d.groupby("Group"):
            vals = sub["distance_bp"].to_numpy()
            vals = vals[~np.isnan(vals)]
            vals.sort()
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, y, label=str(k))
        ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("Nearest-neighbor distance (bp)")
    ax.set_ylabel("ECDF")
    ax.set_title("NND distribution (pooled)")
    _savefig(fig, os.path.join(outdir, "plots", "nnd_ecdf"), pdf=pdf)


def plot_spore_gene_lengths(df: pd.DataFrame, outdir: str, pdf: Optional[PdfPages] = None) -> None:
    """Plot distribution of masked locus lengths (legacy name)."""
    if df.empty or not {"start", "end"}.issubset(df.columns):
        return
    _init_plotting()
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    d = df[df["mask_flag"].fillna(False)].copy()
    if d.empty:
        return
    d["length_bp"] = (d["end"].astype("Int64") - d["start"].astype("Int64") + 1).astype("Int64")
    # phenotype per genome
    # we may not have phenotype per row; enrich via per-genome mapping if present
    fig, ax = plt.subplots(figsize=(6, 4))
    if sns is not None:
        sns.histplot(d, x="length_bp", bins=100, kde=False, ax=ax)
    else:
        ax.hist(d["length_bp"].dropna().to_numpy(), bins=100)
    ax.set_xlabel("Masked locus length (bp)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of masked gene lengths")
    _savefig(fig, os.path.join(outdir, "plots", "masked_gene_length_distribution"), pdf=pdf)


def _add_text_page(pdf: PdfPages, title: str, lines: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    ax.set_title(title, fontsize=16, pad=20)
    text = "\n".join(lines)
    ax.text(0.02, 0.98, text, va='top', ha='left', fontsize=11, family='monospace')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# --------------------------- Window sampling ---------------------------

def build_sampling_frame(genome_cache: Dict[str, GenomeIntervals],
                         allowed_genomes: Optional[Sequence[str]] = None,
                         window_size: int = 1_000_000) -> pd.DataFrame:
    """Create a DataFrame of contigs eligible for sampling windows.
    Columns: gff_filename, seqid, contig_len, mask_union (list of intervals)
    Weight is (contig_len - window_size + 1), clipped at 0.
    """
    rows = []
    for gff, gi in genome_cache.items():
        if allowed_genomes is not None and gff not in allowed_genomes:
            continue
        for seqid, clen in gi.contig_lengths.items():
            if clen and clen >= window_size:
                rows.append({
                    "gff_filename": gff,
                    "seqid": seqid,
                    "contig_len": int(clen),
                    "mask_union": gi.mask_union.get(seqid, []),
                })
    sf = pd.DataFrame(rows)
    if sf.empty:
        return sf
    sf["weight"] = (sf["contig_len"] - window_size + 1).clip(lower=0)
    sf = sf[sf["weight"] > 0].reset_index(drop=True)
    return sf


def sample_windows(sf: pd.DataFrame, W: int, window_size: int, seed: int = 42) -> pd.DataFrame:
    if sf.empty:
        return pd.DataFrame(columns=[
            "gff_filename", "seqid", "start", "end", "mask_bp", "mask_frac"
        ])
    rng = np.random.default_rng(seed)
    probs = sf["weight"].to_numpy(dtype=float)
    probs = probs / probs.sum()

    choices = rng.choice(len(sf), size=W, replace=True, p=probs)
    starts = []
    ends = []
    sp_bp = []
    meta = sf.iloc[choices].reset_index(drop=True)
    for i, row in meta.iterrows():
        L = int(row["contig_len"])
        s = int(rng.integers(1, L - window_size + 2))  # inclusive start
        e = s + window_size - 1
        starts.append(s)
        ends.append(e)
        sp = overlap_length((s, e), row["mask_union"]) if row["mask_union"] else 0
        sp_bp.append(sp)
    out = meta.assign(start=starts, end=ends)
    out["mask_bp"] = sp_bp
    out["mask_frac"] = out["mask_bp"] / float(window_size)
    return out


def summarize_windows(df_w: pd.DataFrame) -> pd.DataFrame:
    if df_w.empty:
        return pd.DataFrame([{"n": 0}])
    s = df_w["mask_frac"].to_numpy()
    q = np.quantile(s, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return pd.DataFrame([{
        "n": len(s),
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "q0": q[0], "q01": q[1], "q05": q[2], "q25": q[3],
        "q50": q[4], "q75": q[5], "q95": q[6], "q99": q[7], "q100": q[8],
    }])


# --------------------------- Phenotype contrast ---------------------------

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    nx, ny = len(x), len(y)
    sx2, sy2 = x.var(ddof=1), y.var(ddof=1)
    sp = math.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if sp == 0:
        return 0.0
    return (x.mean() - y.mean()) / sp


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    # Efficient Cliff's delta using ranks
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    # Use Mann-Whitney relation
    allv = np.concatenate([x, y])
    ranks = pd.Series(allv).rank(method="average").to_numpy()
    rx = ranks[: len(x)].sum()
    U = rx - len(x) * (len(x) + 1) / 2.0
    delta = (2 * U) / (len(x) * len(y)) - 1
    return float(delta)


def auc_mann_whitney(x: np.ndarray, y: np.ndarray) -> float:
    # AUROC from Mann-Whitney U / (n_pos*n_neg)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    allv = np.concatenate([x, y])
    ranks = pd.Series(allv).rank(method="average").to_numpy()
    rx = ranks[: len(x)].sum()
    U = rx - len(x) * (len(x) + 1) / 2.0
    return float(U / (len(x) * len(y)))


def _print_within_order_variation(
    per_genome: pd.DataFrame,
    base_label: str,
    context: str = "",
) -> None:
    """Print a brief summary indicating whether there is within-Order variation
    in phenotype (both Spore+ and Spore- present). Helps diagnose CMH degeneracy.
    """
    try:
        if "Order" not in per_genome.columns:
            print(f"[{context}] No 'Order' column; skipping within-clade variation check.", flush=True)
            return
        col = "phenotype_binary" if "phenotype_binary" in per_genome.columns else None
        if col is None:
            print(f"[{context}] No phenotype column for within-clade check.", flush=True)
            return
        pg = per_genome[["Order", col]].copy()
        pg[col] = pg[col].astype("boolean")
        pg = pg.dropna(subset=[col]).copy()
        if pg.empty:
            print(f"[{context}] No known phenotypes; skipping within-clade variation check.", flush=True)
            return
        counts = pg.groupby("Order")[col].value_counts().unstack(fill_value=0)
        # Ensure both columns exist
        counts = counts.reindex(columns=[True, False], fill_value=0)
        n_orders = counts.shape[0]
        has_true = counts[True] > 0
        has_false = counts[False] > 0
        informative = int((has_true & has_false).sum())
        print(
            f"[{context}] Within-Order phenotype variation for '{base_label}': {informative}/{n_orders} Orders have both phenotypes.",
            flush=True,
        )
        non_info_orders = counts.index[~(has_true & has_false)].tolist()
        if non_info_orders:
            sample = ", ".join(map(str, non_info_orders[:10]))
            suffix = "" if len(non_info_orders) <= 10 else " ..."
            print(
                f"[{context}] Orders without both phenotypes (up to 10): {sample}{suffix}",
                flush=True,
            )
    except Exception as e:
        print(f"[{context}] Warning: failed within-clade variation check: {e}", flush=True)


def phenotype_contrast(per_genome: pd.DataFrame, base_label: str) -> pd.DataFrame:
    # Backward-compatible wrapper for association analysis
    return phenotype_association(per_genome, base_label)



######################## Ground-truth enrichment ########################


def _parse_canonical_name_tokens(value: object) -> Set[str]:
    """Return canonical gene tokens parsed from a heterogeneous value."""
    tokens: Set[str] = set()
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return tokens
    s = str(value).strip()
    if not s:
        return tokens
    for part in re.split(r"[;,/|\s]+", s):
        canon = canonical_gene_name(part)
        if canon:
            tokens.add(canon)
    return tokens


def _collect_canonical_names(row: pd.Series) -> List[str]:
    names: Set[str] = set()
    if "canonical_gene_names" in row:
        names.update(_parse_canonical_name_tokens(row.get("canonical_gene_names")))
    for col in ("gene", "Name", "locus_tag"):
        if col in row:
            canon = canonical_gene_name(row.get(col))
            if canon:
                names.add(canon)
    return sorted(names)


def load_ground_truth_gene_clusters(
    phenotype: str,
    freq_threshold: float = GROUND_TRUTH_FREQ_THRESHOLD,
    min_prev: float = DEFAULT_CLUSTER_MIN_PREV,
    metric: str = DEFAULT_CLUSTER_METRIC,
    cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> pd.DataFrame:
    """Return canonical ground-truth genes with their cluster metadata."""

    cache_dir = _resolve_cache_dir(phenotype)
    if cache_dir is None:
        print(f"[GroundTruth] Warning: missing cache directory for phenotype '{phenotype}'.", flush=True)
        return pd.DataFrame(columns=["canonical_gene", "cluster_id", "cluster_label", "freq_lasso", "freq_rf"])

    cpss_path = cache_dir / "cpss_cache.npz"
    rf_path = cache_dir / "rf_halves_cache.npz"
    if not cpss_path.exists() or not rf_path.exists():
        print(f"[GroundTruth] Warning: missing selection caches under {cache_dir}.", flush=True)
        return pd.DataFrame(columns=["canonical_gene", "cluster_id", "cluster_label", "freq_lasso", "freq_rf"])

    try:
        genes_lasso, sel_lasso = _load_selection_matrix(cpss_path, "sel_matrix")
        genes_rf, sel_rf = _load_selection_matrix(rf_path, "sel_matrix")
    except Exception as exc:
        print(f"[GroundTruth] Warning: failed to load selection matrices for '{phenotype}': {exc}", flush=True)
        return pd.DataFrame(columns=["canonical_gene", "cluster_id", "cluster_label", "freq_lasso", "freq_rf"])

    if len(genes_lasso) != len(genes_rf) or not np.array_equal(genes_lasso, genes_rf):
        index_map = {g: i for i, g in enumerate(genes_rf)}
        aligned = np.zeros_like(sel_lasso, dtype=bool)
        for idx, gene in enumerate(genes_lasso):
            j = index_map.get(gene)
            if j is None:
                continue
            aligned[:, idx] = sel_rf[:, j]
        sel_rf = aligned
        genes = genes_lasso
    else:
        genes = genes_lasso

    mapping, base_clusters = _load_cluster_mapping(
        min_prev=min_prev,
        metric=metric,
        cluster_threshold=cluster_threshold,
    )

    clusters = np.empty(len(genes), dtype=int)
    next_cluster = int(base_clusters)
    for i, gene in enumerate(genes):
        cid = mapping.get(gene)
        if cid is None:
            cid = next_cluster
            next_cluster += 1
        clusters[i] = int(cid)

    n_clusters = int(next_cluster)
    freq_lasso = _compute_cluster_frequencies(sel_lasso, clusters, n_clusters)
    freq_rf = _compute_cluster_frequencies(sel_rf, clusters, n_clusters)
    if freq_lasso.size == 0 or freq_rf.size == 0:
        print(f"[GroundTruth] Warning: empty selection matrices for '{phenotype}'.", flush=True)
        return pd.DataFrame(columns=["canonical_gene", "cluster_id", "cluster_label", "freq_lasso", "freq_rf"])

    mask = (freq_lasso > float(freq_threshold)) & (freq_rf > float(freq_threshold))
    rows: List[Dict[str, object]] = []
    for gene, cluster in zip(genes, clusters):
        if cluster >= len(mask) or not mask[cluster]:
            continue
        canon = canonical_gene_name(gene)
        if not canon:
            continue
        rows.append(
            {
                "canonical_gene": canon,
                "cluster_id": int(cluster),
                "cluster_label": f"cluster_{int(cluster)}",
                "freq_lasso": float(freq_lasso[cluster]),
                "freq_rf": float(freq_rf[cluster]),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["canonical_gene", "cluster_id", "cluster_label", "freq_lasso", "freq_rf"])

    result = pd.DataFrame(rows).drop_duplicates(subset=["canonical_gene"])
    result["cluster_id"] = result["cluster_id"].astype("Int64")
    return result


def build_ground_truth_presence(
    df: pd.DataFrame,
    mask_column: str,
    allowed_genes: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return per-genome presence of ground-truth canonical genes."""

    allowed_set = set(allowed_genes) if allowed_genes is not None else None
    sub = df[df[mask_column].fillna(False)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["gff_filename", "canonical_gene"])

    sub["_canonical_list"] = sub.apply(_collect_canonical_names, axis=1)
    exploded = sub.explode("_canonical_list")
    exploded = exploded.dropna(subset=["_canonical_list"])
    exploded = exploded.rename(columns={"_canonical_list": "canonical_gene"})
    exploded["canonical_gene"] = exploded["canonical_gene"].astype("string")
    if allowed_set is not None:
        exploded = exploded[exploded["canonical_gene"].isin(allowed_set)]
    presence = (
        exploded[["gff_filename", "canonical_gene"]]
        .dropna(subset=["gff_filename", "canonical_gene"])
        .drop_duplicates(ignore_index=True)
    )
    presence["gff_filename"] = presence["gff_filename"].astype("string")
    return presence


def _binary_ground_truth_enrichment(
    presence: pd.DataFrame,
    per_genome: pd.DataFrame,
    gene_clusters: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gene_cols = [
        "canonical_gene",
        "pos_present",
        "neg_present",
        "n_pos",
        "n_neg",
        "prop_pos",
        "prop_neg",
        "log2fc",
        "p_value",
        "fdr_bh",
        "cluster_id",
        "cluster_label",
        "freq_lasso",
        "freq_rf",
    ]
    cluster_cols = [
        "cluster_id",
        "cluster_label",
        "pos_present",
        "neg_present",
        "n_pos",
        "n_neg",
        "prop_pos",
        "prop_neg",
        "log2fc",
        "p_value",
        "fdr_bh",
        "freq_lasso",
        "freq_rf",
        "genes",
    ]

    labels = per_genome[["gff_filename", "phenotype_binary"]].copy()
    labels = labels.dropna(subset=["phenotype_binary"])
    if labels.empty:
        return pd.DataFrame(columns=gene_cols), pd.DataFrame(columns=cluster_cols)

    labels["phenotype_binary"] = labels["phenotype_binary"].astype(bool)
    labels["gff_filename"] = labels["gff_filename"].astype("string")

    presence_labeled = presence.merge(labels, on="gff_filename", how="inner")
    presence_labeled = presence_labeled.drop_duplicates(["canonical_gene", "gff_filename"])
    if presence_labeled.empty:
        return pd.DataFrame(columns=gene_cols), pd.DataFrame(columns=cluster_cols)

    n_pos = int(labels["phenotype_binary"].sum())
    n_total = int(labels.shape[0])
    n_neg = int(n_total - n_pos)
    eps = 1e-6

    counts = presence_labeled.pivot_table(
        index="canonical_gene",
        columns="phenotype_binary",
        values="gff_filename",
        aggfunc="nunique",
        fill_value=0,
    )
    if counts.empty:
        gene_table = pd.DataFrame(columns=gene_cols)
    else:
        counts = counts.rename(columns={True: "pos_present", False: "neg_present"})
        if "pos_present" not in counts.columns:
            counts["pos_present"] = 0
        if "neg_present" not in counts.columns:
            counts["neg_present"] = 0
        counts = counts.reset_index()
        counts["n_pos"] = n_pos
        counts["n_neg"] = n_neg
        counts["pos_absent"] = (counts["n_pos"] - counts["pos_present"]).clip(lower=0)
        counts["neg_absent"] = (counts["n_neg"] - counts["neg_present"]).clip(lower=0)
        counts["prop_pos"] = np.where(counts["n_pos"] > 0, counts["pos_present"] / counts["n_pos"], np.nan)
        counts["prop_neg"] = np.where(counts["n_neg"] > 0, counts["neg_present"] / counts["n_neg"], np.nan)
        counts["log2fc"] = np.log2((counts["prop_pos"] + eps) / (counts["prop_neg"] + eps))
        pvals: List[float] = []
        for row in counts.itertuples(index=False):
            if row.n_pos == 0 or row.n_neg == 0:
                pvals.append(float("nan"))
                continue
            try:
                p = fisher_exact_one_sided_greater(
                    int(row.pos_present),
                    int(row.neg_present),
                    int(row.pos_absent),
                    int(row.neg_absent),
                )
            except Exception:
                p = float("nan")
            pvals.append(p)
        counts["p_value"] = pvals
        counts["fdr_bh"] = bh_fdr(np.array(pvals, dtype=float))
        gene_table = counts

    cluster_table = pd.DataFrame(columns=cluster_cols)

    if not gene_table.empty:
        for col in ("cluster_id", "cluster_label", "freq_lasso", "freq_rf"):
            if col not in gene_table.columns:
                gene_table[col] = pd.NA if col in {"cluster_id", "cluster_label"} else np.nan
        if not gene_clusters.empty:
            gene_table = gene_table.merge(gene_clusters, on="canonical_gene", how="left", suffixes=("", "_gt"))
            for col in ("cluster_id", "cluster_label", "freq_lasso", "freq_rf"):
                gt_col = f"{col}_gt"
                if gt_col in gene_table.columns:
                    gene_table[col] = gene_table[gt_col]
                    gene_table = gene_table.drop(columns=[gt_col])
            if "cluster_id" in gene_table.columns:
                gene_table["cluster_id"] = gene_table["cluster_id"].astype("Int64")

    if not gene_clusters.empty:
        cluster_presence = presence_labeled.merge(
            gene_clusters[["canonical_gene", "cluster_id", "cluster_label", "freq_lasso", "freq_rf"]],
            on="canonical_gene",
            how="left",
        ).dropna(subset=["cluster_id"])
        if not cluster_presence.empty:
            cluster_presence["cluster_id"] = cluster_presence["cluster_id"].astype("Int64")
            cluster_counts = cluster_presence.pivot_table(
                index="cluster_id",
                columns="phenotype_binary",
                values="gff_filename",
                aggfunc="nunique",
                fill_value=0,
            ).rename(columns={True: "pos_present", False: "neg_present"})
            if "pos_present" not in cluster_counts.columns:
                cluster_counts["pos_present"] = 0
            if "neg_present" not in cluster_counts.columns:
                cluster_counts["neg_present"] = 0
            cluster_counts = cluster_counts.reset_index()
            cluster_counts["n_pos"] = n_pos
            cluster_counts["n_neg"] = n_neg
            cluster_counts["pos_absent"] = (cluster_counts["n_pos"] - cluster_counts["pos_present"]).clip(lower=0)
            cluster_counts["neg_absent"] = (cluster_counts["n_neg"] - cluster_counts["neg_present"]).clip(lower=0)
            cluster_counts["prop_pos"] = np.where(cluster_counts["n_pos"] > 0, cluster_counts["pos_present"] / cluster_counts["n_pos"], np.nan)
            cluster_counts["prop_neg"] = np.where(cluster_counts["n_neg"] > 0, cluster_counts["neg_present"] / cluster_counts["n_neg"], np.nan)
            cluster_counts["log2fc"] = np.log2((cluster_counts["prop_pos"] + eps) / (cluster_counts["prop_neg"] + eps))
            pvals_cluster: List[float] = []
            for row in cluster_counts.itertuples(index=False):
                if row.n_pos == 0 or row.n_neg == 0:
                    pvals_cluster.append(float("nan"))
                    continue
                try:
                    p = fisher_exact_one_sided_greater(
                        int(row.pos_present),
                        int(row.neg_present),
                        int(row.pos_absent),
                        int(row.neg_absent),
                    )
                except Exception:
                    p = float("nan")
                pvals_cluster.append(p)
            cluster_counts["p_value"] = pvals_cluster
            cluster_counts["fdr_bh"] = bh_fdr(np.array(pvals_cluster, dtype=float))

            cluster_meta = gene_clusters.groupby("cluster_id").agg({
                "cluster_label": "first",
                "freq_lasso": "max",
                "freq_rf": "max",
            }).reset_index()
            meta_map = {int(row["cluster_id"]): row for row in cluster_meta.to_dict("records")}
            genes_map = cluster_presence.groupby("cluster_id")["canonical_gene"].apply(
                lambda s: ",".join(sorted(set(s.astype(str))))
            ).to_dict()

            cluster_rows: List[Dict[str, object]] = []
            for row in cluster_counts.to_dict("records"):
                cid = int(row["cluster_id"])
                meta = meta_map.get(cid, {})
                cluster_rows.append({
                    **row,
                    "cluster_label": meta.get("cluster_label", f"cluster_{cid}"),
                    "freq_lasso": meta.get("freq_lasso", np.nan),
                    "freq_rf": meta.get("freq_rf", np.nan),
                    "genes": genes_map.get(cid, ""),
                })
            cluster_table = pd.DataFrame(cluster_rows, columns=cluster_cols)
            if "cluster_id" in cluster_table.columns:
                cluster_table["cluster_id"] = cluster_table["cluster_id"].astype("Int64")

    if not gene_table.empty:
        gene_table = gene_table.sort_values("fdr_bh", na_position="last")
    if not cluster_table.empty:
        cluster_table = cluster_table.sort_values("fdr_bh", na_position="last")
    return gene_table, cluster_table


def _numeric_ground_truth_summary(
    presence: pd.DataFrame,
    per_genome: pd.DataFrame,
    gene_clusters: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gene_cols = [
        "canonical_gene",
        "n_present",
        "mean_present",
        "n_absent",
        "mean_absent",
        "delta_mean",
        "cluster_id",
        "cluster_label",
        "freq_lasso",
        "freq_rf",
    ]
    cluster_cols = [
        "cluster_id",
        "cluster_label",
        "n_present",
        "mean_present",
        "n_absent",
        "mean_absent",
        "delta_mean",
        "freq_lasso",
        "freq_rf",
        "genes",
    ]

    labels = per_genome[["gff_filename", "phenotype_value"]].copy()
    labels = labels.dropna(subset=["phenotype_value"])
    if labels.empty:
        return pd.DataFrame(columns=gene_cols), pd.DataFrame(columns=cluster_cols)

    labels = labels.drop_duplicates(subset=["gff_filename"])
    labels["gff_filename"] = labels["gff_filename"].astype("string")

    presence_labeled = presence.merge(labels, on="gff_filename", how="inner")
    presence_labeled = presence_labeled.drop_duplicates(["canonical_gene", "gff_filename"])
    if presence_labeled.empty:
        return pd.DataFrame(columns=gene_cols), pd.DataFrame(columns=cluster_cols)

    label_map = labels.set_index("gff_filename")["phenotype_value"].astype(float)
    gene_rows: List[Dict[str, object]] = []
    for gene, sub in presence_labeled.groupby("canonical_gene", sort=False):
        present_ids = set(sub["gff_filename"].astype(str))
        with_vals = label_map.loc[list(present_ids)].to_numpy(dtype=float)
        without_ids = [g for g in label_map.index if g not in present_ids]
        without_vals = label_map.loc[without_ids].to_numpy(dtype=float) if without_ids else np.array([], dtype=float)
        mean_present = float(np.nanmean(with_vals)) if with_vals.size else float("nan")
        mean_absent = float(np.nanmean(without_vals)) if without_vals.size else float("nan")
        delta = mean_present - mean_absent if not (math.isnan(mean_present) or math.isnan(mean_absent)) else float("nan")
        gene_rows.append({
            "canonical_gene": gene,
            "n_present": len(present_ids),
            "mean_present": mean_present,
            "n_absent": len(without_ids),
            "mean_absent": mean_absent,
            "delta_mean": delta,
        })

    gene_table = pd.DataFrame(gene_rows)
    for col in ("cluster_id", "cluster_label", "freq_lasso", "freq_rf"):
        if col not in gene_table.columns:
            gene_table[col] = pd.NA if col in {"cluster_id", "cluster_label"} else np.nan
    if not gene_clusters.empty:
        gene_table = gene_table.merge(gene_clusters, on="canonical_gene", how="left", suffixes=("", "_gt"))
        for col in ("cluster_id", "cluster_label", "freq_lasso", "freq_rf"):
            gt_col = f"{col}_gt"
            if gt_col in gene_table.columns:
                gene_table[col] = gene_table[gt_col]
                gene_table = gene_table.drop(columns=[gt_col])
        if "cluster_id" in gene_table.columns:
            gene_table["cluster_id"] = gene_table["cluster_id"].astype("Int64")

    cluster_table = pd.DataFrame(columns=cluster_cols)
    if not gene_clusters.empty:
        cluster_presence = presence_labeled.merge(
            gene_clusters[["canonical_gene", "cluster_id", "cluster_label", "freq_lasso", "freq_rf"]],
            on="canonical_gene",
            how="left",
        ).dropna(subset=["cluster_id"])
        if not cluster_presence.empty:
            cluster_presence["cluster_id"] = cluster_presence["cluster_id"].astype("Int64")
            cluster_groups = cluster_presence.groupby("cluster_id", sort=False)
            cluster_meta = gene_clusters.groupby("cluster_id").agg({
                "cluster_label": "first",
                "freq_lasso": "max",
                "freq_rf": "max",
            }).reset_index()
            meta_map = {int(row["cluster_id"]): row for row in cluster_meta.to_dict("records")}
            genes_map = cluster_presence.groupby("cluster_id")["canonical_gene"].apply(
                lambda s: ",".join(sorted(set(s.astype(str))))
            ).to_dict()
            rows: List[Dict[str, object]] = []
            for cid, sub in cluster_groups:
                present_ids = set(sub["gff_filename"].astype(str))
                with_vals = label_map.loc[list(present_ids)].to_numpy(dtype=float)
                without_ids = [g for g in label_map.index if g not in present_ids]
                without_vals = label_map.loc[without_ids].to_numpy(dtype=float) if without_ids else np.array([], dtype=float)
                mean_present = float(np.nanmean(with_vals)) if with_vals.size else float("nan")
                mean_absent = float(np.nanmean(without_vals)) if without_vals.size else float("nan")
                delta = mean_present - mean_absent if not (math.isnan(mean_present) or math.isnan(mean_absent)) else float("nan")
                meta = meta_map.get(int(cid), {})
                rows.append({
                    "cluster_id": int(cid),
                    "cluster_label": meta.get("cluster_label", f"cluster_{cid}"),
                    "n_present": len(present_ids),
                    "mean_present": mean_present,
                    "n_absent": len(without_ids),
                    "mean_absent": mean_absent,
                    "delta_mean": delta,
                    "freq_lasso": meta.get("freq_lasso", np.nan),
                    "freq_rf": meta.get("freq_rf", np.nan),
                    "genes": genes_map.get(cid, ""),
                })
            cluster_table = pd.DataFrame(rows, columns=cluster_cols)
            if "cluster_id" in cluster_table.columns:
                cluster_table["cluster_id"] = cluster_table["cluster_id"].astype("Int64")

    if not gene_table.empty:
        gene_table = gene_table.sort_values("delta_mean", ascending=False, na_position="last")
    if not cluster_table.empty:
        cluster_table = cluster_table.sort_values("delta_mean", ascending=False, na_position="last")
    return gene_table, cluster_table


def _categorical_ground_truth_summary(
    presence: pd.DataFrame,
    per_genome: pd.DataFrame,
    gene_clusters: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gene_cols = [
        "canonical_gene",
        "phenotype_group",
        "n_genomes_with_gene",
        "group_total",
        "gene_total",
        "prop_within_group",
        "prop_within_gene",
        "cluster_id",
        "cluster_label",
        "freq_lasso",
        "freq_rf",
    ]
    cluster_cols = [
        "cluster_id",
        "cluster_label",
        "phenotype_group",
        "n_genomes_with_cluster",
        "group_total",
        "cluster_total",
        "prop_within_group",
        "prop_within_cluster",
        "freq_lasso",
        "freq_rf",
        "genes",
    ]

    label_col = "PhenotypeGroup" if "PhenotypeGroup" in per_genome.columns else "phenotype_value"
    labels = per_genome[["gff_filename", label_col]].copy()
    labels = labels.dropna(subset=[label_col])
    if labels.empty:
        return pd.DataFrame(columns=gene_cols), pd.DataFrame(columns=cluster_cols)

    labels = labels.drop_duplicates(subset=["gff_filename"]).rename(columns={label_col: "phenotype_group"})
    labels["gff_filename"] = labels["gff_filename"].astype("string")

    presence_labeled = presence.merge(labels, on="gff_filename", how="inner")
    presence_labeled = presence_labeled.drop_duplicates(["canonical_gene", "gff_filename"])
    if presence_labeled.empty:
        return pd.DataFrame(columns=gene_cols), pd.DataFrame(columns=cluster_cols)

    group_totals = labels.groupby("phenotype_group")["gff_filename"].nunique().rename("group_total")
    gene_totals = presence_labeled.groupby("canonical_gene")["gff_filename"].nunique().rename("gene_total")

    counts = (
        presence_labeled.groupby(["canonical_gene", "phenotype_group"])["gff_filename"].nunique().reset_index(name="n_genomes_with_gene")
    )
    counts = counts.merge(group_totals.reset_index(), on="phenotype_group", how="left")
    counts = counts.merge(gene_totals.reset_index(), on="canonical_gene", how="left")
    counts["prop_within_group"] = counts["n_genomes_with_gene"] / counts["group_total"].replace({0: np.nan})
    counts["prop_within_gene"] = counts["n_genomes_with_gene"] / counts["gene_total"].replace({0: np.nan})

    gene_table = counts
    for col in ("cluster_id", "cluster_label", "freq_lasso", "freq_rf"):
        if col not in gene_table.columns:
            gene_table[col] = pd.NA if col in {"cluster_id", "cluster_label"} else np.nan
    if not gene_clusters.empty:
        gene_table = gene_table.merge(gene_clusters, on="canonical_gene", how="left", suffixes=("", "_gt"))
        for col in ("cluster_id", "cluster_label", "freq_lasso", "freq_rf"):
            gt_col = f"{col}_gt"
            if gt_col in gene_table.columns:
                gene_table[col] = gene_table[gt_col]
                gene_table = gene_table.drop(columns=[gt_col])
        if "cluster_id" in gene_table.columns:
            gene_table["cluster_id"] = gene_table["cluster_id"].astype("Int64")

    cluster_table = pd.DataFrame(columns=cluster_cols)
    if not gene_clusters.empty:
        cluster_presence = presence_labeled.merge(
            gene_clusters[["canonical_gene", "cluster_id", "cluster_label", "freq_lasso", "freq_rf"]],
            on="canonical_gene",
            how="left",
        ).dropna(subset=["cluster_id"])
        if not cluster_presence.empty:
            cluster_presence["cluster_id"] = cluster_presence["cluster_id"].astype("Int64")
            cluster_totals = cluster_presence.groupby("cluster_id")["gff_filename"].nunique().rename("cluster_total")
            cluster_counts = (
                cluster_presence.groupby(["cluster_id", "cluster_label", "phenotype_group"])["gff_filename"].nunique().reset_index(name="n_genomes_with_cluster")
            )
            cluster_counts = cluster_counts.merge(cluster_totals.reset_index(), on="cluster_id", how="left")
            cluster_counts = cluster_counts.merge(group_totals.reset_index(), on="phenotype_group", how="left")
            cluster_counts["prop_within_group"] = cluster_counts["n_genomes_with_cluster"] / cluster_counts["group_total"].replace({0: np.nan})
            cluster_counts["prop_within_cluster"] = cluster_counts["n_genomes_with_cluster"] / cluster_counts["cluster_total"].replace({0: np.nan})
            genes_map = cluster_presence.groupby("cluster_id")["canonical_gene"].apply(
                lambda s: ",".join(sorted(set(s.astype(str))))
            ).to_dict()
            cluster_counts["genes"] = cluster_counts["cluster_id"].map(genes_map).fillna("")
            cluster_counts = cluster_counts.merge(
                gene_clusters.groupby("cluster_id").agg({
                    "freq_lasso": "max",
                    "freq_rf": "max",
                }).reset_index(),
                on="cluster_id",
                how="left",
            )
            cluster_table = cluster_counts[cluster_cols]
            if "cluster_id" in cluster_table.columns:
                cluster_table["cluster_id"] = cluster_table["cluster_id"].astype("Int64")

    return gene_table, cluster_table


def compute_ground_truth_enrichment(
    df: pd.DataFrame,
    per_genome: pd.DataFrame,
    phenotype: str,
    mask_column: str,
    gene_clusters: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute ground-truth enrichment tables at gene and cluster level."""

    kind = "categorical"
    if "phenotype_kind" in per_genome.columns and not per_genome["phenotype_kind"].dropna().empty:
        kind = str(per_genome["phenotype_kind"].dropna().iloc[0])

    allowed_genes = set(gene_clusters["canonical_gene"].astype(str)) if not gene_clusters.empty else None
    presence = build_ground_truth_presence(df, mask_column, allowed_genes=allowed_genes)
    if presence.empty:
        print("[GroundTruth] No masked loci with canonical gene names; enrichment skipped.", flush=True)
        return pd.DataFrame(), pd.DataFrame()

    if kind == "binary":
        return _binary_ground_truth_enrichment(presence, per_genome, gene_clusters)
    if kind == "numeric":
        return _numeric_ground_truth_summary(presence, per_genome, gene_clusters)
    return _categorical_ground_truth_summary(presence, per_genome, gene_clusters)


def plot_ground_truth_volcano(
    gene_table: pd.DataFrame,
    outdir: str,
    base_label: str,
    pdf: Optional[PdfPages] = None,
) -> None:
    """Render a volcano plot for binary ground-truth enrichment results."""

    if gene_table.empty or "log2fc" not in gene_table.columns or "fdr_bh" not in gene_table.columns:
        return

    _init_plotting()
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    df = gene_table.copy()
    df = df.replace({"fdr_bh": {0.0: 1e-300}})
    df["neglog10_fdr"] = -np.log10(df["fdr_bh"].astype(float))
    df["significant"] = df["fdr_bh"] <= 0.05

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = np.where(df["significant"], "#1f77b4", "#bbbbbb")
    ax.scatter(df["log2fc"], df["neglog10_fdr"], c=colors, s=32, alpha=0.85, edgecolor="none")
    ax.axvline(0.0, color="#666666", lw=1, ls="--")
    ax.axhline(-np.log10(0.05), color="#999999", lw=1, ls=":")
    ax.set_xlabel(f"log2 fold-change (presence in {base_label}+ vs {base_label}-)")
    ax.set_ylabel("-log10 FDR")
    ax.set_title("Ground-truth gene enrichment")

    annot = df.sort_values(["fdr_bh", "log2fc"], ascending=[True, False]).head(10)
    for _, row in annot.iterrows():
        if not np.isfinite(row.get("neglog10_fdr", np.nan)):
            continue
        ax.text(
            row["log2fc"],
            row["neglog10_fdr"],
            str(row.get("canonical_gene", "")),
            fontsize=8,
            ha="center",
            va="bottom",
            rotation=30,
        )

    _savefig(fig, os.path.join(outdir, "plots", "ground_truth_gene_volcano"), pdf=pdf)
    plt.close(fig)


def log_choose(n: int, k: int) -> float:
    """Return log(C(n, k)) using lgamma for numerical stability."""

    try:
        n_int = int(n)
        k_int = int(k)
    except Exception:
        return float("-inf")

    if k_int < 0 or k_int > n_int:
        return float("-inf")
    if n_int < 0:
        return float("-inf")
    if k_int == 0 or k_int == n_int:
        return 0.0
    return math.lgamma(n_int + 1) - math.lgamma(k_int + 1) - math.lgamma(n_int - k_int + 1)


def hypergeom_p(a: int, b: int, c: int, d: int) -> float:
    # Compute probability of this table given fixed margins
    # margins: row sums r1=a+c, r2=b+d; col sums c1=a+b, c2=c+d; total n=r1+r2
    r1 = a + c
    r2 = b + d
    c1 = a + b
    c2 = c + d
    n = r1 + r2
    # P = [C(c1, a) * C(c2, c)] / C(n, r1)
    logp = log_choose(c1, a) + log_choose(c2, c) - log_choose(n, r1)
    return math.exp(logp)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    # Based on summing probabilities for all tables with probability <= observed table
    r1 = a + c
    r2 = b + d
    c1 = a + b
    # feasible range for a: max(0, c1 - r2) .. min(r1, c1)
    a_min = max(0, c1 - r2)
    a_max = min(r1, c1)

    p_obs = hypergeom_p(a, b, c, d)
    p_sum = 0.0
    for aa in range(a_min, a_max + 1):
        cc = r1 - aa
        bb = c1 - aa
        dd = r2 - bb
        p = hypergeom_p(aa, bb, cc, dd)
        if p <= p_obs + 1e-15:
            p_sum += p
    return min(1.0, p_sum)


def fisher_exact_one_sided_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact test P(X>=a) for enrichment in group 1 (Spore+).

    With fixed margins, sum hypergeometric probabilities from a to a_max.
    """
    r1 = a + c
    r2 = b + d
    c1 = a + b
    # feasible range for a: max(0, c1 - r2) .. min(r1, c1)
    a_max = min(r1, c1)
    p_obs = hypergeom_p(a, b, c, d)
    p_sum = 0.0
    for aa in range(a, a_max + 1):
        cc = r1 - aa
        bb = c1 - aa
        dd = r2 - bb
        p = hypergeom_p(aa, bb, cc, dd)
        if p <= p_obs + 1e-15 or aa >= a:  # include all >= a
            p_sum += p
    return min(1.0, p_sum)


def cmh_one_sided_greater(strata: List[Tuple[int, int, int, int]]) -> float:
    """Cochran–Mantel–Haenszel one-sided test for enrichment in group 1 across strata.

    strata: list of (a, b, c, d) per stratum, where rows are group 1 (Spore+) vs group 2 (Spore-)
    and columns are presence vs absence. Returns one-sided p-value using normal approx.

    If all strata are degenerate (no variation in rows or columns), falls back to pooled
    one-sided Fisher's exact test to avoid returning NaN.
    """
    import math
    if not strata:
        return float('nan')

    # Keep track of pooled counts for a fallback Fisher test
    pooled_a = 0
    pooled_b = 0
    pooled_c = 0
    pooled_d = 0

    num = 0.0
    var = 0.0
    has_informative_stratum = False
    for a, b, c, d in strata:
        pooled_a += a
        pooled_b += b
        pooled_c += c
        pooled_d += d
        n = a + b + c + d
        if n <= 1:
            continue
        row1 = a + b
        row2 = c + d
        col1 = a + c
        col2 = b + d
        # Skip degenerate strata with no variation in rows or columns
        if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
            continue
        has_informative_stratum = True
        exp_a = (row1 * col1) / n
        num += (a - exp_a)
        var_i = (row1 * row2 * col1 * col2) / (n * n * (n - 1))
        var += var_i
    if not has_informative_stratum or var <= 0:
        # Fallback: pooled one-sided Fisher's exact test
        try:
            return fisher_exact_one_sided_greater(int(pooled_a), int(pooled_b), int(pooled_c), int(pooled_d))
        except Exception:
            return float('nan')
    z = num / math.sqrt(var)
    # one-sided greater (enrichment in group 1): p = 1 - Phi(z)
    p = 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))
    return max(0.0, min(1.0, p))


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR that is robust to NaNs.

    - Computes FDR only on finite p-values in [0,1].
    - Returns NaN for inputs that are NaN or not finite.
    """
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    if not mask.any():
        return out
    p_valid = p[mask]
    n = len(p_valid)
    order = np.argsort(p_valid)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q = p_valid * n / ranks
    # enforce monotonicity on sorted q-values
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    q_out = np.empty_like(p_valid)
    q_out[order] = np.clip(q_sorted, 0, 1)
    out[mask] = q_out
    return out


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Analyze phenotype-specific signal in processed GFFs")
    ap.add_argument("--input", required=True, help="Directory with processed GFF tables (parquet/csv)")
    ap.add_argument("--outdir", default="phenotype/outputs", help="Base output directory (phenotype subfolder appended automatically)")
    ap.add_argument("--phenotype", type=str, required=True, help="Phenotype column name to analyze")
    ap.add_argument("--phenotype_type", type=str, default="auto", choices=["auto", "binary", "categorical", "numeric"], help="Phenotype type override")
    ap.add_argument("--test_glob", default=None, help="Optional glob on gff_filename to define test set (e.g., '*test*')")
    ap.add_argument("--windows", type=int, default=10_000, help="Number of sampled windows")
    ap.add_argument("--window_size", type=int, default=1_000_000, help="Window size in bp")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--skip_dedup", action="store_true", default=True, help="Skip deduplication step (already done in processing)")
    ap.add_argument(
        "--exclude_fasta_dir",
        default=str(DATA_ROOT / "test"),
        help="Directory of FASTA files to exclude (drop rows whose fasta_filename matches). Set empty string to disable.",
    )
    args = ap.parse_args()

    phenotype = args.phenotype
    slug = phenotype_to_slug(phenotype)
    mask_column = f"gt_{slug}"
    args.outdir = os.path.join(args.outdir, slug)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Phenotype: {phenotype} | mask column: {mask_column}", flush=True)
    print(f"Outputs will be saved to: {os.path.abspath(args.outdir)}", flush=True)

    print("Reading tables...", flush=True)
    mask_column = f"gt_{phenotype_to_slug(args.phenotype)}"
    required_columns = build_required_columns(mask_column, phenotype_col=args.phenotype)
    df = read_all_tables(args.input, required_columns=required_columns)
    print("Validating schema...", flush=True)
    validate_schema(df, mask_column=mask_column, phenotype_col=args.phenotype)
    print("Coercing types...", flush=True)
    df = coerce_types(df, mask_column=mask_column, phenotype_col=args.phenotype)

    # Prepare mask flag and phenotype

    if mask_column not in df.columns:
        raise ValueError(
            f"Processed GFF tables in {args.input} do not contain column '{mask_column}'. "
            "Ensure process_gff.py was rerun with phenotype ground-truth masks."
        )
    df["mask_flag"] = df[mask_column].fillna(False).astype(bool)

    # Apply default split policy:
    #  - If a retention rule is provided (retained_flags set), we will restrict PLOTTING to test only later.
    #  - If no retention rule, exclude test from all analyses/plots by default to avoid leakage.
    retain_mode = bool(getattr(args, "retained_flags", False))
    if not retain_mode:
        if args.exclude_fasta_dir is not None and str(args.exclude_fasta_dir).strip() != "":
            df = exclude_fastas_in_dir(df, args.exclude_fasta_dir)

    # optional subset to test set (rarely used; typically keep None when deriving rules)
    if args.test_glob:
        print(f"Applying test glob filter: {args.test_glob}", flush=True)
        df = apply_test_glob(df, args.test_glob)
        print(f"After filter shape={df.shape}", flush=True)

    # deduplicate possible duplicates (skip by default since processing merged Prokka/Prodigal)
    if args.skip_dedup:
        print("Skipping deduplication (assumed already done during processing)", flush=True)
    else:
        print("Deduplicating loci...", flush=True)
        before = df.shape[0]
        df = deduplicate_loci(df)
        after = df.shape[0]
        print(f"Deduplicated rows: {before} -> {after}", flush=True)

    # 1) Build per-genome phenotype labels
    per_genome_labels = build_per_genome_labels(df, phenotype_col=args.phenotype, phenotype_type=args.phenotype_type, base_label=phenotype)
    phenotype_kind = "categorical"
    if not per_genome_labels.empty and per_genome_labels["phenotype_kind"].notna().any():
        phenotype_kind = str(per_genome_labels["phenotype_kind"].dropna().iloc[0])
    if phenotype_kind == "binary":
        _print_within_order_variation(
            per_genome_labels,
            base_label=phenotype,
            context="Main (initial)",
        )

    # 2) Ground-truth enrichment for canonical genes and clusters
    print("Loading ground-truth gene clusters...", flush=True)
    gt_clusters = load_ground_truth_gene_clusters(phenotype)
    if gt_clusters.empty:
        print("[GroundTruth] No cluster metadata available; falling back to canonical gene names only.", flush=True)
    else:
        n_clusters = gt_clusters["cluster_id"].nunique(dropna=True)
        print(
            f"[GroundTruth] Loaded {len(gt_clusters)} canonical genes across {n_clusters} clusters.",
            flush=True,
        )

    print("Computing ground-truth enrichment summaries...", flush=True)
    gene_enrich, cluster_enrich = compute_ground_truth_enrichment(
        df,
        per_genome_labels,
        phenotype,
        mask_column,
        gt_clusters,
    )
    gene_enrich.to_csv(os.path.join(args.outdir, "ground_truth_gene_enrichment.csv"), index=False)
    cluster_enrich.to_csv(os.path.join(args.outdir, "ground_truth_cluster_enrichment.csv"), index=False)

    if not gene_enrich.empty and "fdr_bh" in gene_enrich.columns:
        sig_genes = int((gene_enrich["fdr_bh"] <= 0.05).sum())
        print(
            f"Ground-truth genes evaluated: {len(gene_enrich)}; significant (FDR<=0.05): {sig_genes}",
            flush=True,
        )
    else:
        print(f"Ground-truth genes evaluated: {len(gene_enrich)}", flush=True)

    # 3) No curated relabeling; analyses use df["mask_flag"]

    # 4) Recompute per-genome metrics from the relabeled (and possibly filtered) table
    per_genome, genome_cache = per_genome_metrics(df)
    per_genome.to_csv(os.path.join(args.outdir, "per_genome_metrics.csv"), index=False)

    if phenotype_kind == "binary":
        _print_within_order_variation(
            per_genome_labels,
            base_label=phenotype,
            context="Main (post)",
        )

    # 6) Window sampling (over selected genomes; if test_glob supplied, it's already filtered)
    print("Building sampling frame...", flush=True)
    sf = build_sampling_frame(genome_cache, allowed_genomes=None, window_size=args.window_size)
    print(f"Sampling {args.windows} windows of size {args.window_size} bp...", flush=True)
    win = sample_windows(sf, W=args.windows, window_size=args.window_size, seed=args.seed)
    win.to_csv(os.path.join(args.outdir, "window_samples.csv"), index=False)
    win_summary = summarize_windows(win)
    win_summary.to_csv(os.path.join(args.outdir, "window_summary.csv"), index=False)

    # 7) NND distribution (pooled across genomes) using mask_flag
    print("Computing NND distributions...", flush=True)
    nnd_rows = []
    for gff, df_g in df.groupby("gff_filename", sort=False):
        sp = df_g[df_g["mask_flag"].fillna(False)]
        for seqid, sub in sp.groupby("seqid", sort=False):
            valid = sub["start"].notna() & sub["end"].notna()
            if not valid.any():
                continue
            starts = sub.loc[valid, "start"].to_numpy(dtype=np.int64)
            ends = sub.loc[valid, "end"].to_numpy(dtype=np.int64)
            mids = ((starts + ends) // 2)
            if len(mids) < 2:
                continue
            mids.sort()
            left = np.empty_like(mids)
            right = np.empty_like(mids)
            left[0] = np.iinfo(mids.dtype).max
            left[1:] = mids[1:] - mids[:-1]
            right[-1] = np.iinfo(mids.dtype).max
            right[:-1] = mids[1:] - mids[:-1]
            nn = np.minimum(left, right)
            for d in nn:
                nnd_rows.append({"gff_filename": gff, "seqid": seqid, "distance_bp": int(d)})
    nnd_df = pd.DataFrame(nnd_rows)
    nnd_df.to_csv(os.path.join(args.outdir, "nnd_distances.csv"), index=False)

    # 8) Phenotype contrast on relabeled per_genome
    print("Computing phenotype contrast...", flush=True)
    contrast = phenotype_contrast(per_genome_labels.merge(per_genome, on="gff_filename", how="left"), base_label=phenotype)
    contrast.to_csv(os.path.join(args.outdir, "phenotype_contrast.csv"), index=False)

    # Plots
    print("Generating plots...", flush=True)
    # If retention mode: restrict plotting to test set only; else plot the current df/per_genome
    if retain_mode and args.exclude_fasta_dir and str(args.exclude_fasta_dir).strip() != "":
        try:
            # Filter per_genome and windows/NND to test-only based on fasta_filename mapping
            # Map gff -> any test fasta observed in df rows (robust if mixed)
            df_test_only = include_fastas_in_dir(df, args.exclude_fasta_dir)
            if not df_test_only.empty:
                per_genome_test, _ = per_genome_metrics(df_test_only)
                per_genome_test = per_genome_labels.merge(per_genome_test, on="gff_filename", how="right")
                plot_per_genome_metrics(per_genome_test, args.outdir, pdf=None, base_label=phenotype)
                # Windows are genome-agnostic samples; replot with test-only labels if available
                plot_windows(win, per_genome_test, args.outdir, pdf=None, base_label=phenotype)
                # Recompute NND for test only for plotting
                nnd_rows_plot = []
                for gff, df_g in df_test_only.groupby("gff_filename", sort=False):
                    sp = df_g[df_g["mask_flag"].fillna(False)]
                    for seqid, sub in sp.groupby("seqid", sort=False):
                        valid = sub["start"].notna() & sub["end"].notna()
                        if not valid.any():
                            continue
                        starts = sub.loc[valid, "start"].to_numpy(dtype=np.int64)
                        ends = sub.loc[valid, "end"].to_numpy(dtype=np.int64)
                        mids = ((starts + ends) // 2)
                        if len(mids) < 2:
                            continue
                        mids.sort()
                        left = np.empty_like(mids)
                        right = np.empty_like(mids)
                        left[0] = np.iinfo(mids.dtype).max
                        left[1:] = mids[1:] - mids[:-1]
                        right[-1] = np.iinfo(mids.dtype).max
                        right[:-1] = mids[1:] - mids[:-1]
                        nn = np.minimum(left, right)
                        for d in nn:
                            nnd_rows_plot.append({"gff_filename": gff, "seqid": seqid, "distance_bp": int(d)})
                nnd_df_plot = pd.DataFrame(nnd_rows_plot)
                plot_nnd(nnd_df_plot, per_genome_test, args.outdir, pdf=None, base_label=phenotype)
            else:
                # Fallback to original plotting if filtering produced empty
                plot_per_genome_metrics(per_genome, args.outdir, pdf=None, base_label=phenotype)
                plot_windows(win, per_genome, args.outdir, pdf=None, base_label=phenotype)
                plot_nnd(nnd_df, per_genome, args.outdir, pdf=None, base_label=phenotype)
        except Exception:
            plot_per_genome_metrics(per_genome, args.outdir, pdf=None, base_label=phenotype)
            plot_windows(win, per_genome, args.outdir, pdf=None, base_label=phenotype)
            plot_nnd(nnd_df, per_genome, args.outdir, pdf=None, base_label=phenotype)
    else:
        # Default plotting
        per_genome_plot = per_genome_labels.merge(per_genome, on="gff_filename", how="right")
        plot_per_genome_metrics(per_genome_plot, args.outdir, pdf=None, base_label=phenotype)
        plot_windows(win, per_genome_plot, args.outdir, pdf=None, base_label=phenotype)
        plot_nnd(nnd_df, per_genome_plot, args.outdir, pdf=None, base_label=phenotype)

    if phenotype_kind == "binary":
        plot_ground_truth_volcano(gene_enrich, args.outdir, base_label=phenotype, pdf=None)

    plot_spore_gene_lengths(df, args.outdir, pdf=None)

    # Write markdown summary with statistical results/tables
    summary_path = os.path.join(args.outdir, "analysis_summary.md")
    with open(summary_path, "w", encoding="utf-8") as md:
        md.write(f"# Phenotype analysis summary: {phenotype}\n\n")
        md.write(f"Ground-truth mask column: {mask_column}\n\n")
        md.write(f"Input dir: {os.path.abspath(args.input)}\n\n")
        md.write(f"Genomes: {df['gff_filename'].nunique()}, total rows: {len(df)}\n\n")
        md.write("## Phenotype contrast (effect sizes)\n\n")
        md.write(contrast.to_string(index=False))
        md.write("\n\n## Ground-truth gene enrichment\n\n")
        if gene_enrich.empty:
            md.write("No ground-truth gene enrichment results available.\n")
        else:
            if phenotype_kind == "binary" and "fdr_bh" in gene_enrich.columns:
                md.write(gene_enrich.sort_values("fdr_bh").head(25).to_string(index=False))
            elif phenotype_kind == "numeric" and "delta_mean" in gene_enrich.columns:
                md.write(gene_enrich.sort_values("delta_mean", ascending=False).head(25).to_string(index=False))
            elif phenotype_kind == "categorical" and "prop_within_group" in gene_enrich.columns:
                md.write(gene_enrich.sort_values("prop_within_group", ascending=False).head(25).to_string(index=False))
            else:
                md.write(gene_enrich.head(25).to_string(index=False))

        md.write("\n\n## Ground-truth cluster enrichment\n\n")
        if cluster_enrich.empty:
            md.write("No ground-truth cluster enrichment results available.\n")
        else:
            if phenotype_kind == "binary" and "fdr_bh" in cluster_enrich.columns:
                md.write(cluster_enrich.sort_values("fdr_bh").head(25).to_string(index=False))
            elif phenotype_kind == "numeric" and "delta_mean" in cluster_enrich.columns:
                md.write(cluster_enrich.sort_values("delta_mean", ascending=False).head(25).to_string(index=False))
            elif phenotype_kind == "categorical" and "prop_within_group" in cluster_enrich.columns:
                md.write(cluster_enrich.sort_values("prop_within_group", ascending=False).head(25).to_string(index=False))
            else:
                md.write(cluster_enrich.head(25).to_string(index=False))

    print(f"Done. Outputs for phenotype '{phenotype}' written to {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
