"""
Analyze spore-related signal in processed GFFs.

Input: a directory of processed GFF tables (parquet/csv) with columns
  seqid, start, end, strand, locus_tag, gene, Name, product, inference,
  protein_id, sources, spore_related, gff_filename, fasta_filename,
  Phylum, Class, Order, Family, Genus, Species, Spore formation

Key analyses (computed on a per-genome basis, where genome ≡ gff_filename):
  1) p_g: fraction of spore-related loci among all loci.
  2) Coverage by bp: union length of spore-related intervals / genome length (sum of contig max end).
     Also report coverage within coding (spore-union / all-loci-union).
  3) Window sampling: draw W windows of size S (default W=10_000, S=1_000_000 bp)
     from selected genomes; compute spore bp fraction per window.
  4) Nearest-neighbor distances between spore loci midpoints (per contig), pooled and by genome.
  5) Phenotype contrast: compare genomes with phenotype Spore formation==True vs False
     using p_g, coverage metrics, N_spore, median NND; report Cohen's d, Cliff's delta, AUROC.
  6) Ambiguous gene families: simple family detection via regex over gene/Name/product; compute
     presence enrichment between phenotype groups, Fisher's exact p, BH-FDR, log2 fold-change.

Outputs (CSV unless stated otherwise):
  - per_genome_metrics.csv
  - window_samples.csv (W rows)
  - window_summary.csv (mean, sd, quantiles)
  - nnd_distances.csv (all NNDs; columns: gff_filename, seqid, distance_bp)
  - phenotype_contrast.csv (effect sizes)
  - family_enrichment_data_driven.csv (data-driven term enrichment summary)

Usage:
  python analyze_sporulation.py \
      --input processed_gff \
      --outdir analysis_out \
      --windows 10000 --window_size 1000000 \
      --test_glob "*test*"  # optional: subset genomes by gff_filename glob

Notes:
  * Assumes each row ≈ a coding locus; if both gene/CDS rows were retained, we deduplicate
    to one row per (gff_filename, locus_tag) preferring records with protein_id.
  * Genome length is estimated as sum over contigs of max(end) (typical for GFF-derived lengths).
  * All computations are performed with the existing boolean 'spore_related'.
  * Only standard libs + pandas/numpy are used (no SciPy/sklearn).
  * Optional filtering: if a retained gene flags CSV (product of retained_spore_genes_analysis.py)
    is provided via --retained_flags, then all analyses except the data-driven family analyses
    are restricted to loci whose canonical gene name passes the chosen rule (--retained_rule in {a,b,c}).
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
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
    desired = set(required_columns or REQUIRED_COLUMNS)
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

REQUIRED_COLUMNS = [
    "seqid", "start", "end", "strand", "locus_tag", "gene", "Name", "product",
    "inference", "protein_id", "sources", "spore_related", "gff_filename",
    "fasta_filename", "Phylum", "Class", "Order", "Family", "Genus", "Species",
    "Spore formation",
]


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # Core integer coordinates
    for c in ("start", "end"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    # Booleans
    for c in ("spore_related", "Spore formation"):
        if c in df.columns:
            df[c] = _coerce_series_to_boolean(df[c])
    # Strings
    for c in [
        "seqid", "strand", "locus_tag", "gene", "Name", "product", "inference",
        "protein_id", "sources", "gff_filename", "fasta_filename", "Phylum", "Class",
        "Order", "Family", "Genus", "Species",
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


# --------------------------- Retained gene filtering ---------------------------

def _load_retained_allowed_genes(retained_csv: str, rule: str) -> Set[str]:
    """Read retained gene flags CSV and return the set of canonical gene names allowed by rule.

    rule in {"a","b","c"} maps to columns pass_a_family_sig, pass_b_nt_pid, pass_c_both.
    """
    df = pd.read_csv(retained_csv)
    if "gene_canonical" not in df.columns:
        # Backward compatibility: allow 'gene_canon'
        if "gene_canon" in df.columns:
            df = df.rename(columns={"gene_canon": "gene_canonical"})
        else:
            raise ValueError("retained flags CSV missing 'gene_canonical' column")
    col_map = {"a": "pass_a_family_sig", "b": "pass_b_nt_pid", "c": "pass_c_both"}
    flag_col = col_map.get(rule)
    if flag_col not in df.columns:
        raise ValueError(f"retained flags CSV missing required column: {flag_col}")
    sub = df[df[flag_col] == True]
    allowed = set(sub["gene_canonical"].astype(str).dropna().tolist())
    return allowed


# (Deprecated) _apply_retained_filter removed; retained filtering is now done via
# _set_spore_related_from_allowed_genes without dropping rows.


# New: redefine spore_related from allowed genes WITHOUT dropping any rows
def _set_spore_related_from_allowed_genes(df: pd.DataFrame, allowed_genes: Set[str]) -> pd.DataFrame:
    if not allowed_genes:
        print("[Retained] Allowed gene set is empty; leaving spore_related unchanged.", flush=True)
        return df
    d = df.copy()
    if "gene" in d.columns:
        d["_gene_canon"] = d["gene"].apply(canonical_gene_name)
    else:
        d["_gene_canon"] = pd.NA
    if "Name" in d.columns:
        d["_name_canon"] = d["Name"].apply(canonical_gene_name)
    else:
        d["_name_canon"] = pd.NA
    new_flag = d["_gene_canon"].isin(allowed_genes) | d["_name_canon"].isin(allowed_genes)
    prev_true = int(d["spore_related"].fillna(False).sum()) if "spore_related" in d.columns else 0
    d["spore_related"] = new_flag.astype("boolean")
    now_true = int(d["spore_related"].fillna(False).sum())
    print(f"[Retained] spore_related reassigned from allowed genes. True: {now_true} (was {prev_true}).", flush=True)
    return d.drop(columns=["_gene_canon", "_name_canon"], errors="ignore")



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
    spore_union: Dict[str, List[Tuple[int, int]]]  # seqid -> merged intervals for spore loci


def build_genome_intervals(df_g: pd.DataFrame) -> GenomeIntervals:
    # Robust contig lengths: max numeric end per contig; treat missing as 0
    ends_numeric = pd.to_numeric(df_g["end"], errors="coerce")
    contig_len_series = ends_numeric.groupby(df_g["seqid"]).max().fillna(0).astype(int)
    contig_lengths: Dict[str, int] = contig_len_series.to_dict()
    all_union: Dict[str, List[Tuple[int, int]]] = {}
    spore_union: Dict[str, List[Tuple[int, int]]] = {}
    for seqid, sub in df_g.groupby("seqid", sort=False):
        # Filter out rows with missing coordinates to avoid int conversion errors
        valid_coords = sub["start"].notna() & sub["end"].notna()
        starts_all = sub.loc[valid_coords, "start"].to_numpy(dtype=np.int64)
        ends_all = sub.loc[valid_coords, "end"].to_numpy(dtype=np.int64)
        all_union[seqid] = merge_intervals(starts_all, ends_all)
        m = sub["spore_related"].fillna(False).to_numpy()
        if m.any():
            sel = valid_coords & m
            starts_sp = sub.loc[sel, "start"].to_numpy(dtype=np.int64)
            ends_sp = sub.loc[sel, "end"].to_numpy(dtype=np.int64)
            spore_union[seqid] = merge_intervals(starts_sp, ends_sp)
        else:
            spore_union[seqid] = []
    return GenomeIntervals(contig_lengths, all_union, spore_union)


def per_genome_metrics(df: pd.DataFrame, progress_every: int = 500) -> Tuple[pd.DataFrame, Dict[str, GenomeIntervals]]:
    rows = []
    genome_cache: Dict[str, GenomeIntervals] = {}

    groups = list(df.groupby("gff_filename", sort=False))
    total = len(groups)
    print(f"Computing per-genome metrics for {total} genomes...", flush=True)
    for idx, (gff, df_g) in enumerate(groups, start=1):
        gi = build_genome_intervals(df_g)
        genome_cache[gff] = gi

        L_genome = int(sum(gi.contig_lengths.values()))
        U_spore = int(sum(intervals_total_length(v) for v in gi.spore_union.values()))
        U_all = int(sum(intervals_total_length(v) for v in gi.all_union.values()))

        n_all = int(len(df_g))
        n_spore = int(df_g["spore_related"].fillna(False).sum())
        p_g = n_spore / n_all if n_all else 0.0

        cov_genome = U_spore / L_genome if L_genome else 0.0
        cov_coding = U_spore / U_all if U_all else 0.0

        # NND median per genome
        nnd_values = nearest_neighbor_distances(df_g)
        median_nnd = float(np.median(nnd_values)) if len(nnd_values) else np.nan

        # phenotype (single value per genome expected)
        pheno_vals = df_g["Spore formation"].dropna().unique()
        phenotype = pheno_vals[0] if len(pheno_vals) else pd.NA

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
            "Spore formation": phenotype,
            "n_all_loci": n_all,
            "n_spore_loci": n_spore,
            "p_spore_loci": p_g,
            "U_spore_bp": U_spore,
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
    sp = df_g[df_g["spore_related"].fillna(False)]
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


def build_per_genome_labels_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight per-genome mapping with only taxonomy and phenotype.

    Returns columns: gff_filename, Order, Spore formation.
    """
    rows = []
    for gff, df_g in df.groupby("gff_filename", sort=False):
        rows.append({
            "gff_filename": gff,
            "Order": first_nonnull(df_g["Order"]) if "Order" in df_g.columns else pd.NA,
            "Spore formation": first_nonnull(df_g["Spore formation"]) if "Spore formation" in df_g.columns else pd.NA,
        })
    return pd.DataFrame(rows)


def relabel_spore_related_from_terms(
    df: pd.DataFrame,
    fam_dd: pd.DataFrame,
    fdr_threshold: float = 0.05,
    min_token_len: int = 3,
    max_token_len: int = 30,
) -> pd.DataFrame:
    """Relabel spore_related: True only if row contains a significant data-driven term.

    Uses the same tokenization as family_enrichment_data_driven.
    """
    prev_true = int(df["spore_related"].fillna(False).sum()) if "spore_related" in df.columns else 0
    # Validate fam_dd
    if fam_dd is None or len(fam_dd) == 0 or ("family" not in fam_dd.columns) or ("fdr_bh" not in fam_dd.columns):
        print("[Relabel] No valid data-driven enrichment table (missing 'family'/'fdr_bh' or empty). Leaving spore_related unchanged.", flush=True)
        return df
    sig = set(
        fam_dd.loc[(pd.to_numeric(fam_dd["fdr_bh"], errors="coerce") <= float(fdr_threshold)) & fam_dd["family"].notna(), "family"]
        .astype(str)
        .str.lower()
        .tolist()
    )
    if not sig:
        print("[Relabel] No significant data-driven terms at FDR <= %.3f. Setting all spore_related to False." % fdr_threshold, flush=True)
        df = df.copy()
        df["spore_related"] = False
        print(f"[Relabel] spore_related True count: 0 (was {prev_true})", flush=True)
        return df

    cols = [c for c in ("gene", "Name", "product", "inference") if c in df.columns]
    if not cols:
        print("[Relabel] No annotation columns to scan; leaving spore_related unchanged.", flush=True)
        return df

    # Build a single compiled regex that matches any significant term as a full token
    # Token boundary is defined as non-[A-Za-z0-9_] on both sides
    escaped_terms = [re.escape(t) for t in sig]
    # Sort by length desc to help regex engine performance on alternations
    escaped_terms.sort(key=len, reverse=True)
    if not escaped_terms:
        print("[Relabel] No valid terms after escaping; setting all spore_related to False.", flush=True)
        df = df.copy()
        df["spore_related"] = False
        print(f"[Relabel] spore_related True count: 0 (was {prev_true})", flush=True)
        return df
    alternation = "|".join(escaped_terms)
    boundary = r"(?<![A-Za-z0-9_])(?:" + alternation + r")(?![A-Za-z0-9_])"
    try:
        pat = re.compile(boundary)
    except Exception as e:
        # Fallback to simple word-boundary if the pattern is too complex
        print(f"[Relabel] Warning: failed to compile boundary regex ({e}); falling back to word-boundary.", flush=True)
        pat = re.compile(r"\b(?:" + alternation + r")\b")

    # Vectorized scan across columns, OR-ed
    has_sig = pd.Series(False, index=df.index)
    for c in cols:
        s = df[c].astype("string").str.lower()
        m = s.str.contains(pat, regex=True, na=False)
        if m.any():
            has_sig = has_sig | m

    df = df.copy()
    df["spore_related"] = has_sig.astype("boolean")
    now_true = int(df["spore_related"].fillna(False).sum())
    print(f"[Relabel] Applied {len(escaped_terms)} significant terms (FDR<=%.3f). spore_related True: {now_true} (was {prev_true})." % fdr_threshold, flush=True)
    return df
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


def _phenotype_labels(s: pd.Series) -> pd.Series:
    """Map boolean/nullable 'Spore formation' to string labels safely.

    True -> 'Spore+', False -> 'Spore-', NA -> 'Unknown'.
    """
    s_bool = s.astype("boolean")
    return s_bool.map({True: "Spore+", False: "Spore-"}).fillna("Unknown")


def plot_per_genome_metrics(per_genome: pd.DataFrame, outdir: str, pdf: Optional[PdfPages] = None) -> None:
    _init_plotting()
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    df = per_genome.copy()
    df["Phenotype"] = _phenotype_labels(df["Spore formation"])

    metrics = [
        ("p_spore_loci", "Fraction spore loci"),
        ("cov_genome", "Spore bp / genome bp"),
        ("cov_coding", "Spore bp / coding bp"),
        ("n_spore_loci", "# spore loci"),
        ("median_nnd_bp", "Median NND (bp)"),
    ]
    for col, label in metrics:
        fig, ax = plt.subplots(figsize=(5, 4))
        if sns is not None:
            sns.violinplot(data=df, x="Phenotype", y=col, inner=None, cut=0, linewidth=0.8, ax=ax)
            sns.boxplot(data=df, x="Phenotype", y=col, whis=1.5, width=0.25, showcaps=True,
                        boxprops={"facecolor": "white"}, ax=ax)
            sns.stripplot(data=df, x="Phenotype", y=col, color="k", alpha=0.35, size=2, jitter=0.15, ax=ax)
        else:
            # Matplotlib fallback: simple boxplot
            df.boxplot(column=col, by="Phenotype", ax=ax)
            ax.get_figure().suptitle("")
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_title(label)
        _savefig(fig, os.path.join(outdir, "plots", f"metric_{col}"), pdf=pdf)


def plot_windows(win: pd.DataFrame, per_genome: pd.DataFrame, outdir: str, pdf: Optional[PdfPages] = None) -> None:
    if win.empty:
        return
    _init_plotting()
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    pheno = per_genome[["gff_filename", "Spore formation"]].copy()
    pheno["Phenotype"] = _phenotype_labels(pheno["Spore formation"])
    w = win.merge(pheno[["gff_filename", "Phenotype"]], on="gff_filename", how="left")

    fig, ax = plt.subplots(figsize=(6, 4))
    if sns is not None:
        sns.kdeplot(data=w, x="spore_frac", hue="Phenotype", fill=True, common_norm=False, alpha=0.3, ax=ax)
    else:
        for k, sub in w.groupby("Phenotype"):
            ax.hist(sub["spore_frac"], bins=50, alpha=0.4, label=str(k), density=True)
        ax.legend()
    ax.set_xlabel("Window spore bp fraction")
    ax.set_ylabel("Density")
    ax.set_title("Window spore fraction distribution")
    _savefig(fig, os.path.join(outdir, "plots", "windows_spore_frac_density"), pdf=pdf)

    # Add zero vs non-zero barplot by phenotype to highlight mass at zero
    w["is_zero"] = (w["spore_frac"] <= 0).astype(int)
    counts = (w.groupby(["Phenotype", "is_zero"]).size().reset_index(name="n"))
    totals = counts.groupby("Phenotype")["n"].transform("sum")
    counts["prop"] = counts["n"] / totals
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for ph in sorted(counts["Phenotype"].unique()):
        subc = counts[counts["Phenotype"] == ph]
        # order: zero (1) then non-zero (0) for consistent labeling
        subc = subc.sort_values("is_zero", ascending=False)
        bars = ax2.bar([f"{ph}\nzero", f"{ph}\nnonzero"], subc["prop"].to_list(), alpha=0.8)
        # annotate percentage on bars
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
    ax2.set_title("Zero vs non-zero window spore fraction")
    ax2.set_ylim(0, min(1.05, max(1.0, ax2.get_ylim()[1])))
    _savefig(fig2, os.path.join(outdir, "plots", "windows_zero_nonzero_bar"), pdf=pdf)


def plot_nnd(nnd_df: pd.DataFrame, per_genome: pd.DataFrame, outdir: str, pdf: Optional[PdfPages] = None) -> None:
    if nnd_df.empty:
        return
    _init_plotting()
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    pheno = per_genome[["gff_filename", "Spore formation"]].copy()
    pheno["Phenotype"] = _phenotype_labels(pheno["Spore formation"])
    d = nnd_df.merge(pheno[["gff_filename", "Phenotype"]], on="gff_filename", how="left")

    fig, ax = plt.subplots(figsize=(6, 4))
    if sns is not None:
        sns.ecdfplot(data=d, x="distance_bp", hue="Phenotype", ax=ax)
    else:
        # Matplotlib fallback: cumulative hist
        for k, sub in d.groupby("Phenotype"):
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


def plot_family_enrichment(fam: pd.DataFrame, outdir: str, label: str = "data_driven", pdf: Optional[PdfPages] = None) -> None:
    if fam.empty:
        return
    _init_plotting()
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    # Volcano-like scatter with significance and ambiguity highlighting
    f = fam.copy()
    # Show zero-FDR points by flooring at a tiny epsilon for -log10
    eps = 1e-300
    fdr_plot = f["fdr_bh"].copy()
    fdr_plot = fdr_plot.where(~(fdr_plot == 0), eps)
    f["neglog10_fdr"] = -np.log10(fdr_plot)
    # Significant threshold at FDR <= 0.05
    f["significant"] = f["fdr_bh"] <= 0.05
    # Ambiguous already present in table
    colors = np.where(f["ambiguous"], "#999999", np.where(f["significant"], "#1f77b4", "#cccccc"))
    sizes = np.where(f["significant"], 24, 12)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(f["log2FC_present"], f["neglog10_fdr"], s=sizes, c=colors, alpha=0.8, edgecolor="none")
    ax.axvline(0.0, color="gray", lw=1)
    ax.axhline(-np.log10(0.05), color="#666666", lw=1, ls=":")
    ax.set_xlabel("log2 Fold-change (presence Spore+ vs Spore-)")
    ax.set_ylabel("-log10 FDR")
    ax.set_title(f"Term enrichment (volcano) - {label}")
    _savefig(fig, os.path.join(outdir, "plots", f"family_enrichment_volcano_{label}"), pdf=pdf)

    # Top families barplot by significance
    top = f.sort_values(["fdr_bh", "log2FC_present"], ascending=[True, False]).head(20)
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(top))
    bar_colors = np.where(top["ambiguous"], "#bbbbbb", "#1f77b4")
    ax.barh(y, top["log2FC_present"].to_numpy(), color=bar_colors)
    ax.set_yticks(y)
    ax.set_yticklabels(top["family"].astype(str).tolist())
    ax.invert_yaxis()
    ax.set_xlabel("log2 Fold-change (Spore+ vs Spore-)")
    ax.set_title(f"Top enriched terms (gray = ambiguous) - {label}")
    _savefig(fig, os.path.join(outdir, "plots", f"family_enrichment_top_{label}"), pdf=pdf)


def plot_spore_gene_lengths(df: pd.DataFrame, outdir: str, pdf: Optional[PdfPages] = None) -> None:
    """Plot distribution of spore-related locus lengths by phenotype (if available)."""
    if df.empty or not {"start", "end"}.issubset(df.columns):
        return
    _init_plotting()
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    d = df[df["spore_related"].fillna(False)].copy()
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
    ax.set_xlabel("Spore-related locus length (bp)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of spore-related gene lengths")
    _savefig(fig, os.path.join(outdir, "plots", "spore_gene_length_distribution"), pdf=pdf)


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
    Columns: gff_filename, seqid, contig_len, spore_union (list of intervals)
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
                    "spore_union": gi.spore_union.get(seqid, []),
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
            "gff_filename", "seqid", "start", "end", "spore_bp", "spore_frac"
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
        sp = overlap_length((s, e), row["spore_union"]) if row["spore_union"] else 0
        sp_bp.append(sp)
    out = meta.assign(start=starts, end=ends)
    out["spore_bp"] = sp_bp
    out["spore_frac"] = out["spore_bp"] / float(window_size)
    return out


def summarize_windows(df_w: pd.DataFrame) -> pd.DataFrame:
    if df_w.empty:
        return pd.DataFrame([{"n": 0}])
    s = df_w["spore_frac"].to_numpy()
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


def _print_within_order_variation(per_genome: pd.DataFrame, context: str = "") -> None:
    """Print a brief summary indicating whether there is within-Order variation
    in phenotype (both Spore+ and Spore- present). Helps diagnose CMH degeneracy.
    """
    try:
        if "Order" not in per_genome.columns:
            print(f"[{context}] No 'Order' column; skipping within-clade variation check.", flush=True)
            return
        pg = per_genome[["Order", "Spore formation"]].copy()
        pg["Spore formation"] = pg["Spore formation"].astype("boolean")
        pg = pg.dropna(subset=["Spore formation"]).copy()
        if pg.empty:
            print(f"[{context}] No known phenotypes; skipping within-clade variation check.", flush=True)
            return
        counts = pg.groupby("Order")["Spore formation"].value_counts().unstack(fill_value=0)
        # Ensure both columns exist
        counts = counts.reindex(columns=[True, False], fill_value=0)
        n_orders = counts.shape[0]
        has_true = counts[True] > 0
        has_false = counts[False] > 0
        informative = int((has_true & has_false).sum())
        print(f"[{context}] Within-Order phenotype variation: {informative}/{n_orders} Orders have both phenotypes.", flush=True)
        non_info_orders = counts.index[~(has_true & has_false)].tolist()
        if non_info_orders:
            sample = ", ".join(map(str, non_info_orders[:10]))
            suffix = "" if len(non_info_orders) <= 10 else " ..."
            print(f"[{context}] Orders without both phenotypes (up to 10): {sample}{suffix}", flush=True)
    except Exception as e:
        print(f"[{context}] Warning: failed within-clade variation check: {e}", flush=True)


def phenotype_contrast(per_genome: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    mask_true = per_genome["Spore formation"].astype("boolean") == True
    mask_false = per_genome["Spore formation"].astype("boolean") == False

    metrics = [
        ("p_spore_loci", "Fraction spore loci"),
        ("cov_genome", "Spore bp / genome bp"),
        ("cov_coding", "Spore bp / coding bp"),
        ("n_spore_loci", "# spore loci"),
        ("median_nnd_bp", "Median NND (bp)"),
    ]
    for col, label in metrics:
        x = per_genome.loc[mask_true, col].to_numpy(dtype=float)
        y = per_genome.loc[mask_false, col].to_numpy(dtype=float)
        out_rows.append({
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
    return pd.DataFrame(out_rows)


# --------------------------- Gene family enrichment ---------------------------

FAMILY_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("sigma_F", re.compile(r"(?i)\b(?:sig|sigma)[ _-]?F\b")),
    ("sigma_E", re.compile(r"(?i)\b(?:sig|sigma)[ _-]?E\b")),
    ("sigma_G", re.compile(r"(?i)\b(?:sig|sigma)[ _-]?G\b")),
    ("sigma_K", re.compile(r"(?i)\b(?:sig|sigma)[ _-]?K\b")),
    ("SpoIIQ", re.compile(r"(?i)\bspoIIQ\b")),
    ("SpoIIIA", re.compile(r"(?i)\bspoIIIA[A-H]\b")),
    ("SpoIIIE", re.compile(r"(?i)\bspoIIIE\b")),
    ("SpoIID", re.compile(r"(?i)\bspoIID\b")),
    ("SpoIIP", re.compile(r"(?i)\bspoIIP\b")),
    ("SpoIIM", re.compile(r"(?i)\bspoIIM\b")),
    ("SpoIVA", re.compile(r"(?i)\bspoIVA\b")),
    ("SpoVID", re.compile(r"(?i)\bspoVID\b")),
    ("SpoVM",  re.compile(r"(?i)\bspoVM\b")),
    ("SafA",   re.compile(r"(?i)\bsafA\b")),
    ("DPA_dpaA", re.compile(r"(?i)\bdpaA\b|\bspoVFA\b")),
    ("DPA_dpaB", re.compile(r"(?i)\bdpaB\b|\bspoVFB\b")),
    ("DPA_spoVA", re.compile(r"(?i)\bspoVA(?:[CDEFGJ])?\b")),
    ("SASP_ssp", re.compile(r"(?i)\bssp[a-z]\b|\bSASP\b")),
    ("Spo0A",   re.compile(r"(?i)\bspo0A\b")),
    ("gerABC",  re.compile(r"(?i)\bger(?:A|B|C)(?:[A-Z])?\b")),
    ("cortex_lytic", re.compile(r"(?i)\bsleB\b|\bcwlJ\b")),
    ("cot",     re.compile(r"(?i)\bcot[a-z0-9]{1,3}\b")),
]


def _vectorized_family_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized family detection with prefilter.

    1) Prefilter rows using fast substring checks across relevant columns to greatly
       reduce the number of rows that run through full regex matching.
    2) For each family pattern, apply vectorized str.contains across columns and OR results.
    3) Return a de-duplicated presence table: (gff_filename, _family, present=True).
    """
    cols = [c for c in ("gene", "Name", "product", "inference") if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=["gff_filename", "_family", "present"])  # type: ignore

    # Restrict to spore-related rows for family detection
    sub_base = df[df["spore_related"].fillna(False)].copy()
    sub = sub_base.loc[:, ["gff_filename", *cols]].copy()
    for c in cols:
        sub[c] = sub[c].astype("string").str.lower()

    # Fast prefilter seeds to avoid running regex on most rows
    seeds = ["spo", "ssp", "cot", "sigma", "sig", "ger", "sleb", "cwlj", "dpa", "spov", "spoii"]
    any_seed = pd.Series(False, index=sub.index)
    for c in cols:
        m = pd.Series(False, index=sub.index)
        for s in seeds:
            m = m | sub[c].str.contains(s, na=False)
        any_seed = any_seed | m
    sub = sub.loc[any_seed]
    if sub.empty:
        return pd.DataFrame(columns=["gff_filename", "_family", "present"])  # type: ignore

    hits = []
    for fam, pat in FAMILY_PATTERNS:
        fam_mask = pd.Series(False, index=sub.index)
        for c in cols:
            fam_mask = fam_mask | sub[c].str.contains(pat, na=False)
        if fam_mask.any():
            fam_hits = sub.loc[fam_mask, ["gff_filename"]].copy()
            fam_hits["_family"] = fam
            fam_hits["present"] = True
            hits.append(fam_hits)

    if not hits:
        return pd.DataFrame(columns=["gff_filename", "_family", "present"])  # type: ignore

    out = pd.concat(hits, ignore_index=True)
    out = out.drop_duplicates(["gff_filename", "_family"])  # any occurrence counts as presence
    return out


def family_enrichment(df: pd.DataFrame, per_genome: pd.DataFrame) -> pd.DataFrame:
    # Vectorized per-genome family presence (any locus with that family pattern)
    fam_presence = _vectorized_family_detection(df)
    if fam_presence.empty:
        return pd.DataFrame(columns=["family", "n_true_with", "n_true_total", "n_false_with", "n_false_total"])  # type: ignore
    # bring phenotype
    pheno = per_genome[["gff_filename", "Spore formation"]].copy()
    fam_presence = fam_presence.merge(pheno, on="gff_filename", how="left")

    rows = []
    # totals
    n_true_total = int((pheno["Spore formation"].astype("boolean") == True).sum())
    n_false_total = int((pheno["Spore formation"].astype("boolean") == False).sum())
    for fam, sub in fam_presence.groupby("_family", sort=False):
        n_true_with = int((sub["Spore formation"].astype("boolean") == True).sum())
        n_false_with = int((sub["Spore formation"].astype("boolean") == False).sum())
        rows.append({
            "family": fam,
            "n_true_with": n_true_with,
            "n_true_total": n_true_total,
            "n_false_with": n_false_with,
            "n_false_total": n_false_total,
        })
    res = pd.DataFrame(rows)

    # proportions
    res["prop_true"] = res["n_true_with"] / res["n_true_total"].replace(0, np.nan)
    res["prop_false"] = res["n_false_with"] / res["n_false_total"].replace(0, np.nan)
    # Add pseudocount to avoid log(0)
    res["log2FC_present"] = np.log2((res["n_true_with"] + 0.5) / (res["n_true_total"] - res["n_true_with"] + 0.5)) - \
                              np.log2((res["n_false_with"] + 0.5) / (res["n_false_total"] - res["n_false_with"] + 0.5))

    # Clade-adjusted one-sided CMH across Orders (if available)
    if "Order" in per_genome.columns:
        _print_within_order_variation(per_genome, context="Curated enrichment")
        # Build strata per family: list of 2x2 tables per Order
        pg = per_genome[["gff_filename", "Order", "Spore formation"]].copy()
        res_orders = []
        for fam, _ in res.groupby("family", sort=False):
            tables: List[Tuple[int, int, int, int]] = []
            for order, sub in pg.groupby("Order", sort=False):
                fam_present = fam_presence[fam_presence["_family"] == fam]["gff_filename"].unique().tolist()
                sub_present = sub["gff_filename"].isin(fam_present)
                a = int(((sub_present) & (sub["Spore formation"].astype("boolean") == True)).sum())
                b = int(((sub_present) & (sub["Spore formation"].astype("boolean") == False)).sum())
                c = int((~sub_present & (sub["Spore formation"].astype("boolean") == True)).sum())
                d = int((~sub_present & (sub["Spore formation"].astype("boolean") == False)).sum())
                if (a + b + c + d) > 0:
                    tables.append((a, b, c, d))
            p_cmh = cmh_one_sided_greater(tables) if tables else float("nan")
            res_orders.append((fam, p_cmh))
        cmh_map = {fam: p for fam, p in res_orders}
        res["p_cmh_one_sided"] = res["family"].map(cmh_map)
        res["fdr_bh"] = bh_fdr(res["p_cmh_one_sided"].to_numpy())
    else:
        # Fallback: one-sided Fisher without stratification
        pvals = []
        for _, r in res.iterrows():
            a = int(r["n_true_with"])  # Spore+ successes
            b = int(r["n_false_with"]) # Spore- successes
            c = int(r["n_true_total"]) - a
            d = int(r["n_false_total"]) - b
            pvals.append(fisher_exact_one_sided_greater(a, b, c, d))
        res["p_fisher_one_sided"] = pvals
        res["fdr_bh"] = bh_fdr(res["p_fisher_one_sided"].to_numpy())

    # ambiguous flag: |log2FC| < 0.2 and FDR >= 0.1
    res["ambiguous"] = (res["log2FC_present"].abs() < 0.2) & (res["fdr_bh"] >= 0.1)
    return res.sort_values("fdr_bh")


# Data-driven family/term enrichment using tokens from spore-related annotations
def family_enrichment_data_driven(
    df: pd.DataFrame,
    per_genome: pd.DataFrame,
    min_genomes_with: int = 5,
    min_token_len: int = 3,
    max_token_len: int = 30,
) -> pd.DataFrame:
    """Discover candidate terms directly from data and compute presence enrichment.

    Strategy:
      - Use only rows flagged as spore_related to mine tokens (data-driven).
      - Tokenize over lowercased concatenation of (gene, Name, product, inference).
      - Build per-(genome, term) presence; aggregate by phenotype to test enrichment.

    Returns a DataFrame with the same output columns as curated family_enrichment,
    with 'family' holding the discovered term labels.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "family", "n_true_with", "n_true_total", "n_false_with", "n_false_total",
            "prop_true", "prop_false", "log2FC_present", "p_cmh_one_sided", "p_fisher_one_sided",
            "fdr_bh", "ambiguous",
        ])  # type: ignore

    cols = [c for c in ("gene", "Name", "product", "inference") if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=[
            "family", "n_true_with", "n_true_total", "n_false_with", "n_false_total",
            "prop_true", "prop_false", "log2FC_present", "p_cmh_one_sided", "p_fisher_one_sided",
            "fdr_bh", "ambiguous",
        ])  # type: ignore

    sub = df[df["spore_related"].fillna(False)].copy()
    if sub.empty:
        return pd.DataFrame(columns=[
            "family", "n_true_with", "n_true_total", "n_false_with", "n_false_total",
            "prop_true", "prop_false", "log2FC_present", "p_cmh_one_sided", "p_fisher_one_sided",
            "fdr_bh", "ambiguous",
        ])  # type: ignore

    # Build combined, lowercased text
    for c in cols:
        sub[c] = sub[c].astype("string").fillna("").str.lower()
    sub["_text"] = sub[cols].agg(" ".join, axis=1)

    # Tokenize using a conservative token regex (letters followed by letters/digits/underscore)
    min_len = max(0, min_token_len - 1)
    max_len = max_token_len - 1
    token_re = rf"[a-z][a-z0-9_]{{{min_len},{max_len}}}"
    sub["_tokens"] = sub["_text"].str.findall(token_re)
    # Explode tokens lazily to avoid huge temporary frames: concatenate lists to strings and then split
    tokens_series = sub["_tokens"].explode().dropna()
    tokens = pd.DataFrame({
        "gff_filename": sub.loc[tokens_series.index, "gff_filename"].values,
        "term": tokens_series.values,
    })
    # Remove generic stopwords that are non-informative in product descriptions
    stop = {
        "protein", "hypothetical", "domain", "family", "like", "putative", "related", "probable",
        "possible", "predicted", "conserved", "unknown", "membrane", "transporter", "binding",
        "subunit", "component", "chain", "small", "large", "type", "-associated", "enzyme",
    }
    tokens = tokens[~tokens["term"].isin(stop)]
    if tokens.empty:
        return pd.DataFrame(columns=[
            "family", "n_true_with", "n_true_total", "n_false_with", "n_false_total",
            "prop_true", "prop_false", "log2FC_present", "p_cmh_one_sided", "p_fisher_one_sided",
            "fdr_bh", "ambiguous",
        ])  # type: ignore

    # Unique presence per genome-term
    presence = tokens.drop_duplicates(["gff_filename", "term"], ignore_index=True).rename(columns={"term": "family"})

    # Bring phenotype per genome
    pheno = per_genome[["gff_filename", "Spore formation"]].copy()
    presence = presence.merge(pheno, on="gff_filename", how="left")
    presence = presence.dropna(subset=["Spore formation"])  # require known phenotype
    if presence.empty:
        return pd.DataFrame(columns=[
            "family", "n_true_with", "n_true_total", "n_false_with", "n_false_total",
            "prop_true", "prop_false", "log2FC_present", "p_cmh_one_sided", "p_fisher_one_sided",
            "fdr_bh", "ambiguous",
        ])  # type: ignore

    # Totals
    n_true_total = int((pheno["Spore formation"].astype("boolean") == True).sum())
    n_false_total = int((pheno["Spore formation"].astype("boolean") == False).sum())

    # Counts per family
    grp = presence.groupby(["family"], sort=False)["Spore formation"].apply(lambda s: (s.astype("boolean") == True).sum()).rename("n_true_with").to_frame()
    grp["n_false_with"] = presence.groupby("family")["Spore formation"].apply(lambda s: (s.astype("boolean") == False).sum())

    # Filter by minimum number of genomes where term appears
    grp = grp[(grp["n_true_with"] + grp["n_false_with"]) >= int(min_genomes_with)]
    if grp.empty:
        return pd.DataFrame(columns=[
            "family", "n_true_with", "n_true_total", "n_false_with", "n_false_total",
            "prop_true", "prop_false", "log2FC_present", "p_cmh_one_sided", "p_fisher_one_sided",
            "fdr_bh", "ambiguous",
        ])  # type: ignore

    grp = grp.reset_index()
    grp["n_true_total"] = n_true_total
    grp["n_false_total"] = n_false_total

    # Compute stats
    res = grp.copy()
    res["prop_true"] = res["n_true_with"] / res["n_true_total"].replace(0, np.nan)
    res["prop_false"] = res["n_false_with"] / res["n_false_total"].replace(0, np.nan)
    res["log2FC_present"] = np.log2((res["n_true_with"] + 0.5) / (res["n_true_total"] - res["n_true_with"] + 0.5)) - \
                             np.log2((res["n_false_with"] + 0.5) / (res["n_false_total"] - res["n_false_with"] + 0.5))

    # Clade-adjusted one-sided CMH across Orders (if available)
    if "Order" in per_genome.columns:
        _print_within_order_variation(per_genome, context="Data-driven enrichment")
        pg = per_genome[["gff_filename", "Order", "Spore formation"]].copy()
        res_orders: List[Tuple[str, float]] = []
        for fam, _ in res.groupby("family", sort=False):
            tables: List[Tuple[int, int, int, int]] = []
            fam_present_genomes = presence[presence["family"] == fam]["gff_filename"].unique().tolist()
            for order, sub in pg.groupby("Order", sort=False):
                sub_present = sub["gff_filename"].isin(fam_present_genomes)
                a = int(((sub_present) & (sub["Spore formation"].astype("boolean") == True)).sum())
                b = int(((sub_present) & (sub["Spore formation"].astype("boolean") == False)).sum())
                c = int((~sub_present & (sub["Spore formation"].astype("boolean") == True)).sum())
                d = int((~sub_present & (sub["Spore formation"].astype("boolean") == False)).sum())
                if (a + b + c + d) > 0:
                    tables.append((a, b, c, d))
            p_cmh = cmh_one_sided_greater(tables) if tables else float("nan")
            res_orders.append((fam, p_cmh))
        cmh_map = {fam: p for fam, p in res_orders}
        res["p_cmh_one_sided"] = res["family"].map(cmh_map)
        res["fdr_bh"] = bh_fdr(res["p_cmh_one_sided"].to_numpy())
    else:
        # Fallback: one-sided Fisher without stratification
        pvals = []
        for _, r in res.iterrows():
            a = int(r["n_true_with"])  # Spore+ successes
            b = int(r["n_false_with"]) # Spore- successes
            c = int(r["n_true_total"]) - a
            d = int(r["n_false_total"]) - b
            pvals.append(fisher_exact_one_sided_greater(a, b, c, d))
        res["p_fisher_one_sided"] = pvals
        res["fdr_bh"] = bh_fdr(res["p_fisher_one_sided"].to_numpy())
    res["ambiguous"] = (res["log2FC_present"].abs() < 0.2) & (res["fdr_bh"] >= 0.1)
    return res.sort_values("fdr_bh")


# Pooled (non-phylo-corrected) data-driven enrichment plus LOO-Order AUROC validation
def family_enrichment_data_driven_pooled(
    df: pd.DataFrame,
    per_genome: pd.DataFrame,
    min_genomes_with: int = 5,
    min_token_len: int = 3,
    max_token_len: int = 30,
) -> pd.DataFrame:
    """Rank terms by pooled enrichment (one-sided Fisher) and validate with leave-one-Order-out AUROC.

    Returns a DataFrame with columns similar to family_enrichment_data_driven, and extra LOO AUROC columns.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "family", "n_true_with", "n_true_total", "n_false_with", "n_false_total",
            "prop_true", "prop_false", "log2FC_present", "p_fisher_one_sided", "fdr_bh",
            "auroc_loo_mean", "auroc_loo_n_orders", "auroc_loo_min", "auroc_loo_max", "ambiguous",
        ])  # type: ignore

    cols = [c for c in ("gene", "Name", "product", "inference") if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=[
            "family", "n_true_with", "n_true_total", "n_false_with", "n_false_total",
            "prop_true", "prop_false", "log2FC_present", "p_fisher_one_sided", "fdr_bh",
            "auroc_loo_mean", "auroc_loo_n_orders", "auroc_loo_min", "auroc_loo_max", "ambiguous",
        ])  # type: ignore

    sub = df[df["spore_related"].fillna(False)].copy()
    if sub.empty:
        return pd.DataFrame(columns=[
            "family", "n_true_with", "n_true_total", "n_false_with", "n_false_total",
            "prop_true", "prop_false", "log2FC_present", "p_fisher_one_sided", "fdr_bh",
            "auroc_loo_mean", "auroc_loo_n_orders", "auroc_loo_min", "auroc_loo_max", "ambiguous",
        ])  # type: ignore

    for c in cols:
        sub[c] = sub[c].astype("string").fillna("").str.lower()
    sub["_text"] = sub[cols].agg(" ".join, axis=1)

    min_len = max(0, min_token_len - 1)
    max_len = max_token_len - 1
    token_re = rf"[a-z][a-z0-9_]{{{min_len},{max_len}}}"
    sub["_tokens"] = sub["_text"].str.findall(token_re)
    tokens_series = sub["_tokens"].explode().dropna()
    tokens = pd.DataFrame({
        "gff_filename": sub.loc[tokens_series.index, "gff_filename"].values,
        "term": tokens_series.values,
    })
    stop = {
        "protein", "hypothetical", "domain", "family", "like", "putative", "related", "probable",
        "possible", "predicted", "conserved", "unknown", "membrane", "transporter", "binding",
        "subunit", "component", "chain", "small", "large", "type", "-associated", "enzyme",
    }
    tokens = tokens[~tokens["term"].isin(stop)]
    if tokens.empty:
        return pd.DataFrame(columns=[
            "family", "n_true_with", "n_true_total", "n_false_with", "n_false_total",
            "prop_true", "prop_false", "log2FC_present", "p_fisher_one_sided", "fdr_bh",
            "auroc_loo_mean", "auroc_loo_n_orders", "auroc_loo_min", "auroc_loo_max", "ambiguous",
        ])  # type: ignore

    presence = tokens.drop_duplicates(["gff_filename", "term"], ignore_index=True).rename(columns={"term": "family"})

    # Bring phenotype and Order per genome
    pg = per_genome[["gff_filename", "Order", "Spore formation"]].copy()
    presence = presence.merge(pg, on="gff_filename", how="left")
    presence = presence.dropna(subset=["Spore formation"])  # require known phenotype
    if presence.empty:
        return pd.DataFrame(columns=[
            "family", "n_true_with", "n_true_total", "n_false_with", "n_false_total",
            "prop_true", "prop_false", "log2FC_present", "p_fisher_one_sided", "fdr_bh",
            "auroc_loo_mean", "auroc_loo_n_orders", "auroc_loo_min", "auroc_loo_max", "ambiguous",
        ])  # type: ignore

    n_true_total = int((pg["Spore formation"].astype("boolean") == True).sum())
    n_false_total = int((pg["Spore formation"].astype("boolean") == False).sum())

    grp = presence.groupby(["family"], sort=False)["Spore formation"].apply(lambda s: (s.astype("boolean") == True).sum()).rename("n_true_with").to_frame()
    grp["n_false_with"] = presence.groupby("family")["Spore formation"].apply(lambda s: (s.astype("boolean") == False).sum())

    grp = grp[(grp["n_true_with"] + grp["n_false_with"]) >= int(min_genomes_with)]
    if grp.empty:
        return pd.DataFrame(columns=[
            "family", "n_true_with", "n_true_total", "n_false_with", "n_false_total",
            "prop_true", "prop_false", "log2FC_present", "p_fisher_one_sided", "fdr_bh",
            "auroc_loo_mean", "auroc_loo_n_orders", "auroc_loo_min", "auroc_loo_max", "ambiguous",
        ])  # type: ignore

    grp = grp.reset_index()
    grp["n_true_total"] = n_true_total
    grp["n_false_total"] = n_false_total

    res = grp.copy()
    res["prop_true"] = res["n_true_with"] / res["n_true_total"].replace(0, np.nan)
    res["prop_false"] = res["n_false_with"] / res["n_false_total"].replace(0, np.nan)
    res["log2FC_present"] = np.log2((res["n_true_with"] + 0.5) / (res["n_true_total"] - res["n_true_with"] + 0.5)) - \
                             np.log2((res["n_false_with"] + 0.5) / (res["n_false_total"] - res["n_false_with"] + 0.5))

    # Pooled one-sided Fisher (no stratification)
    pvals = []
    for _, r in res.iterrows():
        a = int(r["n_true_with"])  # Spore+ successes
        b = int(r["n_false_with"]) # Spore- successes
        c = int(r["n_true_total"]) - a
        d = int(r["n_false_total"]) - b
        pvals.append(fisher_exact_one_sided_greater(a, b, c, d))
    res["p_fisher_one_sided"] = pvals
    res["fdr_bh"] = bh_fdr(res["p_fisher_one_sided"].to_numpy())

    # Leave-one-Order-out AUROC validation
    order_vals = pg["Order"].astype("string") if "Order" in pg.columns else pd.Series(dtype="string")
    unique_orders = sorted([o for o in order_vals.dropna().unique().tolist() if str(o) != "<NA>"])

    fam_to_genomes = presence.groupby("family")["gff_filename"].apply(lambda s: set(s.astype(str))).to_dict()
    gff_to_order = pg.set_index("gff_filename")["Order"].astype("string").to_dict()
    gff_to_pheno = pg.set_index("gff_filename")["Spore formation"].astype("boolean").to_dict()

    loo_means = []
    loo_counts = []
    loo_mins = []
    loo_maxs = []
    for fam in res["family"].astype(str):
        fam_genomes = fam_to_genomes.get(fam, set())
        aucs: List[float] = []
        for od in unique_orders:
            # test set: genomes in this Order with known phenotype
            test_gffs = [g for g, o in gff_to_order.items() if o == od and g in gff_to_pheno]
            if not test_gffs:
                continue
            scores = np.array([1.0 if g in fam_genomes else 0.0 for g in test_gffs], dtype=float)
            labels = np.array([bool(gff_to_pheno[g]) for g in test_gffs], dtype=bool)
            if (labels.sum() == 0) or (labels.sum() == len(labels)):
                continue  # uninformative: only one class in this Order
            x = scores[labels]
            y = scores[~labels]
            auc = auc_mann_whitney(x, y)
            if not np.isnan(auc):
                aucs.append(float(auc))
        if aucs:
            loo_means.append(float(np.mean(aucs)))
            loo_counts.append(int(len(aucs)))
            loo_mins.append(float(np.min(aucs)))
            loo_maxs.append(float(np.max(aucs)))
        else:
            loo_means.append(np.nan)
            loo_counts.append(0)
            loo_mins.append(np.nan)
            loo_maxs.append(np.nan)

    res["auroc_loo_mean"] = loo_means
    res["auroc_loo_n_orders"] = loo_counts
    res["auroc_loo_min"] = loo_mins
    res["auroc_loo_max"] = loo_maxs

    res["ambiguous"] = (res["log2FC_present"].abs() < 0.2) & (res["fdr_bh"] >= 0.1)
    return res.sort_values(["fdr_bh", "auroc_loo_mean"], ascending=[True, False])
# Exact Fisher's exact test (2x2), two-sided using doubling the minimum tail (conservative)
# Table: [[a, c], [b, d]] where columns are present/absent, rows are group True/False

def log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


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
    c2 = c + d
    n = r1 + r2
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
    a_min = max(0, c1 - r2)
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
    ap = argparse.ArgumentParser(description="Analyze spore-related signal in processed GFFs")
    ap.add_argument("--input", required=True, help="Directory with processed GFF tables (parquet/csv)")
    ap.add_argument("--outdir", default="analysis_out", help="Output directory")
    ap.add_argument("--test_glob", default=None, help="Optional glob on gff_filename to define test set (e.g., '*test*')")
    ap.add_argument("--windows", type=int, default=10_000, help="Number of sampled windows")
    ap.add_argument("--window_size", type=int, default=1_000_000, help="Window size in bp")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--skip_dedup", action="store_true", default=True, help="Skip deduplication step (already done in processing)")
    ap.add_argument("--retained_flags", default=None, help="Optional path to retained_gene_flags.csv from retained_spore_genes_analysis.py")
    ap.add_argument("--retained_rule", choices=["a", "b", "c"], default="c", help="Which rule to apply if retained_flags is set (a,b,c)")
    # Removed data-driven relabeling: keep original flags or retained-genes-only when provided
    ap.add_argument(
        "--exclude_fasta_dir",
        default="/vol/projects/BIFO/genomenet/yichen/phenotype/data/test",
        help="Directory of FASTA files to exclude (drop rows whose fasta_filename matches). Set empty string to disable.",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Reading tables...", flush=True)
    df = read_all_tables(args.input, required_columns=REQUIRED_COLUMNS)
    print("Validating schema...", flush=True)
    validate_schema(df)
    print("Coercing types...", flush=True)
    df = coerce_types(df)

    # Apply default split policy:
    #  - If a retention rule is provided (retained_flags set), we will restrict PLOTTING to test only later.
    #  - If no retention rule, exclude test from all analyses/plots by default to avoid leakage.
    retain_mode = bool(args.retained_flags)
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

    # 1) Build lightweight per-genome labels for diagnostics
    per_genome_labels = build_per_genome_labels_fast(df)
    _print_within_order_variation(per_genome_labels, context="Main (initial)")

    # 2) Compute data-driven term enrichment FIRST (always on the full, unfiltered dataset)
    print("Computing data-driven term enrichment...", flush=True)
    fam_dd = family_enrichment_data_driven(df, per_genome_labels, min_genomes_with=5)
    fam_dd.to_csv(os.path.join(args.outdir, "family_enrichment_data_driven.csv"), index=False)
    sig_count = int((fam_dd["fdr_bh"] <= 0.05).sum()) if ("fdr_bh" in fam_dd.columns) else 0
    print(
        "Data-driven terms discovered: %d; significant (FDR<=0.05): %d" % (len(fam_dd), sig_count),
        flush=True,
    )

    # 2b) Also compute pooled (non-phylo) enrichment with LOO-Order AUROC for predictive ranking
    print("Computing pooled enrichment with LOO-Order AUROC (predictive ranking)...", flush=True)
    fam_dd_pooled = family_enrichment_data_driven_pooled(df, per_genome_labels, min_genomes_with=5)
    fam_dd_pooled.to_csv(os.path.join(args.outdir, "family_enrichment_data_driven_pooled.csv"), index=False)

    # 3) No data-driven relabeling; keep original spore_related unless retained_flags provided below

    # Optional: replace spore_related flag by allowed genes for downstream metrics WITHOUT dropping any rows
    # Family enrichment (fam_dd) above is computed on the original df (independent of this step)
    if args.retained_flags:
        try:
            allowed = _load_retained_allowed_genes(args.retained_flags, rule=args.retained_rule)
            print(f"[Retained] Using retained flags from {os.path.abspath(args.retained_flags)} with rule {args.retained_rule} to redefine spore_related.", flush=True)
            df = _set_spore_related_from_allowed_genes(df, allowed)
        except Exception as e:
            print(f"[Retained] Warning: failed to redefine spore_related from retained flags: {e}. Proceeding with current flags.", flush=True)

    # 4) Recompute per-genome metrics from the relabeled (and possibly filtered) table
    per_genome, genome_cache = per_genome_metrics(df)
    per_genome.to_csv(os.path.join(args.outdir, "per_genome_metrics.csv"), index=False)

    # 5) Remove curated family enrichment; keep only data-driven analyses
    _print_within_order_variation(per_genome, context="Main (post-retained)")

    # 6) Window sampling (over selected genomes; if test_glob supplied, it's already filtered)
    print("Building sampling frame...", flush=True)
    sf = build_sampling_frame(genome_cache, allowed_genomes=None, window_size=args.window_size)
    print(f"Sampling {args.windows} windows of size {args.window_size} bp...", flush=True)
    win = sample_windows(sf, W=args.windows, window_size=args.window_size, seed=args.seed)
    win.to_csv(os.path.join(args.outdir, "window_samples.csv"), index=False)
    win_summary = summarize_windows(win)
    win_summary.to_csv(os.path.join(args.outdir, "window_summary.csv"), index=False)

    # 7) NND distribution (pooled across genomes) using relabeled spore_related
    print("Computing NND distributions...", flush=True)
    nnd_rows = []
    for gff, df_g in df.groupby("gff_filename", sort=False):
        sp = df_g[df_g["spore_related"].fillna(False)]
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
    contrast = phenotype_contrast(per_genome)
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
                plot_per_genome_metrics(per_genome_test, args.outdir, pdf=None)
                # Windows are genome-agnostic samples; replot with test-only labels if available
                plot_windows(win, per_genome_test, args.outdir, pdf=None)
                # Recompute NND for test only for plotting
                nnd_rows_plot = []
                for gff, df_g in df_test_only.groupby("gff_filename", sort=False):
                    sp = df_g[df_g["spore_related"].fillna(False)]
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
                plot_nnd(nnd_df_plot, per_genome_test, args.outdir, pdf=None)
            else:
                # Fallback to original plotting if filtering produced empty
                plot_per_genome_metrics(per_genome, args.outdir, pdf=None)
                plot_windows(win, per_genome, args.outdir, pdf=None)
                plot_nnd(nnd_df, per_genome, args.outdir, pdf=None)
        except Exception:
            plot_per_genome_metrics(per_genome, args.outdir, pdf=None)
            plot_windows(win, per_genome, args.outdir, pdf=None)
            plot_nnd(nnd_df, per_genome, args.outdir, pdf=None)
    else:
        # Default plotting
        plot_per_genome_metrics(per_genome, args.outdir, pdf=None)
        plot_windows(win, per_genome, args.outdir, pdf=None)
        plot_nnd(nnd_df, per_genome, args.outdir, pdf=None)
    # Data-driven enrichment volcano/top plots (diagnostic only)
    plot_family_enrichment(fam_dd, args.outdir, label="data_driven", pdf=None)
    # Also plot pooled predictive terms (uses same plotting function)
    try:
        plot_family_enrichment(fam_dd_pooled, args.outdir, label="data_driven_pooled", pdf=None)
    except Exception:
        pass
    plot_spore_gene_lengths(df, args.outdir, pdf=None)

    # Write markdown summary with statistical results/tables
    summary_path = os.path.join(args.outdir, "analysis_summary.md")
    with open(summary_path, "w", encoding="utf-8") as md:
        md.write("# Sporulation analysis summary\n\n")
        md.write(f"Input dir: {os.path.abspath(args.input)}\n\n")
        md.write(f"Genomes: {df['gff_filename'].nunique()}, total rows: {len(df)}\n\n")
        md.write("## Phenotype contrast (effect sizes)\n\n")
        md.write(contrast.to_string(index=False))
        md.write("\n\n## Data-driven term enrichment (top 25 by FDR)\n\n")
        md.write(fam_dd.sort_values("fdr_bh").head(25).to_string(index=False))
        md.write("\n\n## Pooled (predictive) term enrichment with LOO-Order AUROC (top 25 by FDR)\n\n")
        try:
            md.write(fam_dd_pooled.sort_values(["fdr_bh", "auroc_loo_mean"], ascending=[True, False]).head(25).to_string(index=False))
        except Exception:
            md.write("(pooled enrichment failed or unavailable)\n")

    print("Done. Outputs written to", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
