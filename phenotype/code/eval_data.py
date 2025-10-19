import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd

# Reuse robust parquet I/O/type handling from analyze_gff
from analyze_gff import coerce_types, merge_intervals

try:
    from phenotype_utils import (
        PHENOTYPE_COLUMNS,
        phenotype_to_slug,
        read_metadata_table,
        canonical_gene_name,
        build_labels_map_and_classes,
        normalize_label_value,
        DATA_ROOT,
    )
except ImportError:  # pragma: no cover - package-style import fallback
    from .phenotype_utils import (  # type: ignore
        PHENOTYPE_COLUMNS,
        phenotype_to_slug,
        read_metadata_table,
        canonical_gene_name,
        build_labels_map_and_classes,
        normalize_label_value,
        DATA_ROOT,
    )


# --------------------------- Editable defaults ---------------------------

DEFAULT_TEST_DIRS = [str(DATA_ROOT / "test")]
DEFAULT_ALL_DIRS = [
    str(DATA_ROOT / "train"),
    str(DATA_ROOT / "validation"),
    str(DATA_ROOT / "test"),
]
DEFAULT_TRAIN_DIRS = [str(DATA_ROOT / "train")]
DEFAULT_PROCESSED_GFF_DIR = str(DATA_ROOT / "processed_gff")
DEFAULT_METADATA_XLSX = "sporulation/microbe.cards table S1.xlsx"
DEFAULT_PHENOTYPE = "Spore formation"
DEFAULT_SEQ_LEN = 1_000_000
DEFAULT_N_POS = 2500  # number of target-class windows
DEFAULT_N_NEG = 2500  # number of non-target windows
DEFAULT_SEED = 42
DEFAULT_OUT_DIR = str(DATA_ROOT / "eval")


# --------------------------- FASTA utilities ---------------------------

def load_fasta_contig_lengths(fasta_path: str) -> Dict[str, int]:
    """Return dict of contig_id -> length for a multi-FASTA file.

    Only parses headers and sequence lines; ignores description after first token
    on the header line. Coordinates are 1-based in downstream use.
    """
    contig_lengths: Dict[str, int] = {}
    current_id: Optional[str] = None
    current_len: int = 0
    with open(fasta_path, "r") as f:
        for line in f:
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    contig_lengths[current_id] = contig_lengths.get(current_id, 0) + current_len
                header = line[1:].strip()
                current_id = header.split()[0]
                current_len = 0
            else:
                current_len += len(line.strip())
        if current_id is not None:
            contig_lengths[current_id] = contig_lengths.get(current_id, 0) + current_len
    return contig_lengths
# Support listing FASTAs across multiple directories (first occurrence wins on name collisions)
def list_fastas_from_dirs(dirs: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for d in dirs:
        if not d:
            continue
        if not os.path.isdir(d):
            print(f"[eval_data] Warning: FASTA directory not found: {d}")
            continue
        for fn in os.listdir(d):
            if fn.endswith((".fasta", ".fa", ".fna")) and fn not in mapping:
                mapping[fn] = os.path.join(d, fn)
    return mapping




# --------------------------- Data classes ---------------------------

@dataclass
class ContigRecord:
    fasta_filename: str
    seqid: str
    contig_len: int
    mask_union: List[Tuple[int, int]]  # 1-based inclusive intervals


def _find_processed_table(processed_gff_dir: str, fasta_filename: str) -> Optional[str]:
    base = os.path.splitext(fasta_filename)[0]
    for ext in (".parquet", ".pq", ".feather", ".csv", ".tsv"):
        p = os.path.join(processed_gff_dir, base + ext)
        if os.path.exists(p):
            return p
    # try any extension
    import glob as _glob
    globbed = _glob.glob(os.path.join(processed_gff_dir, base + ".*"))
    return globbed[0] if globbed else None


def read_gff_for_fastas(
    processed_gff_dir: str,
    allowed_fastas: List[str],
    mask_column: str,
    required_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Read only processed tables for genomes present in allowed_fastas.

    Returns a concatenated DataFrame with coerce_types applied and reduced to required_columns if provided.
    """
    rows: List[pd.DataFrame] = []
    base_cols = ["fasta_filename", "seqid", "start", "end", mask_column, "contig_len"]
    desired = set(required_columns or base_cols)
    desired.add(mask_column)
    for fa in allowed_fastas:
        path = _find_processed_table(processed_gff_dir, fa)
        if path is None:
            continue
        try:
            if path.endswith((".parquet", ".pq")):
                df = pd.read_parquet(path, columns=None)
            elif path.endswith(".feather"):
                df = pd.read_feather(path)
            elif path.endswith(".tsv"):
                df = pd.read_csv(path, sep="\t")
            else:
                df = pd.read_csv(path)
            keep = [c for c in df.columns if c in desired]
            df = df[keep]
            df = coerce_types(df, mask_column=mask_column)
            rows.append(df)
        except Exception as exc:
            print(
                f"[eval_data] Warning: failed to load processed GFF table for '{fa}': {exc}",
                flush=True,
            )
            continue
    if not rows:
        return pd.DataFrame(columns=list(desired))
    return pd.concat(rows, ignore_index=True)


def build_contig_table(fasta_dir_map: Dict[str, str],
                       fasta_files: List[str]) -> pd.DataFrame:
    """Build a table of contigs with lengths and optional phenotype unions.

    Columns: fasta_filename, seqid, contig_len, mask_union (list-of-tuples)
    """
    # 1) contig lengths from FASTA files
    contig_rows: List[ContigRecord] = []
    fasta_to_lengths: Dict[str, Dict[str, int]] = {}
    for fn in fasta_files:
        path = fasta_dir_map.get(fn, os.path.join(".", fn))
        lengths = load_fasta_contig_lengths(path)
        fasta_to_lengths[fn] = lengths

    for fn in fasta_files:
        lengths = fasta_to_lengths.get(fn, {})
        for seqid, L in lengths.items():
            contig_rows.append(
                ContigRecord(
                    fasta_filename=fn,
                    seqid=seqid,
                    contig_len=int(L),
                    mask_union=[],
                )
            )

    df = pd.DataFrame([r.__dict__ for r in contig_rows])
    if df.empty:
        raise RuntimeError("No contigs found in provided FASTA directories.")
    return df


def build_fasta_to_mask_union_map_from_df(gff_df: pd.DataFrame,
                                          contigs_df: pd.DataFrame,
                                          mask_column: str,
                                          allowed_genes: Optional[Set[str]] = None) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
    """Map (fasta_filename, FASTA seqid) -> merged phenotype intervals via contig length matching.

    Strategy per genome (fasta_filename):
      - Compute GFF contig lengths as max(end) per seqid from processed tables.
      - For a given length L, if there is exactly one GFF seqid with length L and
        exactly one FASTA seqid with length L, map them and attach that GFF seqid's
        merged phenotype intervals to the FASTA seqid.
      - Ambiguous lengths (multiplicity >1 on either side) are skipped.
    """
    # Load only required columns once
    df = gff_df
    if df is None or df.empty:
        return {}

    # Normalize genome key to basename (extension-agnostic) for robust matching
    def _genome_key(name: object) -> str:
        s = str(name)
        return os.path.splitext(s)[0]

    # GFF contig lengths per (genome, seqid) – prefer persisted contig_len from contig meta rows
    if "contig_len" in df.columns:
        # Use the maximum contig_len per (genome, seqid), falling back to max(end) if contig_len missing
        end_max = df.groupby(["fasta_filename", "seqid"], sort=False)["end"].max().rename("end_max").reset_index()
        contig_len_max = df.groupby(["fasta_filename", "seqid"], sort=False)["contig_len"].max().rename("contig_len_max").reset_index()
        gff_len = contig_len_max.merge(end_max, on=["fasta_filename", "seqid"], how="outer")
        gff_len["gff_len"] = gff_len["contig_len_max"].fillna(gff_len["end_max"]).astype("Int64")
        gff_len = gff_len[["fasta_filename", "seqid", "gff_len"]]
    else:
        gff_len = (
            df.groupby(["fasta_filename", "seqid"], sort=False)["end"].max().rename("gff_len").reset_index()
        )
    # Add normalized genome key
    gff_len["genome"] = gff_len["fasta_filename"].map(_genome_key)
    # Drop rows lacking a computable length
    gff_len = gff_len.dropna(subset=["gff_len"]).reset_index(drop=True)
    # GFF phenotype unions per (genome, seqid)
    mask_only = df[df[mask_column].fillna(False)].copy()
    # Optional: restrict to retained genes by canonical gene/name
    if allowed_genes:
        if "gene" not in mask_only.columns:
            mask_only["gene"] = pd.Series(dtype="string")
        if "Name" not in mask_only.columns:
            mask_only["Name"] = pd.Series(dtype="string")
        mask_only["__gene_canon"] = mask_only["gene"].apply(canonical_gene_name)
        mask_only["__name_canon"] = mask_only["Name"].apply(canonical_gene_name)
        mask_allowed = mask_only["__gene_canon"].isin(allowed_genes) | mask_only["__name_canon"].isin(allowed_genes)
        mask_only = mask_only.loc[mask_allowed].copy()
        mask_only = mask_only.drop(columns=["__gene_canon", "__name_canon"], errors="ignore")
    mask_only["genome"] = mask_only["fasta_filename"].map(_genome_key)
    gff_union_map: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    if not mask_only.empty:
        for (genome_key, seqid), sub in mask_only.groupby(["genome", "seqid"], sort=False):
            starts = sub["start"].to_numpy(dtype=np.int64)
            ends = sub["end"].to_numpy(dtype=np.int64)
            gff_union_map[(str(genome_key), str(seqid))] = merge_intervals(starts, ends)

    # FASTA contig lengths per (genome, seqid)
    fasta_len = (
        contigs_df.loc[:, ["fasta_filename", "seqid", "contig_len"]].copy().rename(columns={"contig_len": "fa_len"})
    )
    fasta_len["genome"] = fasta_len["fasta_filename"].map(_genome_key)

    # For each genome, build length multiplicity tables and join unique matches
    out: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    order_based_applied = 0
    # Iterate per genome base; handle extension mismatches gracefully
    for genome_key, sub_fa_all in fasta_len.groupby("genome", sort=False):
        sub_gff = gff_len[gff_len["genome"] == genome_key]
        if sub_gff.empty:
            continue
        # There should normally be exactly one FASTA file per genome base. Handle multiple defensively.
        for fa_file, sub_fa in sub_fa_all.groupby("fasta_filename", sort=False):
            # count multiplicities per length
            fa_counts = sub_fa.groupby("fa_len").size().rename("fa_n").reset_index()
            gff_counts = sub_gff.groupby("gff_len").size().rename("gff_n").reset_index()
            # merge on length where both sides have exactly one
            fa_unique = sub_fa.merge(fa_counts, on="fa_len").query("fa_n == 1")
            gff_unique = sub_gff.merge(gff_counts, on="gff_len").query("gff_n == 1")
            match = fa_unique.merge(gff_unique, left_on="fa_len", right_on="gff_len", how="inner")
            mapped_pairs: Dict[str, str] = {}
            for _, r in match.iterrows():
                fa_seqid = str(r["seqid_x"])  # FASTA seqid
                gff_seqid = str(r["seqid_y"])  # GFF seqid
                intervals = gff_union_map.get((str(genome_key), gff_seqid), [])
                out[(str(fa_file), fa_seqid)] = intervals
                mapped_pairs[fa_seqid] = gff_seqid
            # Fallback: single-contig genomes on both sides map 1:1 regardless of name
            if match.empty:
                fa_seqids = sub_fa["seqid"].astype(str).unique().tolist()
                gff_seqids = sub_gff["seqid"].astype(str).unique().tolist()
                if len(fa_seqids) == 1 and len(gff_seqids) == 1:
                    gff_seqid = gff_seqids[0]
                    fa_seqid = fa_seqids[0]
                    intervals = gff_union_map.get((str(genome_key), gff_seqid), [])
                    out[(str(fa_file), fa_seqid)] = intervals
                    mapped_pairs[fa_seqid] = gff_seqid
            # Additional fallback: order-based mapping when counts are equal and lengths match index-wise
            # Build FASTA and GFF seqid order lists
            fa_seqid_list = sub_fa["seqid"].astype(str).tolist()
            gff_seqid_list = sub_gff["seqid"].astype(str).tolist()
            if len(fa_seqid_list) == len(gff_seqid_list) and len(fa_seqid_list) > 1:
                # Build length dicts
                fa_len_by_id = {str(r.seqid): int(r.fa_len) for r in sub_fa.itertuples(index=False)}
                gff_len_by_id = {str(r.seqid): int(r.gff_len) for r in sub_gff.itertuples(index=False)}
                # Multiset equality of lengths
                fa_lens_sorted = sorted([fa_len_by_id[sid] for sid in fa_seqid_list])
                gff_lens_sorted = sorted([gff_len_by_id[sid] for sid in gff_seqid_list if sid in gff_len_by_id])
                if len(fa_lens_sorted) == len(gff_lens_sorted) and fa_lens_sorted == gff_lens_sorted:
                    applied_here = False
                    for i in range(len(fa_seqid_list)):
                        fa_seqid = fa_seqid_list[i]
                        if fa_seqid in mapped_pairs:
                            continue
                        gff_seqid = gff_seqid_list[i]
                        # Per-index length equality gate
                        if fa_len_by_id.get(fa_seqid) != gff_len_by_id.get(gff_seqid):
                            continue
                        intervals = gff_union_map.get((str(genome_key), gff_seqid), [])
                        out[(str(fa_file), fa_seqid)] = intervals
                        mapped_pairs[fa_seqid] = gff_seqid
                        applied_here = True
                    if applied_here:
                        order_based_applied += 1
    if order_based_applied > 0:
        print(f"  Order-based mapping applied for {order_based_applied} genomes (seqid counts equal)", flush=True)
    return out


def build_fasta_mask_intervals_with_genes_map_from_df(gff_df: pd.DataFrame,
                                                       contigs_df: pd.DataFrame,
                                                       mask_column: str,
                                                       allowed_genes: Optional[Set[str]] = None) -> Dict[Tuple[str, str], List[Tuple[int, int, List[str], str]]]:
    """Map (fasta_filename, FASTA seqid) -> list of (start, end, [gene_names], strand_label) for phenotype ground-truth loci unions.

    Gene name extraction priority: gene -> Name -> locus_tag. Names are ordered by locus start
    within each merged interval. Uses the same contig length matching strategy as
    build_fasta_to_mask_union_map_from_df.
    """
    df = gff_df
    if df is None or df.empty:
        return {}

    def _genome_key(name: object) -> str:
        s = str(name)
        return os.path.splitext(s)[0]

    # GFF contig lengths per (genome, seqid)
    if "contig_len" in df.columns:
        end_max = df.groupby(["fasta_filename", "seqid"], sort=False)["end"].max().rename("end_max").reset_index()
        contig_len_max = df.groupby(["fasta_filename", "seqid"], sort=False)["contig_len"].max().rename("contig_len_max").reset_index()
        gff_len = contig_len_max.merge(end_max, on=["fasta_filename", "seqid"], how="outer")
        gff_len["gff_len"] = gff_len["contig_len_max"].fillna(gff_len["end_max"]).astype("Int64")
        gff_len = gff_len[["fasta_filename", "seqid", "gff_len"]]
    else:
        gff_len = (
            df.groupby(["fasta_filename", "seqid"], sort=False)["end"].max().rename("gff_len").reset_index()
        )
    gff_len["genome"] = gff_len["fasta_filename"].map(_genome_key)
    gff_len = gff_len.dropna(subset=["gff_len"]).reset_index(drop=True)

    # Spore-only with gene labels
    mask_only = df[df[mask_column].fillna(False)].copy()
    # Optional: restrict to retained genes by canonical gene/name
    if allowed_genes:
        for c in ("gene", "Name", "locus_tag"):
            if c not in mask_only.columns:
                mask_only[c] = pd.Series(dtype="string")
        mask_only["__gene_canon"] = mask_only["gene"].apply(canonical_gene_name)
        mask_only["__name_canon"] = mask_only["Name"].apply(canonical_gene_name)
        mask_allowed = mask_only["__gene_canon"].isin(allowed_genes) | mask_only["__name_canon"].isin(allowed_genes)
        mask_only = mask_only.loc[mask_allowed].copy()
        mask_only = mask_only.drop(columns=["__gene_canon", "__name_canon"], errors="ignore")
    mask_only["genome"] = mask_only["fasta_filename"].map(_genome_key)
    # Build gene_label column with priority gene -> Name -> locus_tag
    for c in ("gene", "Name", "locus_tag"):
        if c not in mask_only.columns:
            mask_only[c] = pd.Series(dtype="string")
    mask_only["gene_label"] = mask_only["gene"].astype("string")
    mask_only.loc[mask_only["gene_label"].isna() | (mask_only["gene_label"].str.len() == 0), "gene_label"] = mask_only["Name"].astype("string")
    mask_only.loc[mask_only["gene_label"].isna() | (mask_only["gene_label"].str.len() == 0), "gene_label"] = mask_only["locus_tag"].astype("string")

    # Build merged intervals and aligned gene lists keyed by (genome, gff_seqid)
    gff_intervals_with_genes: Dict[Tuple[str, str], List[Tuple[int, int, List[str], str]]] = {}
    if not mask_only.empty:
        for (genome_key, seqid), sub in mask_only.groupby(["genome", "seqid"], sort=False):
            sub_sorted = sub.sort_values(["start", "end"])  # ensure order
            starts = sub_sorted["start"].to_numpy(dtype=np.int64)
            ends = sub_sorted["end"].to_numpy(dtype=np.int64)
            merged = merge_intervals(starts, ends)
            # Assign genes to merged intervals in order
            genes = sub_sorted["gene_label"].astype("string").fillna("").tolist()
            # Standardize strand values to '+', '-', or None
            strands_raw = sub_sorted.get("strand", pd.Series(index=sub_sorted.index, dtype="string")).astype("string").tolist()
            def _norm_strand(x: object) -> Optional[str]:
                s = str(x) if pd.notna(x) else ""
                if s.startswith("+"):
                    return "+"
                if s.startswith("-"):
                    return "-"
                return None
            strands = [_norm_strand(x) for x in strands_raw]
            starts_list = starts.tolist()
            ends_list = ends.tolist()
            items: List[Tuple[int, int, List[str], str]] = []
            for ms, me in merged:
                names_here: List[str] = []
                strand_here_vals: List[str] = []
                # collect genes overlapping [ms, me], keep order by starts_list
                for s, e, nm, st in zip(starts_list, ends_list, genes, strands):
                    if e < ms:
                        continue
                    if s > me:
                        break
                    if nm and nm != "None" and nm != ".":
                        names_here.append(str(nm))
                    if st in ("+", "-"):
                        strand_here_vals.append(st)
                # decide interval strand label
                strand_label: str
                uniq = sorted(set(strand_here_vals))
                if len(uniq) == 1:
                    strand_label = uniq[0]
                elif len(uniq) == 0:
                    strand_label = "unknown"
                else:
                    strand_label = "mixed"
                items.append((int(ms), int(me), names_here, strand_label))
            gff_intervals_with_genes[(str(genome_key), str(seqid))] = items

    # FASTA contig lengths per (genome, seqid)
    fasta_len = (
        contigs_df.loc[:, ["fasta_filename", "seqid", "contig_len"]].copy().rename(columns={"contig_len": "fa_len"})
    )
    fasta_len["genome"] = fasta_len["fasta_filename"].map(_genome_key)

    # Map to FASTA seqids via unique-length matching, with fallbacks as before
    out: Dict[Tuple[str, str], List[Tuple[int, int, List[str], str]]] = {}
    order_based_applied = 0
    for genome_key, sub_fa_all in fasta_len.groupby("genome", sort=False):
        sub_gff = gff_len[gff_len["genome"] == genome_key]
        if sub_gff.empty:
            continue
        for fa_file, sub_fa in sub_fa_all.groupby("fasta_filename", sort=False):
            fa_counts = sub_fa.groupby("fa_len").size().rename("fa_n").reset_index()
            gff_counts = sub_gff.groupby("gff_len").size().rename("gff_n").reset_index()
            fa_unique = sub_fa.merge(fa_counts, on="fa_len").query("fa_n == 1")
            gff_unique = sub_gff.merge(gff_counts, on="gff_len").query("gff_n == 1")
            match = fa_unique.merge(gff_unique, left_on="fa_len", right_on="gff_len", how="inner")
            mapped_pairs: Dict[str, str] = {}
            for _, r in match.iterrows():
                fa_seqid = str(r["seqid_x"])  # FASTA seqid
                gff_seqid = str(r["seqid_y"])  # GFF seqid
                items = gff_intervals_with_genes.get((str(genome_key), gff_seqid), [])
                out[(str(fa_file), fa_seqid)] = items
                mapped_pairs[fa_seqid] = gff_seqid
            if match.empty:
                fa_seqids = sub_fa["seqid"].astype(str).unique().tolist()
                gff_seqids = sub_gff["seqid"].astype(str).unique().tolist()
                if len(fa_seqids) == 1 and len(gff_seqids) == 1:
                    gff_seqid = gff_seqids[0]
                    fa_seqid = fa_seqids[0]
                    items = gff_intervals_with_genes.get((str(genome_key), gff_seqid), [])
                    out[(str(fa_file), fa_seqid)] = items
                    mapped_pairs[fa_seqid] = gff_seqid
            # order-based fallback when multiset lengths match
            fa_seqid_list = sub_fa["seqid"].astype(str).tolist()
            gff_seqid_list = sub_gff["seqid"].astype(str).tolist()
            if len(fa_seqid_list) == len(gff_seqid_list) and len(fa_seqid_list) > 1:
                fa_len_by_id = {str(r.seqid): int(r.fa_len) for r in sub_fa.itertuples(index=False)}
                gff_len_by_id = {str(r.seqid): int(r.gff_len) for r in sub_gff.itertuples(index=False)}
                fa_lens_sorted = sorted([fa_len_by_id[sid] for sid in fa_seqid_list])
                gff_lens_sorted = sorted([gff_len_by_id[sid] for sid in gff_seqid_list if sid in gff_len_by_id])
                if len(fa_lens_sorted) == len(gff_lens_sorted) and fa_lens_sorted == gff_lens_sorted:
                    applied_here = False
                    for i in range(len(fa_seqid_list)):
                        fa_seqid = fa_seqid_list[i]
                        if fa_seqid in mapped_pairs:
                            continue
                        gff_seqid = gff_seqid_list[i]
                        if fa_len_by_id.get(fa_seqid) != gff_len_by_id.get(gff_seqid):
                            continue
                        items = gff_intervals_with_genes.get((str(genome_key), gff_seqid), [])
                        out[(str(fa_file), fa_seqid)] = items
                        mapped_pairs[fa_seqid] = gff_seqid
                        applied_here = True
                    if applied_here:
                        order_based_applied += 1
    if order_based_applied > 0:
        print(f"  Order-based mapping (with genes) applied for {order_based_applied} genomes (seqid counts equal)", flush=True)
    return out

def sample_windows(contigs_df: pd.DataFrame, W: int, window_size: int, seed: int) -> pd.DataFrame:
    """Sample W windows of length window_size across contigs weighted by positions."""
    df = contigs_df.copy()
    df["weight"] = (df["contig_len"].astype(int) - int(window_size) + 1).clip(lower=0)
    df = df[df["weight"] > 0].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["fasta_filename", "seqid", "start", "end", "mask_union"])

    rng = np.random.default_rng(seed)
    probs = df["weight"].to_numpy(dtype=float)
    probs = probs / probs.sum()
    choices = rng.choice(len(df), size=W, replace=True, p=probs)
    meta = df.iloc[choices].reset_index(drop=True)

    starts: List[int] = []
    ends: List[int] = []
    for _, row in meta.iterrows():
        L = int(row["contig_len"])  # 1-based inclusive coordinate space
        s = int(rng.integers(1, L - window_size + 2))
        e = s + window_size - 1
        starts.append(s)
        ends.append(e)
    out = meta.assign(start=starts, end=ends)
    return out


def intersect_intervals_with_window(window_start: int,
                                    window_end: int,
                                    intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Return list of relative (start,end) within window, 0-based inclusive.

    Inputs intervals and window coordinates are 1-based inclusive.
    """
    rel: List[Tuple[int, int]] = []
    if not intervals:
        return rel
    for s, e in intervals:
        a = max(int(s), int(window_start))
        b = min(int(e), int(window_end))
        if a <= b:
            rs = (a - window_start)  # 0-based inclusive
            re = (b - window_start)  # 0-based inclusive
            rel.append((int(rs), int(re)))
    return rel


def main():
    ap = argparse.ArgumentParser(description="Prepare evaluation data: sample windows and ground-truth masks")
    ap.add_argument("--test_dirs", nargs='+', type=str, default=DEFAULT_TEST_DIRS, help="Directory/directories with test FASTA files (used for sampling windows)")
    ap.add_argument("--all_dirs", nargs='+', type=str, default=DEFAULT_ALL_DIRS, help="Directories with all FASTA files (train/val/test) to build genome_intervals")
    ap.add_argument("--train_dirs", nargs='+', type=str, default=None, help="Directories containing training FASTA files for class balancing checks")
    ap.add_argument("--processed_gff_dir", type=str, default=DEFAULT_PROCESSED_GFF_DIR, help="Directory with processed GFF parquet/csv")
    ap.add_argument("--metadata_xlsx", type=str, default=DEFAULT_METADATA_XLSX, help="Metadata Excel with phenotype annotations")
    ap.add_argument("--phenotype", type=str, default=DEFAULT_PHENOTYPE, help="Phenotype column name to evaluate (must exist in metadata)")
    ap.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN, help="Window length (bp)")
    ap.add_argument("--n_pos", type=int, default=DEFAULT_N_POS, help="Number of target-class windows to sample")
    ap.add_argument("--n_neg", type=int, default=DEFAULT_N_NEG, help="Number of non-target windows to sample")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for prepared data")
    ap.add_argument("--retained_flags", type=str, default=None, help="Optional retained_gene_flags.csv; if set, restrict ground truth to retained genes")
    ap.add_argument("--retained_rule", type=str, default=None, help="Rule to select retained genes (one of a/b/c). Default: none (no filtering)")
    ap.add_argument("--target-class", type=str, default=None, help="Phenotype class treated as target for attribution (defaults to 'true' if present, else first class)")
    args = ap.parse_args()

    phenotype = args.phenotype
    slug = phenotype_to_slug(phenotype)
    mask_column = f"gt_{slug}"
    out_dir = Path(args.out_dir) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)

    # List FASTA files: test-only for sampling; all splits for genome_intervals
    test_fasta_map = list_fastas_from_dirs(list(args.test_dirs))
    test_fasta_files = sorted(test_fasta_map.keys())
    if not test_fasta_files:
        raise FileNotFoundError(f"No FASTA files found under test_dirs: {args.test_dirs}")
    all_fasta_map = list_fastas_from_dirs(list(args.all_dirs))
    all_fasta_files = sorted(all_fasta_map.keys())
    if not all_fasta_files:
        raise FileNotFoundError(f"No FASTA files found under all_dirs: {args.all_dirs}")

    if phenotype not in PHENOTYPE_COLUMNS:
        print(f"Warning: phenotype '{phenotype}' not in default phenotype list; proceeding anyway.", flush=True)

    metadata_path = os.path.abspath(args.metadata_xlsx)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata Excel not found: {metadata_path}")
    metadata_df = read_metadata_table(Path(metadata_path))
    if phenotype not in metadata_df.columns:
        raise ValueError(f"Phenotype column '{phenotype}' not found in metadata Excel")

    train_dirs = list(args.train_dirs) if args.train_dirs else []
    if not train_dirs:
        inferred = [d for d in args.all_dirs if Path(d).name.lower() == "train"]
        train_dirs.extend(inferred)
    if not train_dirs:
        train_dirs = [d for d in DEFAULT_TRAIN_DIRS if os.path.isdir(d)]

    labels_map, classes = build_labels_map_and_classes(
        metadata_df,
        phenotype_col=phenotype,
        file_col="Fasta file",
        train_dirs=train_dirs,
    )
    class_to_id = {c: i for i, c in enumerate(classes)}

    def _norm_name(name: str) -> str:
        return os.path.basename(name).strip().lower()

    target_class = args.target_class
    if target_class is not None:
        target_class = normalize_label_value(target_class)
    else:
        if "true" in class_to_id:
            target_class = "true"
        elif classes:
            target_class = classes[0]
        else:
            raise RuntimeError("No classes available for phenotype; cannot prepare evaluation dataset.")
    if target_class not in class_to_id:
        raise ValueError(f"Target class '{target_class}' not among available classes: {classes}")
    target_class_id = class_to_id[target_class]

    class_fastas: Dict[int, List[str]] = {cid: [] for cid in class_to_id.values()}
    labeled_test_fastas: List[str] = []
    unlabeled_fastas: List[str] = []
    for fn in test_fasta_files:
        lbl = labels_map.get(_norm_name(fn))
        if lbl is None:
            unlabeled_fastas.append(fn)
            continue
        class_fastas.setdefault(lbl, []).append(fn)
        labeled_test_fastas.append(fn)

    pos_fastas = class_fastas.get(target_class_id, [])
    neg_fastas = [fn for cid, fastas in class_fastas.items() if cid != target_class_id for fn in fastas]

    if unlabeled_fastas:
        print(f"Warning: {len(unlabeled_fastas)} test FASTA files lack phenotype '{phenotype}' labels; skipping them.", flush=True)
    if not pos_fastas:
        raise RuntimeError(f"No genomes found for target class '{target_class}' in test set.")
    if not neg_fastas:
        raise RuntimeError("No genomes found for non-target classes in test set.")
    pos_fastas.sort()
    neg_fastas.sort()
    labeled_test_fastas.sort()
    class_counts = {classes[cid]: len(class_fastas.get(cid, [])) for cid in class_fastas}
    print(
        f"Labeled test genomes for '{phenotype}' (target='{target_class}'): "
        f"target={len(pos_fastas)}, other={len(neg_fastas)}, unlabeled={len(unlabeled_fastas)}",
        flush=True,
    )

    # Build contig table (lengths and ground-truth unions if available)
    contigs_test_df = build_contig_table(test_fasta_map, labeled_test_fastas)

    # Separate contigs by phenotype using fasta filename lists
    contigs_pos = contigs_test_df[contigs_test_df["fasta_filename"].isin(pos_fastas)].reset_index(drop=True)
    contigs_neg = contigs_test_df[contigs_test_df["fasta_filename"].isin(neg_fastas)].reset_index(drop=True)
    if contigs_pos.empty:
        raise RuntimeError("No contigs for positive genomes.")
    if contigs_neg.empty:
        raise RuntimeError("No contigs for negative genomes.")

    # Sample windows for each stratum
    win_pos = sample_windows(contigs_pos, W=int(args.n_pos), window_size=int(args.seq_len), seed=int(args.seed))
    win_neg = sample_windows(contigs_neg, W=int(args.n_neg), window_size=int(args.seq_len), seed=int(args.seed) + 1)

    # Annotate labels and groups
    win_pos = win_pos.assign(label=target_class_id, label_name=target_class, group="target")

    win_neg = win_neg.copy()
    win_neg["label"] = win_neg["fasta_filename"].map(lambda x: labels_map.get(_norm_name(x)))
    if win_neg["label"].isnull().any():
        raise RuntimeError("Encountered unlabeled FASTA while assigning classes for non-target windows.")
    win_neg["label"] = win_neg["label"].astype(int)
    win_neg["label_name"] = win_neg["label"].map(lambda idx: classes[idx])
    win_neg["group"] = "other"

    samples = pd.concat([win_pos, win_neg], ignore_index=True)
    samples.insert(0, "sample_id", np.arange(len(samples), dtype=int))
    samples["label_name"] = samples["label"].map(lambda idx: classes[idx])
    samples["target_mask"] = samples["label"] == target_class_id

    # Build robust mapping from FASTA seqids to GFF phenotype unions via length matching (ALL genomes)
    print("Building FASTA->GFF contig mapping via length matching (all splits)...", flush=True)
    gff_df = read_gff_for_fastas(
        args.processed_gff_dir,
        all_fasta_files,
        mask_column=mask_column,
        required_columns=[
            "fasta_filename", "seqid", "start", "end", "strand", mask_column, "contig_len",
            "gene", "Name", "locus_tag"
        ],
    )
    if mask_column not in gff_df.columns:
        raise ValueError(
            f"Processed GFF tables in {args.processed_gff_dir} do not contain column '{mask_column}'. "
            "Re-run process_gff.py after adding phenotype ground-truth masks."
        )
    # Load retained genes if provided
    allowed_genes: Optional[Set[str]] = None
    if args.retained_flags and args.retained_rule:
        retained_path = os.path.abspath(args.retained_flags)
        if not os.path.exists(retained_path):
            raise FileNotFoundError(f"retained_flags CSV not found: {retained_path}")
        df_ret = pd.read_csv(retained_path)
        # Column with canonical names
        if "gene_canonical" not in df_ret.columns and "gene_canon" in df_ret.columns:
            df_ret = df_ret.rename(columns={"gene_canon": "gene_canonical"})
        if "gene_canonical" not in df_ret.columns:
            raise ValueError("retained_flags CSV must contain 'gene_canonical'")
        rule_col_map = {"a": "pass_a_family_sig", "b": "pass_b_nt_pid", "c": "pass_c_both"}
        rule_col = rule_col_map.get(str(args.retained_rule))
        if rule_col not in df_ret.columns:
            raise ValueError(f"retained_flags CSV missing required column '{rule_col}' for rule {args.retained_rule}")
        df_sel = df_ret[df_ret[rule_col] == True]
        allowed_genes = set(df_sel["gene_canonical"].astype(str).dropna().tolist())
        print(f"Retained gene filtering enabled (rule {args.retained_rule}): {len(allowed_genes)} genes allowed.")
    else:
        print("Retained gene filtering disabled (no flags/rule provided).", flush=True)

    contigs_all_df = build_contig_table(all_fasta_map, all_fasta_files)
    fasta_to_union = build_fasta_to_mask_union_map_from_df(
        gff_df,
        contigs_all_df,
        mask_column=mask_column,
        allowed_genes=allowed_genes,
    )
    fasta_to_items = build_fasta_mask_intervals_with_genes_map_from_df(
        gff_df,
        contigs_all_df,
        mask_column=mask_column,
        allowed_genes=allowed_genes,
    )
    mapped_keys = len(fasta_to_union)
    total_keys = contigs_all_df.shape[0]
    print(f"  Contig mappings available: {mapped_keys}/{total_keys} (unique-length matches)", flush=True)

    # Compute genome-level intervals per contig (merged sporulation intervals, no windowing)
    contig_interval_rows: List[str] = []
    for r in contigs_all_df.itertuples(index=False):
        key = (str(getattr(r, "fasta_filename")), str(getattr(r, "seqid")))
        intervals = fasta_to_union.get(key, [])
        contig_interval_rows.append(json.dumps(intervals))
    genome_intervals_df = contigs_all_df.loc[:, ["fasta_filename", "seqid", "contig_len"]].copy()
    genome_intervals_df["intervals_json"] = contig_interval_rows
    # Also include gene names and strand aligned to intervals (order preserved)
    contig_genes_rows: List[str] = []
    contig_strand_rows: List[str] = []
    for r in contigs_all_df.itertuples(index=False):
        key = (str(getattr(r, "fasta_filename")), str(getattr(r, "seqid")))
        items = fasta_to_items.get(key, [])
        if items:
            genes_aligned = [[str(x) for x in (names or [])] for _, _, names, _ in items]
            strands_aligned = [str(st) for _, _, _, st in items]
        else:
            genes_aligned = []
            strands_aligned = []
        contig_genes_rows.append(json.dumps(genes_aligned))
        contig_strand_rows.append(json.dumps(strands_aligned))
    genome_intervals_df["interval_genes_json"] = contig_genes_rows
    genome_intervals_df["interval_strand_json"] = contig_strand_rows

    # For positives (in test only), compute relative intervals within each window (JSON-encoded)
    intervals_json: List[Optional[str]] = []
    interval_genes_json: List[Optional[str]] = []
    interval_strand_json: List[Optional[str]] = []
    for r in samples.itertuples(index=False):
        if bool(getattr(r, "target_mask", False)):
            ws = int(getattr(r, "start"))  # 1-based inclusive
            we = int(getattr(r, "end"))    # 1-based inclusive
            items = fasta_to_items.get((str(getattr(r, "fasta_filename")), str(getattr(r, "seqid"))), [])
            rel_intervals: List[Tuple[int, int]] = []
            rel_genes: List[List[str]] = []
            rel_strands: List[str] = []
            for ms, me, names, st in items:
                a = max(int(ms), ws)
                b = min(int(me), we)
                if a <= b:
                    rs = a - ws
                    re = b - ws
                    rel_intervals.append((int(rs), int(re)))
                    rel_genes.append([str(x) for x in (names or [])])
                    rel_strands.append(str(st))
            intervals_json.append(json.dumps(rel_intervals))
            interval_genes_json.append(json.dumps(rel_genes))
            interval_strand_json.append(json.dumps(rel_strands))
        else:
            intervals_json.append(None)
            interval_genes_json.append(None)
            interval_strand_json.append(None)
    samples["intervals_json"] = intervals_json
    samples["interval_genes_json"] = interval_genes_json
    samples["interval_strand_json"] = interval_strand_json

    # Keep only necessary columns in output; retain mask_union only if desired for debugging
    out_cols = [
        "sample_id", "group", "label", "label_name", "target_mask",
        "fasta_filename", "seqid", "start", "end",
        "intervals_json", "interval_genes_json", "interval_strand_json"
    ]
    samples_out = samples.loc[:, out_cols].copy()

    # Write outputs
    out_samples_path = os.path.join(args.out_dir, "samples.parquet")
    samples_out.to_parquet(out_samples_path, index=False)
    # Write genome-level intervals (per contig)
    out_genomes_path = os.path.join(args.out_dir, "genome_intervals.parquet")
    genome_intervals_df.to_parquet(out_genomes_path, index=False)

    manifest = {
        "seq_len": int(args.seq_len),
        "n_pos": int(args.n_pos),
        "n_neg": int(args.n_neg),
        "test_dirs": [os.path.abspath(d) for d in args.test_dirs],
        "all_dirs": [os.path.abspath(d) for d in args.all_dirs],
        "processed_gff_dir": os.path.abspath(args.processed_gff_dir),
        "metadata_xlsx": metadata_path,
        "phenotype": phenotype,
        "mask_column": mask_column,
        "classes": classes,
        "class_to_id": class_to_id,
        "class_counts": class_counts,
        "target_class": target_class,
        "target_class_id": int(target_class_id),
        "unlabeled_test_fastas": sorted(unlabeled_fastas),
        "samples_path": os.path.abspath(out_samples_path),
        "genomes_path": os.path.abspath(out_genomes_path),
    }
    with open(os.path.join(args.out_dir, "eval_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Prepared samples written to {out_samples_path}")
    print(f"Genome-level intervals written to {out_genomes_path}")
    print(f"Counts: target={len(pos_fastas)}, other={len(neg_fastas)}")


if __name__ == "__main__":
    main()
