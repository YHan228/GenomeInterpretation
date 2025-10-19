import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd

# Reuse robust parquet I/O/type handling from analyze_gff
from analyze_gff import read_all_tables, coerce_types, merge_intervals, canonical_gene_name


# --------------------------- Editable defaults ---------------------------

DATA_ROOT = "/vol/projects/BIFO/genomenet/yichen/phenotype/data"
DEFAULT_TEST_DIRS = [f"{DATA_ROOT}/test"]
DEFAULT_ALL_DIRS = [
    f"{DATA_ROOT}/train",
    f"{DATA_ROOT}/validation",
    f"{DATA_ROOT}/test",
]
DEFAULT_PROCESSED_GFF_DIR = f"{DATA_ROOT}/processed_gff"
DEFAULT_SPOREINFO_CSV = "sporulation/sporeinfo.csv"
DEFAULT_SEQ_LEN = 1_000_000
DEFAULT_N_POS = 2500
DEFAULT_N_NEG = 2500
DEFAULT_SEED = 42
DEFAULT_OUT_DIR = f"{DATA_ROOT}/eval"


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
    spore_union: List[Tuple[int, int]]  # 1-based inclusive intervals


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


def read_gff_for_fastas(processed_gff_dir: str, allowed_fastas: List[str], required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read only processed tables for genomes present in allowed_fastas.

    Returns a concatenated DataFrame with coerce_types applied and reduced to required_columns if provided.
    """
    rows: List[pd.DataFrame] = []
    desired = set(required_columns or ["fasta_filename", "seqid", "start", "end", "spore_related", "contig_len"])
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
            df = coerce_types(df)
            rows.append(df)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=list(desired))
    return pd.concat(rows, ignore_index=True)


def build_contig_table(fasta_dir_map: Dict[str, str],
                       fasta_files: List[str]) -> pd.DataFrame:
    """Build a table of contigs with lengths and optional spore unions.

    Columns: fasta_filename, seqid, contig_len, spore_union (list-of-tuples)
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
                    spore_union=[],
                )
            )

    df = pd.DataFrame([r.__dict__ for r in contig_rows])
    if df.empty:
        raise RuntimeError("No contigs found in provided FASTA directories.")
    return df


def build_fasta_to_spore_union_map_from_df(gff_df: pd.DataFrame,
                                          contigs_df: pd.DataFrame,
                                          allowed_genes: Optional[Set[str]] = None) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
    """Map (fasta_filename, FASTA seqid) -> merged spore intervals via contig length matching.

    Strategy per genome (fasta_filename):
      - Compute GFF contig lengths as max(end) per seqid from processed tables.
      - For a given length L, if there is exactly one GFF seqid with length L and
        exactly one FASTA seqid with length L, map them and attach that GFF seqid's
        merged spore intervals to the FASTA seqid.
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
    # GFF spore unions per (genome, seqid)
    spore_only = df[df["spore_related"].fillna(False)].copy()
    # Optional: restrict to retained genes by canonical gene/name
    if allowed_genes:
        if "gene" not in spore_only.columns:
            spore_only["gene"] = pd.Series(dtype="string")
        if "Name" not in spore_only.columns:
            spore_only["Name"] = pd.Series(dtype="string")
        spore_only["__gene_canon"] = spore_only["gene"].apply(canonical_gene_name)
        spore_only["__name_canon"] = spore_only["Name"].apply(canonical_gene_name)
        mask_allowed = spore_only["__gene_canon"].isin(allowed_genes) | spore_only["__name_canon"].isin(allowed_genes)
        spore_only = spore_only.loc[mask_allowed].copy()
        spore_only = spore_only.drop(columns=["__gene_canon", "__name_canon"], errors="ignore")
    spore_only["genome"] = spore_only["fasta_filename"].map(_genome_key)
    gff_union_map: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    if not spore_only.empty:
            # Key intervals by (genome_base, gff_seqid)
        for (genome_key, seqid), sub in spore_only.groupby(["genome", "seqid"], sort=False):
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


def build_fasta_spore_intervals_with_genes_map_from_df(gff_df: pd.DataFrame,
                                                       contigs_df: pd.DataFrame,
                                                       allowed_genes: Optional[Set[str]] = None) -> Dict[Tuple[str, str], List[Tuple[int, int, List[str], str]]]:
    """Map (fasta_filename, FASTA seqid) -> list of (start, end, [gene_names], strand_label) for spore loci unions.

    Gene name extraction priority: gene -> Name -> locus_tag. Names are ordered by locus start
    within each merged interval. Uses the same contig length matching strategy as
    build_fasta_to_spore_union_map_from_df.
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
    spore_only = df[df["spore_related"].fillna(False)].copy()
    # Optional: restrict to retained genes by canonical gene/name
    if allowed_genes:
        for c in ("gene", "Name", "locus_tag"):
            if c not in spore_only.columns:
                spore_only[c] = pd.Series(dtype="string")
        spore_only["__gene_canon"] = spore_only["gene"].apply(canonical_gene_name)
        spore_only["__name_canon"] = spore_only["Name"].apply(canonical_gene_name)
        mask_allowed = spore_only["__gene_canon"].isin(allowed_genes) | spore_only["__name_canon"].isin(allowed_genes)
        spore_only = spore_only.loc[mask_allowed].copy()
        spore_only = spore_only.drop(columns=["__gene_canon", "__name_canon"], errors="ignore")
    spore_only["genome"] = spore_only["fasta_filename"].map(_genome_key)
    # Build gene_label column with priority gene -> Name -> locus_tag
    for c in ("gene", "Name", "locus_tag"):
        if c not in spore_only.columns:
            spore_only[c] = pd.Series(dtype="string")
    spore_only["gene_label"] = spore_only["gene"].astype("string")
    spore_only.loc[spore_only["gene_label"].isna() | (spore_only["gene_label"].str.len() == 0), "gene_label"] = spore_only["Name"].astype("string")
    spore_only.loc[spore_only["gene_label"].isna() | (spore_only["gene_label"].str.len() == 0), "gene_label"] = spore_only["locus_tag"].astype("string")

    # Build merged intervals and aligned gene lists keyed by (genome, gff_seqid)
    gff_intervals_with_genes: Dict[Tuple[str, str], List[Tuple[int, int, List[str], str]]] = {}
    if not spore_only.empty:
        for (genome_key, seqid), sub in spore_only.groupby(["genome", "seqid"], sort=False):
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
        return pd.DataFrame(columns=["fasta_filename", "seqid", "start", "end", "spore_union"])

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
    ap.add_argument("--processed_gff_dir", type=str, default=DEFAULT_PROCESSED_GFF_DIR, help="Directory with processed GFF parquet/csv")
    ap.add_argument("--sporeinfo_csv", type=str, default=DEFAULT_SPOREINFO_CSV, help="CSV with columns file, ability_FALSE, ability_TRUE")
    ap.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN, help="Window length (bp)")
    ap.add_argument("--n_pos", type=int, default=DEFAULT_N_POS, help="Number of positive windows to sample")
    ap.add_argument("--n_neg", type=int, default=DEFAULT_N_NEG, help="Number of negative windows to sample")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for prepared data")
    ap.add_argument("--retained_flags", type=str, default=None, help="Optional retained_gene_flags.csv; if set, restrict ground truth to retained genes")
    ap.add_argument("--retained_rule", type=str, default=None, help="Rule to select retained genes (one of a/b/c). Default: none (no filtering)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # List FASTA files: test-only for sampling; all splits for genome_intervals
    test_fasta_map = list_fastas_from_dirs(list(args.test_dirs))
    test_fasta_files = sorted(test_fasta_map.keys())
    if not test_fasta_files:
        raise FileNotFoundError(f"No FASTA files found under test_dirs: {args.test_dirs}")
    all_fasta_map = list_fastas_from_dirs(list(args.all_dirs))
    all_fasta_files = sorted(all_fasta_map.keys())
    if not all_fasta_files:
        raise FileNotFoundError(f"No FASTA files found under all_dirs: {args.all_dirs}")

    # Load labels
    labels_df = pd.read_csv(args.sporeinfo_csv)
    labels_df = labels_df[["file", "ability_TRUE", "ability_FALSE"]].copy()
    labels_df["ability_TRUE"] = labels_df["ability_TRUE"].astype(int)
    labels_df["ability_FALSE"] = labels_df["ability_FALSE"].astype(int)

    present = set(test_fasta_files)
    labels_df = labels_df[labels_df["file"].isin(present)].copy()
    if labels_df.empty:
        raise RuntimeError("sporeinfo.csv does not overlap with test FASTA filenames.")

    pos_fastas = labels_df.loc[labels_df["ability_TRUE"] == 1, "file"].astype(str).tolist()
    neg_fastas = labels_df.loc[labels_df["ability_FALSE"] == 1, "file"].astype(str).tolist()
    if not pos_fastas:
        raise RuntimeError("No positive genomes found in sporeinfo for test set.")
    if not neg_fastas:
        raise RuntimeError("No negative genomes found in sporeinfo for test set.")

    # Build contig table (lengths and spore unions if available)
    contigs_test_df = build_contig_table(test_fasta_map, test_fasta_files)

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
    win_pos = win_pos.assign(label=1, group="pos")
    win_neg = win_neg.assign(label=0, group="neg")

    samples = pd.concat([win_pos, win_neg], ignore_index=True)
    samples.insert(0, "sample_id", np.arange(len(samples), dtype=int))

    # Build robust mapping from FASTA seqids to GFF spore unions via length matching (ALL genomes)
    print("Building FASTA->GFF contig mapping via length matching (all splits)...", flush=True)
    gff_df = read_gff_for_fastas(
        args.processed_gff_dir,
        all_fasta_files,
        required_columns=[
            "fasta_filename", "seqid", "start", "end", "strand", "spore_related", "contig_len",
            "gene", "Name", "locus_tag"
        ],
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
    fasta_to_union = build_fasta_to_spore_union_map_from_df(gff_df, contigs_all_df, allowed_genes=allowed_genes)
    fasta_to_items = build_fasta_spore_intervals_with_genes_map_from_df(gff_df, contigs_all_df, allowed_genes=allowed_genes)
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
        if int(getattr(r, "label")) == 1:
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

    # Keep only necessary columns in output; retain spore_union only if desired for debugging
    out_cols = [
        "sample_id", "group", "label", "fasta_filename", "seqid", "start", "end",
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
        "sporeinfo_csv": os.path.abspath(args.sporeinfo_csv),
        "samples_path": os.path.abspath(out_samples_path),
        "genomes_path": os.path.abspath(out_genomes_path),
    }
    with open(os.path.join(args.out_dir, "eval_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Prepared samples written to {out_samples_path}")
    print(f"Genome-level intervals written to {out_genomes_path}")
    print(f"Counts: pos={int(args.n_pos)}, neg={int(args.n_neg)}")


if __name__ == "__main__":
    main()

