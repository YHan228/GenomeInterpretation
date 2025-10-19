#!/usr/bin/env python3
"""Comprehensive data-quality inspection between FASTA files, processed GFF tables,
and original GFF FASTA sections.

Checks per genome (by basename):
- FASTA contigs: counts, total length, N50, basic stats
- Processed GFF contigs: counts, length determination (contig_len or max end)
- Length comparisons: unique-length matches, unmatched/ambiguous counts
- Sequence sanity: 42-mer prefix matches between FASTA contigs and raw GFF FASTA
  sequences (including reverse-complement)
- Optional visualizations: length histograms and sorted-length line plots

Outputs:
- Clear console summaries per genome and an aggregate summary
- Optional plots under --report_dir/plots
- JSON and CSV summaries under --report_dir

Defaults follow project layout but are configurable via CLI.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# --------------------------- Defaults ---------------------------

DATA_ROOT = Path("/vol/projects/BIFO/genomenet/yichen/phenotype/data")
DEFAULT_FASTA_DIR = DATA_ROOT / "test"
DEFAULT_PROCESSED_GFF_DIR = DATA_ROOT / "processed_gff"
DEFAULT_RAW_GFF_DIR = DATA_ROOT / "gff"
DEFAULT_REPORT_DIR = Path("sporulation/reports/data_quality")
DEFAULT_KMER_LEN = 42


# --------------------------- FASTA utilities ---------------------------

def _fasta_iter(path: Path) -> Iterable[Tuple[str, str]]:
    """Yield (header_id, sequence) for a FASTA file.

    Header id is the first whitespace-delimited token after '>'. Sequence is
    concatenated uppercase letters with ambiguous bases preserved.
    """
    header: Optional[str] = None
    chunks: List[str] = []
    with path.open("r") as fh:
        for line in fh:
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).upper()
                header = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            yield header, "".join(chunks).upper()


def read_fasta_contigs(path: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    for hid, seq in _fasta_iter(path):
        seqs[hid] = seq
    return seqs


def reverse_complement(seq: str) -> str:
    comp = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(comp)[::-1]


def n50(lengths: List[int]) -> int:
    if not lengths:
        return 0
    arr = sorted((int(x) for x in lengths), reverse=True)
    total = sum(arr)
    acc = 0
    for L in arr:
        acc += L
        if acc >= (total + 1) // 2:
            return L
    return arr[-1]


# --------------------------- Processed GFF utilities ---------------------------

def _find_processed_table(processed_gff_dir: Path, fasta_filename: str) -> Optional[Path]:
    """Locate a processed table by matching the base name of the FASTA file.

    Tries common extensions: .parquet/.pq/.feather/.csv/.tsv
    """
    base = os.path.splitext(fasta_filename)[0]
    for ext in (".parquet", ".pq", ".feather", ".csv", ".tsv"):
        p = processed_gff_dir / (base + ext)
        if p.exists():
            return p
    # try any extension with same stem
    candidates = list(processed_gff_dir.glob(base + ".*"))
    return candidates[0] if candidates else None


def _read_processed_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix.lower() == ".feather":
        return pd.read_feather(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def gff_contig_lengths_from_processed(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-seqid length for a processed GFF table.

    Uses `contig_len` if present (max per seqid), falling back to max(end).
    Returns a DataFrame with columns: seqid, gff_len (Int64), has_contig_len (bool).
    """
    cols = set(df.columns)
    if {"fasta_filename", "seqid"}.issubset(cols) is False:
        # ensure expected columns exist even if empty
        for c in ("fasta_filename", "seqid", "end", "contig_len"):
            if c not in df.columns:
                df[c] = pd.NA

    # max end per seqid as fallback
    end_max = df.groupby(["seqid"], sort=False)["end"].max().rename("end_max").reset_index()
    if "contig_len" in df.columns:
        contig_len_max = (
            df.groupby(["seqid"], sort=False)["contig_len"].max().rename("contig_len_max").reset_index()
        )
        g = contig_len_max.merge(end_max, on=["seqid"], how="outer")
        g["gff_len"] = g["contig_len_max"].fillna(g["end_max"]).astype("Int64")
        g["has_contig_len"] = g["contig_len_max"].notna()
    else:
        g = end_max.copy()
        g = g.rename(columns={"end_max": "gff_len"})
        g["gff_len"] = g["gff_len"].astype("Int64")
        g["has_contig_len"] = False
    return g[["seqid", "gff_len", "has_contig_len"]]


# --------------------------- Raw GFF FASTA extraction ---------------------------

def read_raw_gff_fasta_sequences(gff_path: Path) -> Dict[str, str]:
    """Parse the FASTA section of a GFF file (after a line starting with '##FASTA').

    Returns a dict seqid -> sequence (uppercase). If no FASTA section, returns empty dict.
    """
    if not gff_path.exists():
        return {}
    seqs: Dict[str, str] = {}
    in_fasta = False
    header: Optional[str] = None
    chunks: List[str] = []
    with gff_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not in_fasta:
                if line.startswith("##FASTA"):
                    in_fasta = True
                continue
            if line.startswith(">"):
                if header is not None:
                    seqs[header] = "".join(chunks).upper()
                header = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if in_fasta and header is not None:
            seqs[header] = "".join(chunks).upper()
    return seqs


def raw_gff_feature_bearing_seqids(gff_path: Path) -> List[str]:
    """Collect seqids that appear in feature lines (not FASTA section).

    This provides a view of how many contigs actually have annotated features in
    the raw GFF, which may be fewer than the contigs in the FASTA section.
    """
    if not gff_path.exists():
        return []
    seqids: set[str] = set()
    in_fasta = False
    with gff_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line:
                continue
            if line.startswith("##FASTA"):
                in_fasta = True
                break
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 9:
                seqids.add(parts[0])
    return sorted(seqids)


# --------------------------- Metrics and checks ---------------------------

def unique_length_match_stats(fa_lens: List[int], gff_lens: List[int]) -> Dict[str, int]:
    fa_counts = Counter(fa_lens)
    gff_counts = Counter(gff_lens)
    matched = 0
    fa_unique = 0
    gff_unique = 0
    ambiguous = 0
    for L, c in fa_counts.items():
        if c == 1:
            fa_unique += 1
        if L in gff_counts:
            if c == 1 and gff_counts[L] == 1:
                matched += 1
            else:
                ambiguous += min(c, gff_counts[L])
    for L, c in gff_counts.items():
        if c == 1:
            gff_unique += 1
    return {
        "matched_unique_lengths": matched,
        "fasta_unique_lengths": fa_unique,
        "gff_unique_lengths": gff_unique,
        "ambiguous_same_lengths": ambiguous,
    }


def kmer_prefix_match_stats(fa_seqs: Dict[str, str], gff_seqs: Dict[str, str], k: int) -> Dict[str, int]:
    if k <= 0:
        return {"tested": 0, "matched": 0, "too_short": 0}
    tested = 0
    matched = 0
    too_short = 0
    # Build set of prefix kmers from raw GFF FASTA and their reverse complements
    kmers: set[str] = set()
    for seq in gff_seqs.values():
        if len(seq) >= k:
            p = seq[:k]
            kmers.add(p)
            kmers.add(reverse_complement(p))
    for seq in fa_seqs.values():
        if len(seq) < k:
            too_short += 1
            continue
        tested += 1
        p = seq[:k]
        if p in kmers:
            matched += 1
    return {"tested": tested, "matched": matched, "too_short": too_short}


def basic_stats(lengths: List[int]) -> Dict[str, int]:
    if not lengths:
        return {"count": 0, "min": 0, "median": 0, "max": 0, "n50": 0, "total": 0}
    arr = sorted(lengths)
    total = sum(arr)
    mid = len(arr) // 2
    if len(arr) % 2 == 0:
        med = (arr[mid - 1] + arr[mid]) // 2
    else:
        med = arr[mid]
    return {
        "count": len(arr),
        "min": arr[0],
        "median": med,
        "max": arr[-1],
        "n50": n50(arr),
        "total": total,
    }


# --------------------------- Visualization ---------------------------

def maybe_plot_lengths(base: str, fa_lens: List[int], gff_lens: List[int], out_dir: Path) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Histogram
    bins = 30 if (fa_lens or gff_lens) else 1
    axes[0].hist(fa_lens, bins=bins, alpha=0.6, label="FASTA")
    axes[0].hist(gff_lens, bins=bins, alpha=0.6, label="GFF")
    axes[0].set_title("Contig length histogram")
    axes[0].set_xlabel("Length (bp)")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    # Sorted lengths line plot
    fa_sorted = sorted(fa_lens)
    gff_sorted = sorted(gff_lens)
    axes[1].plot(range(len(fa_sorted)), fa_sorted, label="FASTA")
    axes[1].plot(range(len(gff_sorted)), gff_sorted, label="GFF")
    axes[1].set_title("Sorted contig lengths")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Length (bp)")
    axes[1].legend()
    fig.suptitle(base)
    out_path = out_dir / f"{base}_lengths.png"
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    try:
        plt.close(fig)
    except Exception:
        pass
    return out_path


# --------------------------- Per-genome summary ---------------------------

@dataclass
class GenomeSummary:
    fasta_file: str
    raw_gff_file: Optional[str]
    processed_table: Optional[str]
    fa_count: int
    fa_total: int
    fa_n50: int
    gff_count: int
    gff_total: int
    gff_feature_contigs: int
    raw_gff_fasta_contigs: int
    gff_has_contig_len_rows: int
    gff_len_from_end_only: int
    matched_unique_lengths: int
    fasta_unique_lengths: int
    gff_unique_lengths: int
    ambiguous_same_lengths: int
    kmer_tested: int
    kmer_matched: int
    kmer_too_short: int
    raw_gff_has_fasta_section: bool
    plot_path: Optional[str]
    warnings: List[str]


# --------------------------- Main driver ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Check data quality between FASTA, processed GFF, and raw GFF FASTA sections")
    ap.add_argument("--fasta_dir", type=str, default=str(DEFAULT_FASTA_DIR), help="Directory with FASTA files")
    ap.add_argument("--processed_gff_dir", type=str, default=str(DEFAULT_PROCESSED_GFF_DIR), help="Directory with processed GFF tables")
    ap.add_argument("--raw_gff_dir", type=str, default=str(DEFAULT_RAW_GFF_DIR), help="Directory with raw .gff files (with FASTA section)")
    ap.add_argument("--report_dir", type=str, default=str(DEFAULT_REPORT_DIR), help="Output directory for reports and plots")
    ap.add_argument("--kmer_len", type=int, default=DEFAULT_KMER_LEN, help="K-mer length for prefix matching (e.g., 42)")
    ap.add_argument("--visualize", action="store_true", help="Generate per-genome plots of contig length distributions")
    ap.add_argument("--limit_genomes", type=int, default=0, help="Optional limit on number of genomes to process (0 = no limit)")
    args = ap.parse_args()

    fasta_dir = Path(args.fasta_dir)
    processed_dir = Path(args.processed_gff_dir)
    raw_gff_dir = Path(args.raw_gff_dir)
    report_dir = Path(args.report_dir)
    plots_dir = report_dir / "plots"

    report_dir.mkdir(parents=True, exist_ok=True)

    fasta_files = [p for p in sorted(fasta_dir.iterdir()) if p.is_file() and p.suffix.lower() in (".fasta", ".fa", ".fna")]
    if not fasta_files:
        raise SystemExit(f"No FASTA files found in {fasta_dir}")

    summaries: List[GenomeSummary] = []

    if args.limit_genomes and args.limit_genomes > 0:
        fasta_files = fasta_files[: args.limit_genomes]

    for idx, fa_path in enumerate(fasta_files, start=1):
        base = fa_path.stem
        print(f"[{idx}/{len(fasta_files)}] {fa_path.name}")

        # FASTA sequences and lengths
        fa_seqs = read_fasta_contigs(fa_path)
        fa_lens = [len(s) for s in fa_seqs.values()]
        fa_stats = basic_stats(fa_lens)

        # Processed GFF table (by matching base name)
        processed_path = _find_processed_table(processed_dir, fa_path.name)
        gff_len_list: List[int] = []
        gff_count = 0
        gff_total = 0
        has_contig_len_rows = 0
        len_from_end_only = 0
        proc_path_str: Optional[str] = None
        if processed_path and processed_path.exists():
            try:
                df = _read_processed_df(processed_path)
                g = gff_contig_lengths_from_processed(df)
                gff_len_list = [int(x) for x in g["gff_len"].dropna().astype(int).tolist()]
                has_contig_len_rows = int(g["has_contig_len"].sum())
                len_from_end_only = int((~g["has_contig_len"]).sum())
                gff_count = len(g)
                gff_total = sum(gff_len_list)
                proc_path_str = str(processed_path)
            except Exception as e:
                print(f"  ! Failed to read processed table {processed_path.name}: {e}")
        else:
            print(f"  ! Processed table not found for {fa_path.name} under {processed_dir}")

        # Unique-length matching stats
        ulm = unique_length_match_stats(fa_lens, gff_len_list)

        # Raw GFF: FASTA sequences and feature-bearing seqids
        raw_gff_path = raw_gff_dir / (base + ".gff")
        raw_gff_seqs = read_raw_gff_fasta_sequences(raw_gff_path)
        raw_gff_fasta_n = len(raw_gff_seqs)
        raw_gff_feature_seqids = raw_gff_feature_bearing_seqids(raw_gff_path)
        gff_feature_n = len(raw_gff_feature_seqids)
        kstats = kmer_prefix_match_stats(fa_seqs, raw_gff_seqs, int(args.kmer_len))

        warnings: List[str] = []
        if not raw_gff_seqs:
            warnings.append("raw_gff_missing_or_no_fasta_section")
        if gff_count == 0:
            warnings.append("processed_gff_missing_or_empty")
        if ulm["matched_unique_lengths"] == 0 and gff_count > 0 and fa_stats["count"] > 0:
            warnings.append("no_unique_length_matches")
        if kstats["tested"] > 0 and kstats["matched"] == 0:
            warnings.append("no_kmer_prefix_matches")

        plot_path: Optional[str] = None
        if args.visualize:
            p = maybe_plot_lengths(base, fa_lens, gff_len_list, plots_dir)
            plot_path = str(p) if p else None

        # Console summary for this genome
        print(f"  FASTA contigs: n={fa_stats['count']} total={fa_stats['total']} n50={fa_stats['n50']} min/med/max={fa_stats['min']}/{fa_stats['median']}/{fa_stats['max']}")
        print(f"  GFF contigs:   n={gff_count} total={gff_total} contig_len_rows={has_contig_len_rows} end_only_rows={len_from_end_only}")
        print(f"  Raw GFF: fasta_contigs={raw_gff_fasta_n} feature_contigs={gff_feature_n}")
        print(
            f"  Unique-length matches: {ulm['matched_unique_lengths']} (FASTA-unique={ulm['fasta_unique_lengths']}, GFF-unique={ulm['gff_unique_lengths']}, ambiguous={ulm['ambiguous_same_lengths']})"
        )
        if kstats["tested"] > 0:
            frac = (kstats["matched"] / max(1, kstats["tested"])) * 100.0
            print(f"  {args.kmer_len}-mer prefix matches: {kstats['matched']}/{kstats['tested']} ({frac:.1f}%), too_short={kstats['too_short']}")
        else:
            print(f"  {args.kmer_len}-mer prefix matches: not tested (no sequences long enough or no raw GFF FASTA)")
        if warnings:
            print(f"  Warnings: {', '.join(warnings)}")

        summaries.append(
            GenomeSummary(
                fasta_file=str(fa_path),
                raw_gff_file=str(raw_gff_path) if raw_gff_path.exists() else None,
                processed_table=proc_path_str,
                fa_count=fa_stats["count"],
                fa_total=fa_stats["total"],
                fa_n50=fa_stats["n50"],
                gff_count=gff_count,
                gff_total=gff_total,
                gff_feature_contigs=gff_feature_n,
                raw_gff_fasta_contigs=raw_gff_fasta_n,
                gff_has_contig_len_rows=has_contig_len_rows,
                gff_len_from_end_only=len_from_end_only,
                matched_unique_lengths=ulm["matched_unique_lengths"],
                fasta_unique_lengths=ulm["fasta_unique_lengths"],
                gff_unique_lengths=ulm["gff_unique_lengths"],
                ambiguous_same_lengths=ulm["ambiguous_same_lengths"],
                kmer_tested=kstats["tested"],
                kmer_matched=kstats["matched"],
                kmer_too_short=kstats["too_short"],
                raw_gff_has_fasta_section=bool(raw_gff_seqs),
                plot_path=plot_path,
                warnings=warnings,
            )
        )

    # Aggregate summary
    print("\n=== Aggregate summary ===")
    total_genomes = len(summaries)
    total_fa_contigs = sum(s.fa_count for s in summaries)
    total_gff_contigs = sum(s.gff_count for s in summaries)
    total_unique_length_matches = sum(s.matched_unique_lengths for s in summaries)
    genomes_no_unique_matches = sum(1 for s in summaries if s.matched_unique_lengths == 0 and s.gff_count > 0 and s.fa_count > 0)
    genomes_no_kmer_matches = sum(1 for s in summaries if s.kmer_tested > 0 and s.kmer_matched == 0)
    genomes_missing_raw_gff_fasta = sum(1 for s in summaries if not s.raw_gff_has_fasta_section)

    print(f"Genomes processed: {total_genomes}")
    print(f"Total FASTA contigs: {total_fa_contigs}")
    print(f"Total GFF contigs:   {total_gff_contigs}")
    print(f"Total unique-length matches: {total_unique_length_matches}")
    print(f"Genomes with zero unique-length matches: {genomes_no_unique_matches}")
    print(f"Genomes with zero {args.kmer_len}-mer prefix matches (when tested): {genomes_no_kmer_matches}")
    print(f"Genomes missing raw GFF FASTA section: {genomes_missing_raw_gff_fasta}")

    # Write JSON and CSV summaries
    per_genome_dicts = [asdict(s) for s in summaries]
    with (report_dir / "data_quality_summary.json").open("w") as fh:
        json.dump(per_genome_dicts, fh, indent=2)

    try:
        df_out = pd.DataFrame(per_genome_dicts)
        df_out.to_csv(report_dir / "data_quality_summary.csv", index=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()

