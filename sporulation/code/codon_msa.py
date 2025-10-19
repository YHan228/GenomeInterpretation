"""
Codon-aware MSAs for sporulation-related genes across test genomes using MACSE.

Workflow:
  1) Load species phenotype from sporeinfo.csv; keep Spore+ in test set.
  2) Load contig-level gene intervals from genome_intervals.parquet; subset to Spore+ test genomes.
  3) Tally gene presence across genomes; select genes with species frequency over a threshold (CLI: --min_frequency).
  4) Extract per-gene nucleotide CDS segments from genome FASTAs (strand-aware; trim to multiple of 3).
  5) Run MACSE codon-aware MSA per gene (NT and AA alignments).
  6) Compute cross-species conservation metrics from alignments and plot results.

Inputs:
  - test_dir: directory with genome FASTAs (one per species; filenames like GCA_..._genomic.fasta)
  - sporeinfo.csv: columns [file, ability_FALSE, ability_TRUE]
  - genome_intervals.parquet: columns include [fasta_filename, seqid, intervals_json, interval_genes_json, interval_strand_json]

Outputs (under outdir):
  - gene_frequency.csv: presence across genomes for sporulation-related genes
  - selected_genes.csv: genes chosen for MSA with counts
  - per-gene FASTAs under outdir/fastas/
  - MACSE outputs under outdir/alignments/{gene}/
  - conservation_summary.csv: per-gene metrics from alignments
  - plots/ frequency barplot and conservation vs frequency scatter

Notes:
  - Requires MACSE installed (conda package `bioconda::macse` typically provides `macse` entrypoint).
  - If `macse` is not on PATH, pass `--macse_jar` path to the macse v2 jar and ensure `java` is available.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns  # type: ignore
    except Exception:
        sns = None
except Exception as e:
    raise RuntimeError("matplotlib is required for plotting") from e


# --------------------------- Utilities ---------------------------

def _init_plotting() -> None:
    if 'MPLBACKEND' not in os.environ:
        os.environ['MPLBACKEND'] = 'Agg'
    if sns is not None:
        sns.set_theme(context="paper", style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _savefig(fig: "plt.Figure", out_base: str) -> None:
    fig.tight_layout()
    fig.savefig(out_base + ".png", bbox_inches="tight")
    try:
        fig.savefig(out_base + ".pdf", bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)


def read_sporeinfo_positive(sporeinfo_csv: str, test_fasta_basenames: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(sporeinfo_csv)
    required = {"file", "ability_TRUE"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"sporeinfo missing required columns: {missing}")
    df = df[["file", "ability_TRUE"]].copy()
    df["file"] = df["file"].astype(str)
    df["ability_TRUE"] = pd.to_numeric(df["ability_TRUE"], errors="coerce").fillna(0).astype(int)
    df = df[df["ability_TRUE"] == 1].copy()
    # Keep only files present in test set
    in_test = set(test_fasta_basenames)
    df = df[df["file"].isin(in_test)].copy()
    return df


def list_fastas_from_dirs(dirs: Sequence[str]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    checked_any = False
    for d in dirs:
        if not d:
            continue
        checked_any = True
        if not os.path.isdir(d):
            print(f"[codon_msa] Warning: directory not found: {d}", flush=True)
            continue
        for name in os.listdir(d):
            if not (name.endswith(".fasta") or name.endswith(".fa") or name.endswith(".fna")):
                continue
            # prefer first occurrence if duplicates across dirs
            paths.setdefault(name, os.path.join(d, name))
    if not checked_any or not paths:
        raise FileNotFoundError(f"No FASTA files found in provided directories: {dirs}")
    return paths


def load_intervals(intervals_path: str, allowed_fastas: Sequence[str]) -> pd.DataFrame:
    try:
        df = pd.read_parquet(intervals_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to read genome_intervals parquet. Ensure pyarrow is installed (conda install -c conda-forge pyarrow)."
        ) from e
    req = {"fasta_filename", "seqid", "intervals_json", "interval_genes_json", "interval_strand_json"}
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"intervals parquet missing required columns: {missing}")
    df = df[df["fasta_filename"].isin(set(allowed_fastas))].copy()
    # Parse JSON columns
    for col in ("intervals_json", "interval_genes_json", "interval_strand_json"):
        df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else (x if x is not None else []))
    return df


def canonical_gene_name(name: str) -> Optional[str]:
    if name is None:
        return None
    s = str(name).strip()
    if not s or s == "." or s.lower() == "none":
        return None
    s = s.lower()
    # drop common locus_tag-like suffixes (_1234)
    s = re.sub(r"_[0-9]+$", "", s)
    # unify sigma notations (sigf -> sigma_f)
    s = re.sub(r"^sig([a-z])$", r"sigma_\1", s)
    # restrict to sensible tokens
    if not re.match(r"^[a-z0-9_\-]+$", s):
        return None
    return s


# Curated sporulation-related keyword filter (case-insensitive) similar to analyze_gff
SPORULATION_KEYWORDS_RE = re.compile(
    r"(?i)\b(spo|ssp|cot|sigma|sig|ger|sleb|cwlj|dpa|spov|spoii|spo0)"
)


def gene_is_sporulation_related(gene: str) -> bool:
    if gene is None:
        return False
    return SPORULATION_KEYWORDS_RE.search(gene) is not None


@dataclass
class GeneOccurrence:
    fasta_filename: str
    seqid: str
    start: int
    end: int
    strand: str


def build_gene_index(df_intervals: pd.DataFrame, restrict_to_sporulation_keywords: bool = False) -> Dict[Tuple[str, str], List[GeneOccurrence]]:
    """Create index: (fasta_filename, canonical_gene) -> list of occurrences."""
    index: Dict[Tuple[str, str], List[GeneOccurrence]] = defaultdict(list)
    for _, row in df_intervals.iterrows():
        fasta_fn = str(row["fasta_filename"])  # species file
        seqid = str(row["seqid"])              # contig
        ivals = row["intervals_json"] or []
        genes = row["interval_genes_json"] or []
        strands = row["interval_strand_json"] or []
        # Normalize record shapes: each element may itself be a list of names
        for i, iv in enumerate(ivals):
            try:
                s, e = int(iv[0]), int(iv[1])
            except Exception:
                continue
            # genes per interval: list or string
            gnames = []
            if i < len(genes):
                gi = genes[i]
                if isinstance(gi, str):
                    gnames = [gi]
                elif isinstance(gi, list):
                    gnames = [str(x) for x in gi]
            strand = "+"
            if i < len(strands):
                st = strands[i]
                strand = str(st) if isinstance(st, str) and st in ("+", "-") else "+"
            # consider each name for this interval
            for g in gnames:
                cg = canonical_gene_name(g)
                if cg is None:
                    continue
                if restrict_to_sporulation_keywords and not gene_is_sporulation_related(cg):
                    continue
                index[(fasta_fn, cg)].append(GeneOccurrence(fasta_fn, seqid, s, e, strand))
    return index


def tally_gene_presence(index: Dict[Tuple[str, str], List[GeneOccurrence]]) -> pd.DataFrame:
    """Return DataFrame with columns: gene, species_count, total_occurrences."""
    gene_to_species: Dict[str, set] = defaultdict(set)
    gene_to_occ: Counter[str] = Counter()
    for (fasta_fn, gene), occs in index.items():
        gene_to_species[gene].add(fasta_fn)
        gene_to_occ[gene] += len(occs)
    rows = []
    for gene, species in gene_to_species.items():
        rows.append({
            "gene": gene,
            "species_count": len(species),
            "total_occurrences": int(gene_to_occ[gene]),
        })
    freq = pd.DataFrame(rows).sort_values(["species_count", "total_occurrences"], ascending=[False, False])
    return freq


def select_genes_for_alignment(freq: pd.DataFrame, min_frequency: int) -> pd.DataFrame:
    """Select genes with species frequency strictly greater than min_frequency.

    Returns a DataFrame with selected genes sorted by species_count desc then total_occurrences desc.
    """
    if freq.empty:
        return freq
    cand = freq[freq["species_count"] > int(min_frequency)].copy()
    cand = cand.sort_values(["species_count", "total_occurrences"], ascending=[False, False])
    return cand.reset_index(drop=True)


# --------------------------- FASTA I/O ---------------------------

def reverse_complement(seq: str) -> str:
    comp = str.maketrans("ACGTUMRWSYKVHDBNacgtumrwsykvhdbn", "TGCAAKYWSRMBDHVNtgcaakywsrmbdhvn")
    return seq.translate(comp)[::-1]


def read_fasta_as_dict(path: str) -> Dict[str, str]:
    contigs: Dict[str, List[str]] = {}
    header: Optional[str] = None
    with open(path, "r") as fh:
        for line in fh:
            if not line:
                continue
            if line.startswith(">"):
                # header up to first whitespace
                header = line[1:].strip().split()[0]
                if header not in contigs:
                    contigs[header] = []
            else:
                if header is None:
                    continue
                contigs[header].append(line.strip())
    return {h: "".join(seq_list).upper() for h, seq_list in contigs.items()}


def extract_cds(seq: str, start: int, end: int, strand: str) -> str:
    # Coordinates expected 1-based inclusive
    s0 = max(1, int(start)) - 1
    e0 = min(len(seq), int(end))
    if e0 <= s0:
        return ""
    subseq = seq[s0:e0]
    if strand == "-":
        subseq = reverse_complement(subseq)
    # Trim to multiple of 3 from the end
    r = len(subseq) % 3
    if r != 0:
        subseq = subseq[:len(subseq)-r]
    return subseq


# --------------------------- MACSE wrapper ---------------------------

def detect_macse(macse_cmd: Optional[str], macse_jar: Optional[str]) -> Tuple[List[str], str]:
    """
    Returns (base_command_list, mode) where mode in {"cmd", "jar"}.
    Raises if neither available.
    """
    if macse_cmd:
        exe = shutil.which(macse_cmd)
        if exe:
            return [exe], "cmd"
    # default try PATH 'macse'
    exe = shutil.which("macse")
    if exe:
        return [exe], "cmd"
    # fallback to jar
    if macse_jar:
        jar = os.path.abspath(macse_jar)
        if not os.path.exists(jar):
            raise FileNotFoundError(f"MACSE jar not found: {jar}")
        java = shutil.which("java")
        if not java:
            raise FileNotFoundError("java not found on PATH; required to run MACSE jar")
        return [java, "-jar", jar], "jar"
    raise FileNotFoundError("MACSE not found. Provide --macse or --macse_jar, or ensure 'macse' is on PATH.")


def run_macse_align(in_fasta: str, out_nt: str, out_aa: str, macse_cmd: Optional[str], macse_jar: Optional[str]) -> None:
    base_cmd, _mode = detect_macse(macse_cmd, macse_jar)
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_nt), exist_ok=True)
    os.makedirs(os.path.dirname(out_aa), exist_ok=True)
    cmd = [*base_cmd, "-prog", "alignSequences", "-seq", os.path.abspath(in_fasta), "-out_NT", os.path.abspath(out_nt), "-out_AA", os.path.abspath(out_aa)]
    # Do not set timeouts; let caller manage runtime
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"MACSE failed for {in_fasta}:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")


# --------------------------- Conservation metrics ---------------------------

def _read_alignment(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    seqs: Dict[str, List[str]] = {}
    name: Optional[str] = None
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith(">"):
                name = line[1:].strip()
                if name not in seqs:
                    seqs[name] = []
            else:
                if name is None:
                    continue
                seqs[name].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def column_conservation(aln_seqs: List[str], alphabet: str, treat_unknown_as_gap: bool = True) -> float:
    """Mean per-column conservation (majority character frequency among non-gaps)."""
    if not aln_seqs:
        return float("nan")
    L = len(aln_seqs[0])
    if L == 0:
        return float("nan")
    arr = np.array([list(s) for s in aln_seqs])
    gaps = set(["-"])
    if treat_unknown_as_gap:
        gaps.update(["N", "n", "X", "x", "?"])
    cons = []
    for j in range(L):
        col = arr[:, j]
        mask = np.array([c not in gaps for c in col])
        vals = col[mask]
        if vals.size == 0:
            continue
        # count
        uniq, counts = np.unique(vals, return_counts=True)
        cons.append(int(counts.max()) / float(vals.size))
    if not cons:
        return float("nan")
    return float(np.mean(cons))


def mean_pairwise_identity(aln_seqs: List[str], treat_gap_as_mismatch: bool = True, unknown: Iterable[str] = ("N", "n", "X", "x", "?")) -> float:
    if not aln_seqs or len(aln_seqs) < 2:
        return float("nan")
    n = len(aln_seqs)
    L = len(aln_seqs[0])
    if any(len(s) != L for s in aln_seqs):
        return float("nan")
    unknown_set = set(unknown)
    def pair_ident(a: str, b: str) -> float:
        matches = 0
        valid = 0
        for ca, cb in zip(a, b):
            if ca in unknown_set or cb in unknown_set:
                continue
            if ca == '-' or cb == '-':
                valid += 1
                if not treat_gap_as_mismatch:
                    matches += 1
                continue
            valid += 1
            if ca == cb:
                matches += 1
        return (matches / valid) if valid else float('nan')
    vals = []
    for i in range(n):
        for j in range(i+1, n):
            vals.append(pair_ident(aln_seqs[i], aln_seqs[j]))
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float('nan')


# --------------------------- Main pipeline ---------------------------

def run_pipeline(
    genome_dirs: Sequence[str],
    sporeinfo_csv: str,
    intervals_path: str,
    outdir: str,
    min_frequency: int,
    macse_cmd: Optional[str],
    macse_jar: Optional[str],
    n_workers: int,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    # If cached CSVs are present, skip re-computation and only plot
    summary_csv = os.path.join(outdir, "conservation_summary.csv")
    selected_csv = os.path.join(outdir, "selected_genes.csv")
    plot_dir = os.path.join(outdir, "plots")
    if os.path.exists(summary_csv) and os.path.exists(selected_csv):
        print(f"[codon_msa] Found cached results (CSV). Skipping alignment; generating plots only.", flush=True)
        os.makedirs(plot_dir, exist_ok=True)
        selected = pd.read_csv(selected_csv)
        summary = pd.read_csv(summary_csv)
        # Plots from cached data
        _init_plotting()
        # 1) Frequency barplot (selected genes)
        fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(selected)), 4))
        sf = selected.copy()
        sf = sf.sort_values(["species_count", "total_occurrences"], ascending=[True, True])
        ax.barh(sf["gene"], sf["species_count"], color="#1f77b4")
        ax.set_xlabel("Species count (Spore+ test)")
        ax.set_ylabel("Gene")
        ax.set_title("Frequent sporulation-related genes")
        _savefig(fig, os.path.join(plot_dir, "gene_frequency_selected"))

        # 2) Conservation vs frequency scatter (AA)
        sc = summary.copy()
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.scatter(sc.get("species_count", pd.Series(dtype=float)), sc.get("aa_mean_col_conservation", pd.Series(dtype=float)), s=40, alpha=0.9)
        ax2.set_xlabel("Species count (Spore+ test)")
        ax2.set_ylabel("Mean AA column conservation")
        ax2.set_ylim(0, 1.05)
        ax2.set_title("Conservation vs frequency")
        _savefig(fig2, os.path.join(plot_dir, "conservation_vs_frequency"))

        # 3) NT column conservation vs NT pairwise identity scatter
        if {"nt_mean_col_conservation", "nt_mean_pairwise_identity"}.issubset(sc.columns):
            fig3, ax3 = plt.subplots(figsize=(6, 5))
            ax3.scatter(sc["nt_mean_col_conservation"], sc["nt_mean_pairwise_identity"], s=40, alpha=0.9)
            ax3.set_xlabel("NT mean column conservation")
            ax3.set_ylabel("NT mean pairwise identity")
            ax3.set_xlim(0, 1.05)
            ax3.set_ylim(0, 1.05)
            ax3.set_title("NT conservation vs NT pairwise identity")
            _savefig(fig3, os.path.join(plot_dir, "nt_conservation_vs_nt_pairwise_identity"))
        else:
            print("[codon_msa] Warning: NT metrics not found in summary CSV; skipping NT-vs-NT plot.", flush=True)
        return
    print(f"[codon_msa] Listing FASTA genomes under: {', '.join(os.path.abspath(d) for d in genome_dirs)}", flush=True)
    fasta_map = list_fastas_from_dirs(genome_dirs)
    print(f"[codon_msa] Found {len(fasta_map)} FASTA files across provided directories.", flush=True)
    positive = read_sporeinfo_positive(sporeinfo_csv, list(fasta_map.keys()))
    if positive.empty:
        raise RuntimeError("No Spore+ genomes found in test set per sporeinfo.csv")
    pos_basenames = positive["file"].tolist()
    print(f"[codon_msa] Spore+ genomes present in test set: {len(pos_basenames)}", flush=True)

    # Load intervals for Spore+ genomes only
    df_ivals = load_intervals(intervals_path, pos_basenames)
    if df_ivals.empty:
        raise RuntimeError("No interval records for Spore+ genomes in intervals parquet")
    print(f"[codon_msa] Loaded intervals table: shape={df_ivals.shape}", flush=True)

    # Build gene index and tally frequencies
    # Intervals were built from spore-related loci only (see eval_data.py), so no additional regex filter.
    gene_index = build_gene_index(df_ivals, restrict_to_sporulation_keywords=False)
    freq = tally_gene_presence(gene_index)
    freq.to_csv(os.path.join(outdir, "gene_frequency.csv"), index=False)

    selected = select_genes_for_alignment(freq, min_frequency=min_frequency)
    if selected.empty:
        raise RuntimeError("No genes selected for alignment given thresholds")
    selected.to_csv(os.path.join(outdir, "selected_genes.csv"), index=False)
    print(f"[codon_msa] Selected {len(selected)} genes for MACSE (species_count > {min_frequency}).", flush=True)

    # Prepare directories
    fasta_out_dir = os.path.join(outdir, "fastas")
    aln_out_dir = os.path.join(outdir, "alignments")
    plot_dir = os.path.join(outdir, "plots")
    os.makedirs(fasta_out_dir, exist_ok=True)
    os.makedirs(aln_out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Cache contigs per genome only when needed
    contig_cache: Dict[str, Dict[str, str]] = {}

    # Extract sequences per gene
    gene_to_input_fa: Dict[str, str] = {}
    processed_files: set = set()  # unique genome FASTA basenames that yielded at least one CDS
    for gene in selected["gene"].tolist():
        print(f"[codon_msa] Preparing input FASTA for gene '{gene}'...", flush=True)
        # Aggregate occurrences per genome; pick the longest occurrence per genome
        by_genome: Dict[str, GeneOccurrence] = {}
        for (fasta_fn, g), occs in gene_index.items():
            if g != gene:
                continue
            # retain only genomes that are Spore+ in test set
            if fasta_fn not in pos_basenames:
                continue
            best = max(occs, key=lambda o: (abs(o.end - o.start + 1)))
            by_genome[fasta_fn] = best
        if not by_genome:
            continue

        out_fa = os.path.join(fasta_out_dir, f"{gene}.fa")
        n_written = 0
        with open(out_fa, "w") as w:
            for fasta_fn, occ in sorted(by_genome.items()):
                fasta_path = fasta_map.get(fasta_fn)
                if not fasta_path or not os.path.exists(fasta_path):
                    continue
                # load contigs if not cached
                if fasta_fn not in contig_cache:
                    contig_cache[fasta_fn] = read_fasta_as_dict(fasta_path)
                contigs = contig_cache[fasta_fn]
                seq = contigs.get(occ.seqid)
                if not seq:
                    # Some headers may include version suffix; try to find by prefix match
                    # Fallback: try exact match over split by whitespace
                    candidates = [k for k in contigs.keys() if occ.seqid in k or k in occ.seqid]
                    seq = contigs.get(candidates[0]) if candidates else None
                if not seq:
                    continue
                cds = extract_cds(seq, occ.start, occ.end, occ.strand)
                if len(cds) < 30:  # skip very short
                    continue
                header = f">{fasta_fn}|{occ.seqid}:{occ.start}-{occ.end}({occ.strand})"
                w.write(header + "\n")
                # wrap to 80 cols
                for i in range(0, len(cds), 80):
                    w.write(cds[i:i+80] + "\n")
                n_written += 1
                if fasta_fn not in processed_files:
                    processed_files.add(fasta_fn)
                    if len(processed_files) % 100 == 0:
                        print(f"[codon_msa] Processed {len(processed_files)} genome FASTA files so far...", flush=True)
        if n_written >= 2:
            gene_to_input_fa[gene] = out_fa
            print(f"[codon_msa] Wrote {n_written} CDS sequences for gene '{gene}'.", flush=True)
        else:
            print(f"[codon_msa] Skipping gene '{gene}' (only {n_written} valid sequences).", flush=True)

    if not gene_to_input_fa:
        raise RuntimeError("No input FASTAs with >=2 sequences were generated for selected genes.")
    if processed_files:
        print(f"[codon_msa] Completed sequence extraction for {len(processed_files)} unique genome FASTA files.", flush=True)

    # Run MACSE and collect conservation metrics (parallel across genes)
    def _align_and_summarize_gene(gene: str, in_fa: str) -> Dict[str, object]:
        gene_dir = os.path.join(aln_out_dir, gene)
        os.makedirs(gene_dir, exist_ok=True)
        out_nt = os.path.join(gene_dir, f"{gene}_NT.fasta")
        out_aa = os.path.join(gene_dir, f"{gene}_AA.fasta")
        try:
            print(f"[codon_msa] Running MACSE for gene '{gene}' (input: {os.path.basename(in_fa)})...", flush=True)
            run_macse_align(in_fa, out_nt, out_aa, macse_cmd=macse_cmd, macse_jar=macse_jar)
            print(f"[codon_msa] MACSE finished for gene '{gene}'.", flush=True)
            aln_nt = _read_alignment(out_nt)
            aln_aa = _read_alignment(out_aa)
            nseq = max(len(aln_nt), len(aln_aa))
            nt_len = max((len(s) for s in aln_nt.values()), default=0)
            aa_len = max((len(s) for s in aln_aa.values()), default=0)
            nt_cons = column_conservation(list(aln_nt.values()), alphabet="DNA") if aln_nt else float("nan")
            aa_cons = column_conservation(list(aln_aa.values()), alphabet="AA") if aln_aa else float("nan")
            nt_pid = mean_pairwise_identity(list(aln_nt.values())) if aln_nt else float("nan")
            aa_pid = mean_pairwise_identity(list(aln_aa.values())) if aln_aa else float("nan")
            return {
                "gene": gene,
                "n_seqs": nseq,
                "nt_aln_len": nt_len,
                "aa_aln_len": aa_len,
                "nt_mean_col_conservation": nt_cons,
                "aa_mean_col_conservation": aa_cons,
                "nt_mean_pairwise_identity": nt_pid,
                "aa_mean_pairwise_identity": aa_pid,
                "status": "ok",
            }
        except Exception as e:
            print(f"[codon_msa] ERROR during MACSE for gene '{gene}': {e}", flush=True)
            return {
                "gene": gene,
                "n_seqs": 0,
                "nt_aln_len": 0,
                "aa_aln_len": 0,
                "nt_mean_col_conservation": float("nan"),
                "aa_mean_col_conservation": float("nan"),
                "nt_mean_pairwise_identity": float("nan"),
                "aa_mean_pairwise_identity": float("nan"),
                "status": f"error: {e}",
            }

    summary_rows: List[Dict[str, object]] = []
    tasks = list(gene_to_input_fa.items())
    total_tasks = len(tasks)
    n_workers = max(1, int(n_workers))
    print(f"[codon_msa] Launching MACSE across {total_tasks} genes with n_workers={n_workers}...", flush=True)
    if n_workers > 1 and total_tasks > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_align_and_summarize_gene, gene, in_fa): gene for gene, in_fa in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                summary_rows.append(row)
                if (i % 1) == 0:
                    print(f"[codon_msa] Alignment progress: {i}/{total_tasks} genes completed.", flush=True)
    else:
        for i, (gene, in_fa) in enumerate(tasks, 1):
            row = _align_and_summarize_gene(gene, in_fa)
            summary_rows.append(row)
            print(f"[codon_msa] Alignment progress: {i}/{total_tasks} genes completed.", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary = summary.merge(selected[["gene", "species_count", "total_occurrences"]], on="gene", how="left")
    summary = summary.sort_values(["aa_mean_col_conservation", "species_count"], ascending=[False, False])
    summary.to_csv(os.path.join(outdir, "conservation_summary.csv"), index=False)

    # Plots
    _init_plotting()
    # 1) Frequency barplot (selected genes)
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(selected)), 4))
    sf = selected.copy()
    sf = sf.sort_values(["species_count", "total_occurrences"], ascending=[True, True])
    ax.barh(sf["gene"], sf["species_count"], color="#1f77b4")
    ax.set_xlabel("Species count (Spore+ test)")
    ax.set_ylabel("Gene")
    ax.set_title("Frequent sporulation-related genes")
    _savefig(fig, os.path.join(plot_dir, "gene_frequency_selected"))

    # 2) Conservation vs frequency scatter
    sc = summary.copy()
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(sc["species_count"], sc["aa_mean_col_conservation"], s=40, alpha=0.9)
    for _, r in sc.iterrows():
        ax2.text(r["species_count"] + 0.05, r["aa_mean_col_conservation"], r["gene"], fontsize=8, alpha=0.8)
    ax2.set_xlabel("Species count (Spore+ test)")
    ax2.set_ylabel("Mean AA column conservation")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Conservation vs frequency")
    _savefig(fig2, os.path.join(plot_dir, "conservation_vs_frequency"))

    # 3) NT column conservation vs NT pairwise identity scatter
    if {"nt_mean_col_conservation", "nt_mean_pairwise_identity"}.issubset(sc.columns):
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        ax3.scatter(sc["nt_mean_col_conservation"], sc["nt_mean_pairwise_identity"], s=40, alpha=0.9)
        ax3.set_xlabel("NT mean column conservation")
        ax3.set_ylabel("NT mean pairwise identity")
        ax3.set_xlim(0, 1.05)
        ax3.set_ylim(0, 1.05)
        ax3.set_title("NT conservation vs NT pairwise identity")
        _savefig(fig3, os.path.join(plot_dir, "nt_conservation_vs_nt_pairwise_identity"))


def main():
    ap = argparse.ArgumentParser(description="Codon-aware MSAs for sporulation-related genes using MACSE")
    ap.add_argument("--genome_dirs", nargs='+', default=[
        "/vol/projects/BIFO/genomenet/yichen/phenotype/data/train",
        "/vol/projects/BIFO/genomenet/yichen/phenotype/data/validation",
    ], help="One or more directories with FASTA genomes (e.g., train and validation)")
    ap.add_argument("--sporeinfo", default="/home/yhan/GenomeInterpretation/sporulation/sporeinfo.csv", help="Path to sporeinfo.csv")
    ap.add_argument("--intervals", default="/vol/projects/BIFO/genomenet/yichen/phenotype/data/eval/genome_intervals.parquet", help="Path to genome_intervals.parquet")
    ap.add_argument("--outdir", default="/home/yhan/GenomeInterpretation/sporulation/analysis_out/codon_msa", help="Output directory")
    ap.add_argument("--min_frequency", type=int, default=10, help="Select genes with species_count strictly greater than this value")
    ap.add_argument("--macse", default=None, help="MACSE command on PATH (e.g., 'macse'). If unset, tries PATH or --macse_jar")
    ap.add_argument("--macse_jar", default=None, help="Path to macse v2 jar; requires 'java' on PATH")
    ap.add_argument("--n_workers", type=int, default=8, help="Number of parallel workers for MACSE (genes in parallel)")
    args = ap.parse_args()

    run_pipeline(
        genome_dirs=args.genome_dirs,
        sporeinfo_csv=args.sporeinfo,
        intervals_path=args.intervals,
        outdir=args.outdir,
        min_frequency=args.min_frequency,
        macse_cmd=args.macse,
        macse_jar=args.macse_jar,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()

