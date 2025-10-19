"""
Analyze retention of spore-related genes under two filters and their intersection:

Filters:
  a) Gene belongs to a family/term marked significant in the pooled predictive
     enrichment (from family_enrichment_data_driven_pooled.csv; those
     visualized in plot 'family_enrichment_volcano_data_driven_pooled').
  b) Gene has NT mean pairwise identity > threshold (default 0.4) from
     codon-aware alignment summary (conservation_summary.csv; those
     visualized in plot 'nt_conservation_vs_nt_pairwise_identity').
  c) Both a and b.

Inputs (CSV paths):
  --fam_pooled: path to family_enrichment_data_driven_pooled.csv (preferred)
  --fam_dd:     (deprecated) path to family_enrichment_data_driven.csv
  --summary:    path to conservation_summary.csv (from codon_msa.py)

Outputs:
  - Prints counts and percentages for filters a, b, and c.
  - Optionally writes a CSV with per-gene flags via --out_csv.

Example:
  python retained_spore_genes_analysis.py \
      --fam_pooled /path/to/family_enrichment_data_driven_pooled.csv \
      --summary /path/to/conservation_summary.csv \
      --pid_thr 0.4 --fdr 0.05 --match_mode exact \
      --out_csv /path/to/retained_gene_flags.csv
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Iterable, Optional, Set, Tuple

import pandas as pd


def canonical_gene_name(name: Optional[str]) -> Optional[str]:
    """Normalize a gene/family token for robust matching.

    Mirrors the normalization used in codon_msa.canonical_gene_name.
    """
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


def load_significant_terms(fam_csv: str, fdr: float) -> Set[str]:
    """Load pooled predictive enriched families/terms with FDR <= threshold.

    Returns a set of canonicalized term strings.
    """
    df = pd.read_csv(fam_csv)
    if "family" not in df.columns or "fdr_bh" not in df.columns:
        raise ValueError("Input enrichment CSV must contain 'family' and 'fdr_bh' columns")
    df = df.copy()
    df = df[df["family"].notna()]
    df = df[pd.to_numeric(df["fdr_bh"], errors="coerce") <= float(fdr)]
    terms: Set[str] = set()
    for t in df["family"].astype(str):
        ct = canonical_gene_name(t)
        if ct:
            terms.add(ct)
    return terms


def load_conservation_summary(summary_csv: str) -> pd.DataFrame:
    """Load conservation summary with at least columns 'gene' and 'nt_mean_pairwise_identity'."""
    df = pd.read_csv(summary_csv)
    req = {"gene", "nt_mean_pairwise_identity"}
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"conservation_summary.csv missing required columns: {missing}")
    # Canonicalize gene names for matching
    df = df.copy()
    df["gene_canon"] = df["gene"].apply(canonical_gene_name)
    df = df[df["gene_canon"].notna()]
    # Ensure numeric
    df["nt_mean_pairwise_identity"] = pd.to_numeric(df["nt_mean_pairwise_identity"], errors="coerce")
    return df


def gene_matches_terms(gene_canon: str, terms: Set[str], mode: str = "exact") -> bool:
    """Return True if gene matches any significant term under the provided mode.

    Modes:
      - exact: gene == term
      - prefix: gene startswith term or term startswith gene (to tolerate minor variants)
      - substring: term in gene (case already normalized)
    """
    if not terms:
        return False
    if mode == "exact":
        return gene_canon in terms
    if mode == "prefix":
        for t in terms:
            if gene_canon.startswith(t) or t.startswith(gene_canon):
                return True
        return False
    if mode == "substring":
        for t in terms:
            if t in gene_canon:
                return True
        return False
    raise ValueError("Unsupported match_mode. Use one of: exact, prefix, substring.")


def analyze_retention(
    fam_csv: str,
    summary_csv: str,
    pid_thr: float = 0.4,
    fdr: float = 0.05,
    match_mode: str = "exact",
) -> Tuple[pd.DataFrame, pd.Series]:
    terms = load_significant_terms(fam_csv, fdr=fdr)
    summary = load_conservation_summary(summary_csv)

    # Evaluate filters per gene (unique genes in summary)
    # Use the best available nt_mean_pairwise_identity per gene (if multiple rows exist)
    agg = (
        summary
        .groupby(["gene", "gene_canon"], as_index=False)["nt_mean_pairwise_identity"]
        .max()
    )

    agg["pass_a_family_sig"] = agg["gene_canon"].apply(lambda g: gene_matches_terms(g, terms, mode=match_mode))
    agg["pass_b_nt_pid"] = agg["nt_mean_pairwise_identity"].apply(lambda v: pd.notna(v) and float(v) > float(pid_thr))
    agg["pass_c_both"] = agg["pass_a_family_sig"] & agg["pass_b_nt_pid"]

    total_genes = len(agg)
    counts = pd.Series({
        "total_genes": total_genes,
        "filter_a_family_sig": int(agg["pass_a_family_sig"].sum()),
        "filter_b_nt_pid": int(agg["pass_b_nt_pid"].sum()),
        "filter_c_both": int(agg["pass_c_both"].sum()),
    })
    return agg, counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze retention of spore-related genes under filters (a), (b), and (c)")
    ap.add_argument("--fam_pooled", required=False, help="Path to family_enrichment_data_driven_pooled.csv (preferred)")
    ap.add_argument("--fam_dd", required=False, help="(deprecated) Path to family_enrichment_data_driven.csv")
    ap.add_argument("--summary", required=True, help="Path to conservation_summary.csv")
    ap.add_argument("--pid_thr", type=float, default=0.4, help="NT mean pairwise identity threshold (default 0.4)")
    ap.add_argument("--fdr", type=float, default=0.05, help="FDR threshold for significant terms (default 0.05)")
    ap.add_argument(
        "--match_mode",
        choices=["exact", "prefix", "substring"],
        default="exact",
        help="How to match gene names against significant terms (default: exact)",
    )
    ap.add_argument("--out_csv", default=None, help="Optional path to write per-gene flags CSV")
    args = ap.parse_args()

    fam_csv = args.fam_pooled or args.fam_dd
    if not fam_csv:
        raise SystemExit("Please provide --fam_pooled (preferred) or --fam_dd path to the enrichment CSV.")

    gene_flags, counts = analyze_retention(
        fam_csv=fam_csv,
        summary_csv=args.summary,
        pid_thr=args.pid_thr,
        fdr=args.fdr,
        match_mode=args.match_mode,
    )

    total = counts["total_genes"]
    a = counts["filter_a_family_sig"]
    b = counts["filter_b_nt_pid"]
    c = counts["filter_c_both"]

    def pct(x: int, d: int) -> str:
        return f"{(100.0 * x / d):.1f}%" if d else "n/a"

    print("Retention analysis (unique genes):")
    print(f"  Total genes: {total}")
    print(f"  a) Family significant (FDR<= {args.fdr}): {a} ({pct(a, total)})")
    print(f"  b) NT mean pairwise identity > {args.pid_thr}: {b} ({pct(b, total)})")
    print(f"  c) Both a and b: {c} ({pct(c, total)})")

    if args.out_csv:
        out_path = os.path.abspath(args.out_csv)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        gene_flags.rename(columns={"gene_canon": "gene_canonical"}, inplace=True)
        gene_flags.to_csv(out_path, index=False)
        print(f"Wrote per-gene flags to: {out_path}")


if __name__ == "__main__":
    main()


