#!/usr/bin/env python3
"""Annotate Rashomon frequent genes with functional descriptions from GFF files.

Reads rashomon_features.csv and matches genes to product annotations from
processed GFF parquet files. Outputs a table for manual biological assessment.

Usage:
    python rashomon_annotate.py --phenotype "Spore formation"
    python rashomon_annotate.py --phenotype "Motility"
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from tqdm import tqdm


# Paths
PROCESSED_GFF_DIR = Path("/vol/projects/BIFO/genomenet/yichen/phenotype/data/processed_gff")
RASHOMON_RESULTS_DIR = Path("sporulation/results/rashomon")


def build_gene_annotation_lookup(gff_dir: Path, sample_limit: Optional[int] = None) -> Dict[str, str]:
    """Build gene -> product annotation lookup from processed GFF files.

    For genes with multiple annotations across genomes, keep the most common one.
    """
    gene_products: Dict[str, Counter] = {}

    gff_files = list(gff_dir.glob("*.parquet"))
    if sample_limit:
        gff_files = gff_files[:sample_limit]

    for gff_file in tqdm(gff_files, desc="Reading GFFs"):
        try:
            df = pd.read_parquet(gff_file, columns=["gene", "product", "canonical_gene_names"])
            # Use canonical gene names (lowercase, semicolon-separated tokens)
            for _, row in df.dropna(subset=["product"]).iterrows():
                product = str(row["product"]).strip()
                if not product or product == "hypothetical protein":
                    continue
                # Extract canonical tokens
                canonical = row.get("canonical_gene_names", "")
                if pd.notna(canonical):
                    for token in str(canonical).split(";"):
                        token = token.strip().lower()
                        if token and len(token) > 1:
                            if token not in gene_products:
                                gene_products[token] = Counter()
                            gene_products[token][product] += 1
                # Also use the gene column directly
                gene = row.get("gene", "")
                if pd.notna(gene):
                    gene_lower = str(gene).strip().lower()
                    # Remove suffix like _1, _2 for base matching
                    base_gene = gene_lower.rstrip("_0123456789").rstrip("_")
                    for g in [gene_lower, base_gene]:
                        if g and len(g) > 1:
                            if g not in gene_products:
                                gene_products[g] = Counter()
                            gene_products[g][product] += 1
        except Exception as e:
            continue

    # Keep most common annotation per gene
    lookup = {}
    for gene, counter in gene_products.items():
        if counter:
            lookup[gene] = counter.most_common(1)[0][0]

    return lookup


def annotate_rashomon_genes(phenotype: str, freq_threshold: float = 0.5) -> pd.DataFrame:
    """Load Rashomon results and annotate frequent genes."""

    phen_safe = phenotype.replace(" ", "_")
    results_dir = RASHOMON_RESULTS_DIR / phen_safe
    features_file = results_dir / "rashomon_features.csv"

    if not features_file.exists():
        raise FileNotFoundError(f"Results not found: {features_file}")

    # Load features
    df = pd.read_csv(features_file)

    # Filter for frequent genes (both methods >= threshold)
    frequent = df[(df["freq_logistic"] >= freq_threshold) & (df["freq_rf"] >= freq_threshold)].copy()
    frequent = frequent.sort_values("freq_logistic", ascending=False)

    print(f"Found {len(frequent)} genes with freq >= {freq_threshold} in both methods")

    # Build annotation lookup
    print("Building gene annotation lookup from GFF files...")
    lookup = build_gene_annotation_lookup(PROCESSED_GFF_DIR)
    print(f"Lookup contains {len(lookup)} genes")

    # Match genes to annotations
    def get_annotation(gene: str) -> str:
        gene_lower = gene.lower().strip()
        # Try exact match first
        if gene_lower in lookup:
            return lookup[gene_lower]
        # Try without suffix
        base = gene_lower.rstrip("_0123456789").rstrip("_")
        if base in lookup:
            return lookup[base]
        # Try first part before underscore
        if "_" in gene_lower:
            first_part = gene_lower.split("_")[0]
            if first_part in lookup:
                return lookup[first_part]
        return ""

    frequent["product"] = frequent["gene"].apply(get_annotation)

    # Add status column
    def get_status(row):
        nec_log = row["necessary_logistic"]
        nec_rf = row["necessary_rf"]
        if nec_log and nec_rf:
            return "★★ NECESSARY (both)"
        elif nec_log:
            return "★ Necessary (Log)"
        elif nec_rf:
            return "★ Necessary (RF)"
        else:
            return "Common"

    frequent["status"] = frequent.apply(get_status, axis=1)

    # Select and order columns
    result = frequent[[
        "gene", "status", "freq_logistic", "freq_rf",
        "imp_mean_logistic", "imp_mean_rf", "product"
    ]].copy()

    result.columns = ["Gene", "Status", "Freq_Log", "Freq_RF", "Imp_Log", "Imp_RF", "Product_Annotation"]

    return result


def main():
    parser = argparse.ArgumentParser(description="Annotate Rashomon frequent genes")
    parser.add_argument("--phenotype", type=str, required=True, help="Phenotype name")
    parser.add_argument("--freq_threshold", type=float, default=0.5, help="Frequency threshold")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    result = annotate_rashomon_genes(args.phenotype, args.freq_threshold)

    # Output path
    phen_safe = args.phenotype.replace(" ", "_")
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = RASHOMON_RESULTS_DIR / phen_safe / "rashomon_annotated_frequent.csv"

    result.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")

    # Also print table
    print(f"\n{'='*80}")
    print(f"FREQUENT GENES FOR: {args.phenotype}")
    print(f"{'='*80}\n")

    # Print in markdown-like format
    print(result.to_string(index=False, max_colwidth=60))

    # Summary stats
    n_nec_both = (result["Status"] == "★★ NECESSARY (both)").sum()
    n_nec_log = result["Status"].str.contains("Log").sum()
    n_nec_rf = result["Status"].str.contains("RF").sum()
    n_annotated = (result["Product_Annotation"] != "").sum()

    print(f"\n--- Summary ---")
    print(f"Total frequent genes: {len(result)}")
    print(f"Necessary in both: {n_nec_both}")
    print(f"Necessary in Logistic: {n_nec_log}")
    print(f"Necessary in RF: {n_nec_rf}")
    print(f"With annotation: {n_annotated}/{len(result)}")


if __name__ == "__main__":
    main()
