#!/usr/bin/env python3
"""Process GFF files, join with species metadata, derive canonical gene names,
and save compressed, Python-ready outputs with phenotype-aware ground truth masks.

Outputs one Parquet file per GFF (Zstandard compressed) and a manifest summarizing
processing results in the `processed_gff/` directory.

Default locations (hardcoded):
- GFF directory: /vol/projects/BIFO/genomenet/yichen/phenotype/data/gff
- Metadata Excel: /home/yhan/GenomeInterpretation/sporulation/microbe.cards table S1.xlsx
- Output directory: /vol/projects/BIFO/genomenet/yichen/phenotype/data/processed_gff

- We preserve all GFF fields (the canonical 9 columns) and all metadata columns
  from the Excel sheet. We also add convenience columns extracted from GFF
  attributes (e.g., gene, Name, product, inference, note, function, locus_tag),
  plus a semicolon-joined `canonical_gene_names` field.
- For each phenotype listed in `phenotype_utils.PHENOTYPE_COLUMNS` we attach the
  metadata value and a boolean `gt_<phenotype_slug>` column indicating whether
  the locus belongs to ground-truth clusters (>60% selection in both LASSO and RF)
  derived from clustered multiplicity analyses.
- To keep storage parsimonious while remaining Python-ready, outputs are saved
  as Parquet with Zstandard compression; categorical columns are dictionary-
  encoded when feasible. The large, raw `attributes` field is kept as a single
  string column (which compresses efficiently) instead of expanding to many
  sparse columns.
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Sequence

from urllib.parse import unquote

# Sporulation-related gene regex (consistent with sporulation/code/analyze_gff.py)
SPORULATION_REGEX = re.compile(
    r"(?i)\b(spo|ssp|cot|sigma|sig|ger|sleb|cwlj|dpa|spov|spoii|spo0)"
)

import pandas as pd

try:
    from phenotype_utils import (
        PHENOTYPE_COLUMNS,
        phenotype_to_slug,
        read_metadata_table,
        load_ground_truth_gene_sets,
        extract_canonical_gene_tokens,
        DATA_ROOT,
    )
except ImportError:  # pragma: no cover - fallback for package-style imports
    from .phenotype_utils import (  # type: ignore
        PHENOTYPE_COLUMNS,
        phenotype_to_slug,
        read_metadata_table,
        load_ground_truth_gene_sets,
        extract_canonical_gene_tokens,
        DATA_ROOT,
    )


# Default paths in this project
DEFAULT_GFF_DIR = DATA_ROOT / "gff"
DEFAULT_METADATA_XLSX = Path("sporulation/microbe.cards table S1.xlsx")
DEFAULT_OUTPUT_DIR = DATA_ROOT / "processed_gff"
def parse_gff_attributes(attr_str: str) -> Dict[str, str]:
    """Parse the 9th GFF column into a dict of attributes.

    - Attributes are semicolon-separated key=value pairs; values may be percent-encoded.
    - Duplicate keys are concatenated with commas.
    - Keys are returned in their original case (common keys include: ID, Name, gene,
      locus_tag, product, inference, note, function).
    """
    attrs: Dict[str, str] = {}
    if not attr_str or attr_str == ".":
        return attrs

    for item in attr_str.split(";"):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            key = unquote(key.strip())
            value = unquote(value.strip())
        else:
            key = unquote(item.strip())
            value = ""
        if key in attrs:
            attrs[key] = f"{attrs[key]},{value}"
        else:
            attrs[key] = value
    return attrs


def coerce_gff_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric columns are numeric; keep text as strings for readability."""
    if "start" in df.columns:
        df["start"] = pd.to_numeric(df["start"], errors="coerce").astype("Int32")
    if "end" in df.columns:
        df["end"] = pd.to_numeric(df["end"], errors="coerce").astype("Int32")
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"].replace({".": None}), errors="coerce").astype(
            "Float32"
        )
    # Keep strings as standard pandas string dtype for easier readability
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype("string")
    return df


def process_single_gff(
    gff_path: Path,
    metadata_df: pd.DataFrame,
    output_dir: Path,
    phenotype_gt: Dict[str, Set[str]],
    phenotype_columns: Sequence[str],
) -> Tuple[Path, int, bool]:
    """Process one GFF file and write a compressed Parquet.

    Aggregates Prokka/Prodigal entries by locus (locus_tag/Parent/ID) so one row reflects
    one genomic locus. Keeps a curated set of informative columns and selected metadata.

    Returns (output_parquet_path, num_rows, metadata_matched).
    """
    if not gff_path.is_file():
        raise FileNotFoundError(gff_path)

    fasta_filename = gff_path.name.replace(".gff", ".fasta")
    fasta_norm = os.path.basename(fasta_filename).strip().lower()

    # Find metadata row by normalized 'Fasta file'
    md_match = metadata_df[metadata_df["Fasta file_norm"] == fasta_norm]
    has_metadata = len(md_match) > 0
    md_row = md_match.iloc[0] if has_metadata else None

    # Only keep selected metadata columns
    base_md_cols = [
        "Binomial name",
        "Phylum",
        "Class",
        "Order",
        "Family",
        "Genus",
        "Species",
        "Fasta file",
    ]
    selected_md_cols = [col for col in base_md_cols if col in metadata_df.columns]
    for col in phenotype_columns:
        if col in metadata_df.columns and col not in selected_md_cols:
            selected_md_cols.append(col)

    # Aggregation by locus
    aggregated: Dict[str, Dict[str, object]] = {}

    def choose_text(existing: Optional[str], new_val: Optional[str]) -> Optional[str]:
        if new_val and pd.notna(new_val) and str(new_val).strip() != "":
            if not existing or pd.isna(existing) or str(existing).strip() == "":
                return str(new_val)
            if str(new_val) not in str(existing).split(" | "):
                return f"{existing} | {new_val}"
        return existing

    def extract_names(attrs: Dict[str, str]) -> Set[str]:
        names: Set[str] = set()
        for key in ("gene", "Name", "gene_synonym", "locus_tag"):
            if key in attrs:
                names.update(extract_canonical_gene_tokens(attrs[key]))
        return names

    # Collect contig lengths from meta-lines
    contig_lengths: Dict[str, int] = {}

    with gff_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line:
                continue
            if line.startswith("#"):
                # Capture contig length metadata from common directives
                if line.startswith("##sequence-region"):
                    # Format: ##sequence-region <seqid> <start> <end>
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        seqid = parts[1]
                        try:
                            end = int(parts[3])
                            contig_lengths[seqid] = end
                        except Exception:
                            pass
                elif line.startswith("##contig="):
                    # Format (Prokka): ##contig=<ID=...,length=...>
                    m = re.search(r"ID=([^,>]+)", line)
                    mlen = re.search(r"length=([0-9]+)", line)
                    if m and mlen:
                        try:
                            seqid = m.group(1)
                            clen = int(mlen.group(1))
                            contig_lengths[seqid] = clen
                        except Exception:
                            pass
                elif line.startswith("##FASTA"):
                    break
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue
            seqid, source, ftype, start, end, score, strand, phase, attr_str = parts
            attrs = parse_gff_attributes(attr_str)
            names_from_attrs = extract_names(attrs)

            # Determine aggregation key
            gene_key = (
                attrs.get("locus_tag")
                or attrs.get("Parent")
                or (attrs.get("ID") if ftype.lower() == "gene" else None)
            )
            if not gene_key:
                # Last resort fallback on coordinates to avoid data loss
                gene_key = f"{seqid}:{start}:{end}:{strand}"

            # Initialize aggregate record
            if gene_key not in aggregated:
                aggregated[gene_key] = {
                    "seqid": seqid,
                    "start": pd.to_numeric(start, errors="coerce"),
                    "end": pd.to_numeric(end, errors="coerce"),
                    "strand": strand,
                    # annotations
                    "locus_tag": attrs.get("locus_tag"),
                    "Name": attrs.get("Name"),
                    "gene": attrs.get("gene"),
                    "product": attrs.get("product"),
                    "inference": attrs.get("inference"),
                    "protein_id": attrs.get("protein_id"),
                    "canonical_names": set(names_from_attrs),
                    "sources_raw": [source],
                }
            else:
                agg = aggregated[gene_key]
                # Expand coordinates to cover all related features
                s = pd.to_numeric(start, errors="coerce")
                e = pd.to_numeric(end, errors="coerce")
                if pd.notna(s):
                    if pd.isna(agg["start"]):
                        agg["start"] = s
                    else:
                        agg["start"] = min(agg["start"], s)  # type: ignore[arg-type]
                if pd.notna(e):
                    if pd.isna(agg["end"]):
                        agg["end"] = e
                    else:
                        agg["end"] = max(agg["end"], e)  # type: ignore[arg-type]
                # Strand/seqid: keep original unless missing
                if not agg.get("seqid") and seqid:
                    agg["seqid"] = seqid
                if (not agg.get("strand")) and strand:
                    agg["strand"] = strand
                # Merge sources
                agg.setdefault("sources_raw", []).append(source)
                # Merge annotations (prefer CDS product if available)
                agg["locus_tag"] = agg["locus_tag"] or attrs.get("locus_tag")
                agg["Name"] = choose_text(agg.get("Name"), attrs.get("Name"))
                agg["gene"] = choose_text(agg.get("gene"), attrs.get("gene"))
                agg["product"] = choose_text(agg.get("product"), attrs.get("product"))
                agg["inference"] = choose_text(agg.get("inference"), attrs.get("inference"))
                agg["protein_id"] = agg.get("protein_id") or attrs.get("protein_id")
                agg.setdefault("canonical_names", set()).update(names_from_attrs)

    # Convert to DataFrame
    rows: List[Dict[str, object]] = []
    for _, agg in aggregated.items():
        canon_names: Set[str] = set()
        for field in ("gene", "Name", "locus_tag"):
            canon_names.update(extract_canonical_gene_tokens(agg.get(field)))
        canon_names.update(set(agg.get("canonical_names", set())))
        canon_names = {name for name in canon_names if name}
        row: Dict[str, object] = {
            "seqid": agg.get("seqid"),
            "start": agg.get("start"),
            "end": agg.get("end"),
            "strand": agg.get("strand"),
            "locus_tag": agg.get("locus_tag"),
            "gene": agg.get("gene"),
            "Name": agg.get("Name"),
            "product": agg.get("product"),
            "inference": agg.get("inference"),
            "protein_id": agg.get("protein_id"),
            "sources": ";".join(sorted({str(s) for s in agg.get("sources_raw", []) if s})),
            "canonical_gene_names": ";".join(sorted(canon_names)) if canon_names else "",
            "gff_filename": gff_path.name,
            "fasta_filename": fasta_filename,
            "row_type": "locus",
            "contig_len": int(contig_lengths.get(str(agg.get("seqid")), 0)) if contig_lengths else pd.NA,
        }

        # Phenotype metadata and ground-truth masks
        for phenotype in phenotype_columns:
            slug = phenotype_to_slug(phenotype)
            gt_set = phenotype_gt.get(phenotype, set())
            # Use regex-based detection for spore_formation (consistent with sporulation/ analysis)
            if slug == "spore_formation":
                row[f"gt_{slug}"] = any(
                    SPORULATION_REGEX.search(name) for name in canon_names if name
                )
            else:
                row[f"gt_{slug}"] = any(name in gt_set for name in canon_names)
            if has_metadata and phenotype in metadata_df.columns:
                row[phenotype] = md_row[phenotype]
            else:
                row[phenotype] = pd.NA

        # Attach selected metadata as constants per file
        if has_metadata:
            for col in selected_md_cols:
                if col in metadata_df.columns:
                    row[col] = md_row[col]
                else:
                    row[col] = pd.NA
        rows.append(row)

    # Add contig meta rows to ensure we persist true contig lengths (even if no features)
    for seqid, clen in contig_lengths.items():
        row = {
            "seqid": seqid,
            "start": pd.NA,
            "end": pd.NA,
            "strand": pd.NA,
            "locus_tag": pd.NA,
            "gene": pd.NA,
            "Name": pd.NA,
            "product": pd.NA,
            "inference": pd.NA,
            "protein_id": pd.NA,
            "sources": pd.NA,
            "canonical_gene_names": "",
            "gff_filename": gff_path.name,
            "fasta_filename": fasta_filename,
            "row_type": "contig",
            "contig_len": int(clen),
        }
        for phenotype in phenotype_columns:
            slug = phenotype_to_slug(phenotype)
            row[f"gt_{slug}"] = False
            row[phenotype] = md_row[phenotype] if has_metadata and phenotype in metadata_df.columns else pd.NA
        for col in selected_md_cols:
            if has_metadata and col in metadata_df.columns:
                row[col] = md_row[col]
            else:
                row[col] = pd.NA
        rows.append(row)

    df = pd.DataFrame.from_records(rows)

    # Ensure presence of columns even if empty
    if df.empty:
        columns: List[str] = [
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
            "row_type",
            "contig_len",
        ]
        for phenotype in phenotype_columns:
            gt_col = f"gt_{phenotype_to_slug(phenotype)}"
            if gt_col not in columns:
                columns.append(gt_col)
        for col in selected_md_cols:
            if col not in columns:
                columns.append(col)
        for phenotype in phenotype_columns:
            if phenotype not in columns:
                columns.append(phenotype)
        df = pd.DataFrame(columns=columns)

    # Final type tweaks for readability
    df = coerce_gff_types(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (gff_path.stem + ".parquet")
    try:
        df.to_parquet(out_path, engine="pyarrow", compression="zstd", compression_level=7, index=False)
    except Exception:
        df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)

    return out_path, int(len(df)), bool(has_metadata)


def process_all(
    gff_dir: Path,
    metadata_xlsx: Path,
    output_dir: Path,
) -> Path:
    """Process all .gff files in `gff_dir` and write outputs to `output_dir`.

    Returns the path to the manifest Parquet file.
    """
    if not gff_dir.exists():
        raise FileNotFoundError(f"GFF directory not found: {gff_dir}")
    if not metadata_xlsx.exists():
        raise FileNotFoundError(f"Metadata Excel not found: {metadata_xlsx}")

    metadata_df = read_metadata_table(metadata_xlsx)
    phenotype_columns: List[str] = list(PHENOTYPE_COLUMNS)
    phenotype_gt = load_ground_truth_gene_sets(phenotype_columns)

    manifest_rows: List[Dict[str, object]] = []

    gff_files = sorted([p for p in gff_dir.iterdir() if p.is_file() and p.suffix.lower() == ".gff"])
    total = len(gff_files)

    for idx, gff_path in enumerate(gff_files, start=1):
        out_path, n_rows, has_md = process_single_gff(
            gff_path=gff_path,
            metadata_df=metadata_df,
            output_dir=output_dir,
            phenotype_gt=phenotype_gt,
            phenotype_columns=phenotype_columns,
        )

        row: Dict[str, object] = {
            "gff_filename": gff_path.name,
            "parquet_path": str(out_path),
            "num_rows": int(n_rows),
            "metadata_matched": bool(has_md),
        }

        # Include a couple of common metadata identifiers if present
        for candidate in [
            "Fasta file",
            "Species",
            "Organism",
            "NCBI Taxon ID",
            "NCBI Taxonomy ID",
            "TaxID",
        ]:
            if candidate in metadata_df.columns:
                # Use the value from the matched row when available, else NA
                md_match = metadata_df[metadata_df["Fasta file_norm"] == gff_path.name.replace(".gff", ".fasta").lower()]
                if len(md_match) > 0:
                    row[candidate] = md_match.iloc[0][candidate]
                else:
                    row[candidate] = pd.NA

        manifest_rows.append(row)

        # Progress logging every 100 files (and at the end)
        if (idx % 100 == 0) or (idx == total):
            print(f"{idx}/{total} done")

    manifest_df = pd.DataFrame.from_records(manifest_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.parquet"
    # Save compressed manifest as well
    try:
        manifest_df.to_parquet(manifest_path, engine="pyarrow", compression="zstd", compression_level=7, index=False)
    except Exception:
        manifest_df.to_parquet(manifest_path, engine="pyarrow", compression="snappy", index=False)

    # Also a tiny CSV for quick peeks
    try:
        manifest_df.to_csv(output_dir / "manifest.csv", index=False)
    except Exception:
        pass

    return manifest_path


 


def main() -> None:
    manifest_path = process_all(DEFAULT_GFF_DIR, DEFAULT_METADATA_XLSX, DEFAULT_OUTPUT_DIR)
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
