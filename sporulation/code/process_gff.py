#!/usr/bin/env python3
"""Process GFF files, join with species metadata, detect sporulation annotations,
and save compressed, Python-ready outputs.

Outputs one Parquet file per GFF (Zstandard compressed) and a manifest summarizing
processing results in the `processed_gff/` directory.

Default locations (hardcoded):
- GFF directory: /vol/projects/BIFO/genomenet/yichen/phenotype/data/gff
- Metadata Excel: /home/yhan/GenomeInterpretation/sporulation/microbe.cards table S1.xlsx
- Output directory: /vol/projects/BIFO/genomenet/yichen/phenotype/data/processed_gff

Notes:
- We preserve all GFF fields (the canonical 9 columns) and all metadata columns
  from the Excel sheet. We also add convenience columns extracted from GFF
  attributes (e.g., gene, Name, product, inference, note, function, locus_tag),
  and a boolean `spore_related` column based on curated heuristics.
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
from typing import Dict, Iterable, List, Tuple, Optional

from urllib.parse import unquote

import pandas as pd


# Default paths in this project
DATA_ROOT = Path("/vol/projects/BIFO/genomenet/yichen/phenotype/data")
DEFAULT_GFF_DIR = DATA_ROOT / "gff"
DEFAULT_METADATA_XLSX = Path("sporulation/microbe.cards table S1.xlsx")
DEFAULT_OUTPUT_DIR = DATA_ROOT / "processed_gff"


def compile_sporulation_pattern() -> re.Pattern[str]:
    """Return a compiled regex pattern capturing sporulation-related terms.

    Heuristics include:
    - Generic terms: spore, sporulation, forespore, endospore, exosporium, cortex, sporangium
    - Processes: germination, dipicolinic (acid)
    - Classical gene/module names: spo0A/B/F, spoI/II/III/IV/V modules, sspX genes (SASP),
      cotX spore coat genes, sigma factors specific to sporulation (SigF/E/G/K)

    The pattern is case-insensitive and targets annotation-rich fields (product, note,
    inference, gene, Name, function).
    """
    # Case-insensitive pattern; keep fairly specific to avoid common false positives.
    # Note: we avoid a plain "coat" match (too broad) and instead require "spore coat".
    pattern = re.compile(
        r"(?i)("
        r"sporu\w+|"  # sporulation*, sporulate*, sporu*
        r"spore\w*|endospore\w*|exospori\w*|forespore\w*|sporang\w*|"  # spore*, endospore*, exosporium*, etc.
        r"spore[-_\s]coat|"  # spore coat
        r"cortex\w*|dipicolin\w*|"  # cortex, dipicolinic acid
        r"germin\w*|"  # germination/germinant/germinate
        r"\bspo0[a-z]\b|"  # spo0a/0b/0f etc.
        r"\bspo(?:i|ii|iii|iv|v)[a-z]*\b|"  # spoI/II/III/IV/V modules and subgenes
        r"\bssp[a-z]\b|"  # small acid-soluble spore proteins (SASP)
        r"\bcot[a-z0-9]{1,3}\b|"  # spore coat proteins
        r"\bsig[feGK]\b|\bsigma[-_\s]?[feGK]\b"  # sporulation sigma factors
        r")",
    )
    return pattern


def read_metadata_table(xlsx_path: Path) -> pd.DataFrame:
    """Load the species metadata Excel table, ensuring we have a 'Fasta file' column.

    Returns a DataFrame with all original columns, plus a normalized helper column:
    - 'Fasta file_norm': lowercased, stripped version of the basename for robust matching.
    """
    try:
        metadata_df = pd.read_excel(xlsx_path)  # engine=auto (requires openpyxl for .xlsx)
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Reading .xlsx requires openpyxl. Please install it: pip install openpyxl"
        ) from exc

    if "Fasta file" not in metadata_df.columns:
        raise ValueError("Expected column 'Fasta file' in metadata Excel sheet")

    # Normalize to basenames and lowercase for robust matching
    def _norm_basename(val: object) -> str:
        s = str(val) if not pd.isna(val) else ""
        s = os.path.basename(s)
        return s.strip().lower()

    metadata_df["Fasta file_norm"] = metadata_df["Fasta file"].map(_norm_basename)
    return metadata_df


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


def build_annotation_text(row_attrs: Dict[str, str], fallback_fields: Iterable[str]) -> str:
    """Concatenate relevant annotation-bearing fields into a single text blob for matching.

    `fallback_fields` typically includes columns like 'type' or 'source' from the core GFF
    fields to increase the chance of recognizing sporulation-relevant records.
    """
    parts: List[str] = []
    for k in ("product", "note", "function", "inference", "Name", "gene", "locus_tag"):
        v = row_attrs.get(k)
        if v:
            parts.append(str(v))
    for v in fallback_fields:
        if v:
            parts.append(str(v))
    return " ".join(parts)


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
    spore_pattern: re.Pattern[str],
    output_dir: Path,
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
    selected_md_cols = [
        "Phylum",
        "Class",
        "Order",
        "Family",
        "Genus",
        "Species",
        "Spore formation",
    ]

    # Aggregation by locus
    aggregated: Dict[str, Dict[str, object]] = {}

    def choose_text(existing: Optional[str], new_val: Optional[str]) -> Optional[str]:
        if new_val and pd.notna(new_val) and str(new_val).strip() != "":
            if not existing or pd.isna(existing) or str(existing).strip() == "":
                return str(new_val)
            if str(new_val) not in str(existing).split(" | "):
                return f"{existing} | {new_val}"
        return existing

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
                    "sources": set([source]),
                    # annotations
                    "locus_tag": attrs.get("locus_tag"),
                    "Name": attrs.get("Name"),
                    "gene": attrs.get("gene"),
                    "product": attrs.get("product"),
                    "inference": attrs.get("inference"),
                    "protein_id": attrs.get("protein_id"),
                    "spore_related": False,
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
                agg["sources"].add(source)  # type: ignore[index]
                # Merge annotations (prefer CDS product if available)
                agg["locus_tag"] = agg["locus_tag"] or attrs.get("locus_tag")
                agg["Name"] = choose_text(agg.get("Name"), attrs.get("Name"))
                agg["gene"] = choose_text(agg.get("gene"), attrs.get("gene"))
                agg["product"] = choose_text(agg.get("product"), attrs.get("product"))
                agg["inference"] = choose_text(agg.get("inference"), attrs.get("inference"))
                agg["protein_id"] = agg.get("protein_id") or attrs.get("protein_id")

            # Sporulation flag (OR across related features)
            anno_text = build_annotation_text(attrs, fallback_fields=[ftype, source])
            if spore_pattern.search(anno_text):
                aggregated[gene_key]["spore_related"] = True

    # Convert to DataFrame
    rows: List[Dict[str, object]] = []
    for _, agg in aggregated.items():
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
            "sources": ";".join(sorted(list(agg.get("sources", set())))),
            "spore_related": bool(agg.get("spore_related", False)),
            "gff_filename": gff_path.name,
            "fasta_filename": fasta_filename,
            "row_type": "locus",
            "contig_len": int(contig_lengths.get(str(agg.get("seqid")), 0)) if contig_lengths else pd.NA,
        }

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
        rows.append({
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
            "spore_related": False,
            "gff_filename": gff_path.name,
            "fasta_filename": fasta_filename,
            "row_type": "contig",
            "contig_len": int(clen),
        })

    df = pd.DataFrame.from_records(rows)

    # Ensure presence of columns even if empty
    if df.empty:
        df = pd.DataFrame(
            columns=[
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
                "spore_related",
                "gff_filename",
                "fasta_filename",
                "Phylum",
                "Class",
                "Order",
                "Family",
                "Genus",
                "Species",
                "Fasta file",
                "Spore formation",
                "row_type",
                "contig_len",
            ]
        )

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
    spore_pat = compile_sporulation_pattern()

    manifest_rows: List[Dict[str, object]] = []

    gff_files = sorted([p for p in gff_dir.iterdir() if p.is_file() and p.suffix.lower() == ".gff"])
    total = len(gff_files)

    for idx, gff_path in enumerate(gff_files, start=1):
        out_path, n_rows, has_md = process_single_gff(
            gff_path=gff_path,
            metadata_df=metadata_df,
            spore_pattern=spore_pat,
            output_dir=output_dir,
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

