#!/usr/bin/env python3
"""Quick preview script for processed GFF Parquet outputs.

Reads one Parquet file from the hardcoded processed directory and prints:
- chosen file path
- all available columns (from Parquet schema when possible)
- shape (rows, columns) for the loaded preview subset
- dtypes of the loaded columns
- first 10 rows of key columns
 - basic counts for the chosen phenotype ground-truth column if present

Selection rule:
- Only consider a file whose metadata column for the chosen phenotype is truthy in
  at least one row (fallback: any row with ground-truth mask True).

Hardcoded directory to match the processing script defaults:
- /vol/projects/BIFO/genomenet/yichen/phenotype/data/processed_gff
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    from phenotype_utils import DATA_ROOT
except ImportError:  # pragma: no cover
    from .phenotype_utils import DATA_ROOT  # type: ignore


PROCESSED_DIR = DATA_ROOT / "processed_gff"

try:
    from phenotype_utils import phenotype_to_slug
except ImportError:  # pragma: no cover - package-style import fallback
    from .phenotype_utils import phenotype_to_slug  # type: ignore


def is_truthy(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return value != 0
    s = str(value).strip().lower()
    return s in {"true", "yes", "y", "1", "t"}


def find_case_insensitive_column(columns: List[str], target: str) -> Optional[str]:
    target_norm = target.strip().lower()
    for c in columns:
        if c.strip().lower() == target_norm:
            return c
    return None


def pick_parquet_file(directory: Path, metadata_col: str, mask_col: str) -> Optional[Path]:
    if not directory.exists():
        return None
    candidates: List[Path] = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".parquet"
    )
    # Filter out manifest first
    data_files = [p for p in candidates if p.name != "manifest.parquet"]

    # Probe each candidate: prefer metadata column; fallback to mask column
    for pth in data_files:
        cols = get_schema_columns(pth)
        if cols is None:
            cols = []
        meta_col = find_case_insensitive_column(cols, metadata_col) if cols else None
        mask_col_found = find_case_insensitive_column(cols, mask_col) if cols else None
        try:
            if meta_col:
                series = pd.read_parquet(pth, engine="pyarrow", columns=[meta_col])
                if series.iloc[:, 0].map(is_truthy).any():
                    return pth
            if mask_col_found:
                series = pd.read_parquet(pth, engine="pyarrow", columns=[mask_col_found])
                col = series.iloc[:, 0]
                if pd.api.types.is_bool_dtype(col) or pd.api.types.is_bool_dtype(col.dropna()):
                    if col.fillna(False).any():
                        return pth
                else:
                    if col.map(is_truthy).any():
                        return pth
        except Exception:
            # Could not read required column; skip this file
            continue

    # No suitable file found
    return None


def get_schema_columns(parquet_path: Path) -> Optional[List[str]]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        pf = pq.ParquetFile(parquet_path)
        return [f.name for f in pf.schema_arrow]
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview processed GFF parquet outputs")
    ap.add_argument("--directory", type=str, default=str(PROCESSED_DIR), help="Directory containing processed GFF parquet files")
    ap.add_argument("--phenotype", type=str, default="Spore formation", help="Phenotype column to inspect (metadata column name)")
    args = ap.parse_args()

    directory = Path(args.directory)
    phenotype = args.phenotype
    mask_column = f"gt_{phenotype_to_slug(phenotype)}"

    parquet_path = pick_parquet_file(directory, phenotype, mask_column)
    if parquet_path is None:
        print(f"No suitable Parquet files found in: {directory}")
        return

    print(f"Using Parquet file: {parquet_path}")
    print(f"Phenotype: {phenotype} | mask column: {mask_column}")

    schema_cols = get_schema_columns(parquet_path)
    if schema_cols is not None:
        print(f"Available columns (from schema): {len(schema_cols)}")
        print("  ", ", ".join(schema_cols))
    else:
        print("Could not fetch schema via pyarrow; will infer after reading.")

    # Read all columns for full visibility
    df = pd.read_parquet(parquet_path, engine="pyarrow")

    print(f"Data shape (rows, cols): {df.shape}")
    print("Dtypes:")
    # Convert to string for concise printing
    dtypes_str = ", ".join([f"{c}:{dt}" for c, dt in df.dtypes.items()])
    print("  ", dtypes_str)

    # Show head
    try:
        print("\nHead (first 1 rows):")
        print(df.head(1).to_string(index=False))
    except Exception:
        print(df.head(1))

    # Ground-truth mask summary
    mask_col_present = find_case_insensitive_column(list(df.columns), mask_column)
    if mask_col_present and mask_col_present in df.columns:
        mask_series = df[mask_col_present]
        if not pd.api.types.is_bool_dtype(mask_series):
            mask_series = mask_series.map(is_truthy)
        mask_counts = mask_series.fillna(False).value_counts(dropna=False)
        print(f"\n{mask_col_present} counts:")
        for k, v in mask_counts.items():
            print(f"  {k}: {v}")
    else:
        print(f"\nMask column '{mask_column}' not present in this Parquet file.")

    # Metadata phenotype column summary, if present
    md_col = None
    if schema_cols is not None:
        md_col = find_case_insensitive_column(schema_cols, phenotype)
    if not md_col:
        md_col = find_case_insensitive_column(list(df.columns), phenotype)
    if md_col and md_col in df.columns:
        counts = df[md_col].map(is_truthy).value_counts(dropna=False)
        print(f"\nMetadata '{md_col}' (truthy) counts:")
        for k, v in counts.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
