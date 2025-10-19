#!/usr/bin/env python3
"""Quick preview script for processed GFF Parquet outputs.

Reads one Parquet file from the hardcoded processed directory and prints:
- chosen file path
- all available columns (from Parquet schema when possible)
- shape (rows, columns) for the loaded preview subset
- dtypes of the loaded columns
- first 10 rows of key columns
- basic counts for `spore_related` if present

Selection rule:
- Only consider a file whose metadata column 'Spore formation' (case-insensitive)
  is truthy in at least one row.

Hardcoded directory to match the processing script defaults:
- /vol/projects/BIFO/genomenet/yichen/phenotype/data/processed_gff
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


PROCESSED_DIR = Path("/vol/projects/BIFO/genomenet/yichen/phenotype/data/processed_gff")


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


def pick_parquet_file(directory: Path) -> Optional[Path]:
    if not directory.exists():
        return None
    candidates: List[Path] = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".parquet"
    )
    # Filter out manifest first
    data_files = [p for p in candidates if p.name != "manifest.parquet"]

    # Probe each candidate: must have a 'Spore formation' column with any truthy value
    for pth in data_files:
        cols = get_schema_columns(pth)
        if cols is None:
            cols = []
        spore_col = find_case_insensitive_column(cols, "Spore formation") if cols else None
        try:
            if spore_col is None:
                series = pd.read_parquet(pth, engine="pyarrow", columns=["Spore formation"])  # type: ignore[arg-type]
                spore_col = "Spore formation"
            else:
                series = pd.read_parquet(pth, engine="pyarrow", columns=[spore_col])
        except Exception:
            # Could not read required column; skip this file
            continue

        any_truthy = series.iloc[:, 0].map(is_truthy).any()
        if bool(any_truthy):
            return pth

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
    parquet_path = pick_parquet_file(PROCESSED_DIR)
    if parquet_path is None:
        print(f"No Parquet files found in: {PROCESSED_DIR}")
        return

    print(f"Using Parquet file: {parquet_path}")

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

    # Basic sporulation summary
    if "spore_related" in df.columns:
        counts = df["spore_related"].value_counts(dropna=False)
        print("\nspore_related counts:")
        for k, v in counts.items():
            print(f"  {k}: {v}")

    # Also summarize metadata 'Spore formation' if present
    spore_md_col = None
    if schema_cols is not None:
        spore_md_col = find_case_insensitive_column(schema_cols, "Spore formation")
    if not spore_md_col:
        # Fall back by scanning df columns
        spore_md_col = find_case_insensitive_column(list(df.columns), "Spore formation")
    if spore_md_col and spore_md_col in df.columns:
        counts = df[spore_md_col].map(is_truthy).value_counts(dropna=False)
        print("\nMetadata 'Spore formation' (truthy) counts:")
        for k, v in counts.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

