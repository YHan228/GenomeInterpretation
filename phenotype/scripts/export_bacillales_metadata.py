#!/usr/bin/env python3
"""
Export Bacillales species info into a CSV and prune a Newick tree to include only
the available Bacillales genomes (tree tip names expected to match FASTA basenames
without extensions) observed in the FASTA data directories.

Outputs (CSV columns):
- fasta_name
- phylum
- class
- order
- family
- genus
- species
- sporulation          (from "Spore formation" in the Excel)
- GC%                  (percent over A/C/G/T only)
- Genome Length        (bp; counts A/C/G/T/N)

Defaults:
- Excel: /home/yhan/GenomeInterpretation/sporulation/microbe.cards table S1.xlsx
- Data dirs (scanned recursively):
  - /vol/projects/BIFO/genomenet/yichen/phenotype/bacillales/train
  - /vol/projects/BIFO/genomenet/yichen/phenotype/bacillales/validation
  - /vol/projects/BIFO/genomenet/yichen/phenotype/bacillales/test
- Output CSV: /home/yhan/GenomeInterpretation/phenotype/bacillales.csv
- Input tree: /home/yhan/GenomeInterpretation/phenotype/fulltree.treefile
- Output tree: /home/yhan/GenomeInterpretation/phenotype/bacillales_pruned.treefile
"""

import argparse
import gzip
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


DEFAULT_EXCEL_PATH = "/home/yhan/GenomeInterpretation/sporulation/microbe.cards table S1.xlsx"
DEFAULT_DATA_DIRS = [
	"/vol/projects/BIFO/genomenet/yichen/phenotype/bacillales/train",
	"/vol/projects/BIFO/genomenet/yichen/phenotype/bacillales/validation",
	"/vol/projects/BIFO/genomenet/yichen/phenotype/bacillales/test",
]
DEFAULT_OUTPUT_CSV = "/home/yhan/GenomeInterpretation/phenotype/bacillales.csv"
DEFAULT_TREE_IN = "/home/yhan/GenomeInterpretation/phenotype/fulltree.treefile"
DEFAULT_TREE_OUT = "/home/yhan/GenomeInterpretation/phenotype/bacillales_pruned.treefile"

# Nucleotide FASTA extensions (case-insensitive); supports optional ".gz"
NUC_FASTA_EXTENSIONS = {".fa", ".fasta", ".fna", ".fas", ".fsa"}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Export Bacillales metadata CSV and prune Newick tree.")
	parser.add_argument("--excel", type=str, default=DEFAULT_EXCEL_PATH, help="Path to taxonomy Excel file.")
	parser.add_argument(
		"--data_dirs",
		type=str,
		default=",".join(DEFAULT_DATA_DIRS),
		help="Comma-separated list of directories to scan for FASTA files.",
	)
	parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV, help="Path to write bacillales.csv")
	parser.add_argument("--tree_in", type=str, default=DEFAULT_TREE_IN, help="Path to input Newick treefile")
	parser.add_argument("--tree_out", type=str, default=DEFAULT_TREE_OUT, help="Path to write pruned Newick tree")
	parser.add_argument("--skip_tree", action="store_true", help="Skip tree pruning even if tree file exists")
	return parser.parse_args()


def find_column_name(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
	"""
	Find a column in 'columns' matching one of the candidate names using:
	- exact match
	- case-insensitive exact match
	- case-insensitive substring match
	"""
	columns_list = list(columns)
	lower_map = {c.lower(): c for c in columns_list}

	for cand in candidates:
		if cand in columns_list:
			return cand
	for cand in candidates:
		if cand.lower() in lower_map:
			return lower_map[cand.lower()]
	for cand in candidates:
		cand_low = cand.lower()
		for c in columns_list:
			if cand_low in c.lower():
				return c
	return None


def normalize_fasta_basename(filename: str) -> str:
	"""Return basename, stripping a trailing '.gz' if present."""
	basename = os.path.basename(filename)
	if basename.endswith(".gz"):
		basename = basename[:-3]
	return basename


def strip_fasta_extension(name: str) -> str:
	"""
	Strip a recognized FASTA extension from the end of 'name' (case-insensitive).
	Examples: 'sample.fna' -> 'sample', 'sample.fasta' -> 'sample'.
	"""
	lower = name.lower()
	for ext in sorted(NUC_FASTA_EXTENSIONS, key=len, reverse=True):
		if lower.endswith(ext):
			return name[: -len(ext)]
	return name


def is_nucleotide_fasta(path: str) -> bool:
	lower = path.lower()
	for ext in NUC_FASTA_EXTENSIONS:
		if lower.endswith(ext) or lower.endswith(ext + ".gz"):
			return True
	return False


def scan_fasta_files(dirs: Iterable[str]) -> Dict[str, str]:
	"""
	Scan directories for nucleotide FASTA files and return
	mapping: normalized basename -> absolute path (first occurrence wins).
	"""
	result: Dict[str, str] = {}
	for root in dirs:
		if not root:
			continue
		if not os.path.isdir(root):
			continue
		for dirpath, _, filenames in os.walk(root):
			for name in filenames:
				full = os.path.join(dirpath, name)
				if not is_nucleotide_fasta(full):
					continue
				key = normalize_fasta_basename(full)
				if key not in result:
					result[key] = full
	return result


def compute_gc_and_length(fasta_path: str) -> Tuple[int, float]:
	"""
	Stream a FASTA and compute genome length and GC percentage.
	- Genome Length counts A/C/G/T/N (uppercase)
	- GC% is 100 * (G+C) / (A+C+G+T)
	"""
	length = 0
	known_total = 0
	gc_count = 0

	open_fn = gzip.open if fasta_path.lower().endswith(".gz") else open
	with open_fn(fasta_path, "rt") as handle:
		for line in handle:
			if not line or line.startswith(">"):
				continue
			seq = line.strip().upper()
			for ch in seq:
				if ch in ("A", "C", "G", "T", "N"):
					length += 1
				if ch in ("A", "C", "G", "T"):
					known_total += 1
					if ch in ("G", "C"):
						gc_count += 1
	gc_percent = (100.0 * gc_count / known_total) if known_total > 0 else 0.0
	return length, gc_percent


def load_bacillales_metadata(excel_path: str) -> pd.DataFrame:
	"""
	Load the Excel metadata and return only rows for Order == 'Bacillales',
	with a normalized 'normalized_fasta' column for matching.
	"""
	if not os.path.isfile(excel_path):
		raise FileNotFoundError(f"Excel file not found: {excel_path}")
	df = pd.read_excel(excel_path)
	if df.empty:
		return pd.DataFrame()

	fasta_col = find_column_name(df.columns, ["Fasta file", "FASTA file", "Fasta", "fasta file"])
	order_col = find_column_name(df.columns, ["Order", "order"])
	if fasta_col is None or order_col is None:
		raise KeyError(f"Required columns not found in Excel: have={list(df.columns)} need~=['Fasta file','Order']")

	df = df.copy()
	df["normalized_fasta"] = df[fasta_col].astype(str).map(normalize_fasta_basename)
	df["_order_norm"] = df[order_col].astype(str).str.strip().str.lower()
	df = df[df["_order_norm"] == "bacillales"].copy()
	return df, fasta_col


def prune_tree_to_fasta_labels(tree_in: str, tree_out: str, keep_labels: Set[str]) -> None:
	"""
	Prune the Newick tree to only include terminals whose normalized FASTA basename
	(without .fa/.fna/.fasta etc.) is in 'keep_labels'.
	Uses Biopython if available; otherwise, prints a warning and skips pruning.
	"""
	if not os.path.isfile(tree_in):
		print(f"[Warn] Tree file not found, skipping prune: {tree_in}", file=sys.stderr)
		return
	try:
		from Bio import Phylo  # type: ignore
	except Exception as exc:
		print(f"[Warn] Biopython not available ({exc}); skipping tree pruning.", file=sys.stderr)
		return

	tree = Phylo.read(tree_in, "newick")
	terminals = list(tree.get_terminals())
	for leaf in list(terminals):
		label = leaf.name if leaf.name is not None else ""
		# Normalize tree leaf label the same way as FASTA basenames:
		normalized = strip_fasta_extension(normalize_fasta_basename(label))
		if normalized not in keep_labels:
			try:
				tree.prune(leaf)
			except Exception:
				# Continue pruning even if some prunes fail
				pass
	# Write pruned tree
	out_dir = os.path.dirname(tree_out)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)
	Phylo.write(tree, tree_out, "newick")
	print(f"Pruned tree saved to: {tree_out}")


def main() -> None:
	args = parse_args()
	data_dirs = [d.strip() for d in args.data_dirs.split(",") if d.strip()]

	# Load metadata (Bacillales only)
	try:
		meta_df, fasta_col = load_bacillales_metadata(args.excel)
	except Exception as exc:
		print(f"[Error] Failed to load/filter Excel metadata: {exc}", file=sys.stderr)
		sys.exit(1)
	if meta_df.empty:
		print("[Info] No Bacillales rows found in Excel; nothing to export.", file=sys.stderr)
		sys.exit(0)

	# Identify taxonomy/sporulation columns
	phylum_col = find_column_name(meta_df.columns, ["Phylum", "phylum"])
	class_col = find_column_name(meta_df.columns, ["Class", "class"])
	order_col = find_column_name(meta_df.columns, ["Order", "order"])
	family_col = find_column_name(meta_df.columns, ["Family", "family"])
	genus_col = find_column_name(meta_df.columns, ["Genus", "genus"])
	species_col = find_column_name(meta_df.columns, ["Species", "species"])
	spore_col = find_column_name(meta_df.columns, ["Spore formation", "Spore Formation", "sporulation"])
	for needed, name in {
		"Phylum": phylum_col,
		"Class": class_col,
		"Order": order_col,
		"Family": family_col,
		"Genus": genus_col,
		"Species": species_col,
		"Spore formation": spore_col,
	}.items():
		if name is None:
			print(f"[Warn] Column not found in Excel: {needed}", file=sys.stderr)

	# Map FASTA basenames present in data dirs
	basename_to_path = scan_fasta_files(data_dirs)
	if not basename_to_path:
		print("[Error] No FASTA files found in provided data_dirs.", file=sys.stderr)
		sys.exit(1)

	# Keep only rows whose 'Fasta file' exists on disk
	meta_df = meta_df[meta_df["normalized_fasta"].isin(basename_to_path.keys())].copy()
	if meta_df.empty:
		print("[Info] No Bacillales FASTA files found in the provided data_dirs.", file=sys.stderr)
		sys.exit(0)

	# Build output records
	records: List[Dict[str, object]] = []
	keep_fastas: Set[str] = set()
	for _, row in meta_df.iterrows():
		fname = str(row["normalized_fasta"])
		fpath = basename_to_path.get(fname)
		if not fpath:
			continue
		length, gc_percent = compute_gc_and_length(fpath)
		phylum = row.get(phylum_col, "")
		clazz = row.get(class_col, "")
		order = row.get(order_col, "")
		family = row.get(family_col, "")
		genus = row.get(genus_col, "")
		species = row.get(species_col, "")
		sporulation = row.get(spore_col, "")
		records.append(
			{
				"fasta_name": fname,
				"phylum": phylum,
				"class": clazz,
				"order": order,
				"family": family,
				"genus": genus,
				"species": species,
				"sporulation": sporulation,
				"GC%": round(float(gc_percent), 4),
				"Genome Length": int(length),
			}
		)
		# Derive the tree tip label from FASTA basename (without extension)
		tree_label = strip_fasta_extension(fname)
		keep_fastas.add(tree_label)

	# Write CSV
	out_path = Path(args.output_csv)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	df_out = pd.DataFrame.from_records(
		records,
		columns=[
			"fasta_name",
			"phylum",
			"class",
			"order",
			"family",
			"genus",
			"species",
			"sporulation",
			"GC%",
			"Genome Length",
		],
	)
	df_out.to_csv(out_path, index=False)
	print(f"Wrote CSV with {len(df_out)} rows to: {out_path}")

	# Prune tree
	if not args.skip_tree:
		prune_tree_to_fasta_labels(args.tree_in, args.tree_out, keep_fastas)


if __name__ == "__main__":
	main()


