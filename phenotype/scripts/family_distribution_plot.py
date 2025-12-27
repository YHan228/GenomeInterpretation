#!/usr/bin/env python3
"""
Create a stacked bar plot showing Genus distribution across three data folders
(train, validation, test) for Bacillales FASTA files matched via an Excel file.

Defaults:
- Excel file: /home/yhan/GenomeInterpretation/sporulation/microbe.cards table S1.xlsx
- Data folders:
  - Train:      /vol/projects/BIFO/genomenet/yichen/phenotype/bacillales/train
  - Validation: /vol/projects/BIFO/genomenet/yichen/phenotype/bacillales/validation
  - Test:       /vol/projects/BIFO/genomenet/yichen/phenotype/bacillales/test
- Output dir:   /home/yhan/GenomeInterpretation/phenotype/plots

The script:
- Reads the Excel to map "Fasta file" -> "Genus"
- Scans the three folders for FASTA files (common extensions, supports .gz suffix)
- Matches files to Genus, counts per dataset, aggregates, and plots a stacked bar
- By default, includes an "Unknown" bin for files not present in the Excel mapping
- Plots all genera (no grouping)
"""

import argparse
import os
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd


DEFAULT_EXCEL_PATH = "/home/yhan/GenomeInterpretation/sporulation/microbe.cards table S1.xlsx"
DEFAULT_TRAIN_DIR = "/vol/projects/BIFO/genomenet/yichen/phenotype/bacillales/train"
DEFAULT_VALIDATION_DIR = "/vol/projects/BIFO/genomenet/yichen/phenotype/bacillales/validation"
DEFAULT_TEST_DIR = "/vol/projects/BIFO/genomenet/yichen/phenotype/bacillales/test"
DEFAULT_OUTPUT_DIR = "/home/yhan/GenomeInterpretation/phenotype/plots"
DEFAULT_OUTPUT_FILENAME = "genus_distribution_stacked_bar.png"

# Common FASTA file extensions (case-insensitive)
FASTA_EXTENSIONS = {".fa", ".fasta", ".fna", ".faa", ".fas", ".fsa"}


def parse_arguments() -> argparse.Namespace:
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(
		description="Create stacked bar plot of Genus distribution across datasets."
	)
	parser.add_argument(
		"--excel",
		default=DEFAULT_EXCEL_PATH,
		help='Path to taxonomy Excel file containing "Fasta file" and "Genus" columns.',
	)
	parser.add_argument(
		"--train_dir",
		default=DEFAULT_TRAIN_DIR,
		help="Path to training FASTA folder (recursively scanned).",
	)
	parser.add_argument(
		"--validation_dir",
		default=DEFAULT_VALIDATION_DIR,
		help="Path to validation FASTA folder (recursively scanned).",
	)
	parser.add_argument(
		"--test_dir",
		default=DEFAULT_TEST_DIR,
		help="Path to test FASTA folder (recursively scanned).",
	)
	parser.add_argument(
		"--output_dir",
		default=DEFAULT_OUTPUT_DIR,
		help="Directory to save the plot.",
	)
	parser.add_argument(
		"--output_filename",
		default=DEFAULT_OUTPUT_FILENAME,
		help="Filename for the saved plot (PNG).",
	)
	# Normalization is enabled by default; allow opting out with --no-normalize
	norm_group = parser.add_mutually_exclusive_group()
	norm_group.add_argument(
		"--normalize",
		dest="normalize",
		action="store_true",
		help="Plot proportions instead of raw counts (stacked to 1.0).",
	)
	norm_group.add_argument(
		"--no-normalize",
		dest="normalize",
		action="store_false",
		help="Plot raw counts (no normalization).",
	)
	parser.set_defaults(normalize=True)
	parser.add_argument(
		"--exclude-unknown",
		action="store_true",
		help="Exclude files whose genus is unknown (i.e., not found in Excel).",
	)
	return parser.parse_args()


def normalize_fasta_basename(filename: str) -> str:
	"""
	Return a normalized base name for matching:
	- Take os.path.basename
	- Strip a trailing .gz if present
	- Preserve the underlying extension (e.g., .fna, .fasta)
	"""
	basename = os.path.basename(filename)
	if basename.endswith(".gz"):
		basename = basename[:-3]
	return basename


def is_fasta_file(path: str, allowed_exts: Set[str]) -> bool:
	"""Return True if file has a FASTA-like extension (case-insensitive)."""
	lower = path.lower()
	for ext in allowed_exts:
		if lower.endswith(ext):
			return True
		if lower.endswith(ext + ".gz"):
			return True
	return False


def find_fasta_files(root_dir: str, allowed_exts: Set[str]) -> List[str]:
	"""Recursively collect FASTA file paths under root_dir."""
	if not os.path.isdir(root_dir):
		return []
	paths: List[str] = []
	for dirpath, _, filenames in os.walk(root_dir):
		for name in filenames:
			full = os.path.join(dirpath, name)
			if is_fasta_file(full, allowed_exts):
				paths.append(full)
	return paths


def find_column_name(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
	"""
	Attempt to find a column matching one of the candidate names.
	Try exact, case-insensitive exact, and case-insensitive substring match.
	"""
	columns_list = list(columns)
	lower_map = {c.lower(): c for c in columns_list}

	# 1) Exact
	for cand in candidates:
		if cand in columns_list:
			return cand

	# 2) Case-insensitive exact
	for cand in candidates:
		if cand.lower() in lower_map:
			return lower_map[cand.lower()]

	# 3) Case-insensitive substring
	for cand in candidates:
		cand_low = cand.lower()
		for c in columns_list:
			if cand_low in c.lower():
				return c
	return None


def load_fasta_to_genus_mapping(excel_path: str) -> Dict[str, str]:
	"""
	Load Excel and return a mapping: normalized 'Fasta file' basename -> 'Genus' string.
	If multiple rows per FASTA exist, the first non-null genus is used.
	"""
	if not os.path.isfile(excel_path):
		raise FileNotFoundError(f"Excel file not found: {excel_path}")

	taxonomy_dataframe = pd.read_excel(excel_path)
	if taxonomy_dataframe.empty:
		raise ValueError(f"Excel file appears empty: {excel_path}")

	fasta_col = find_column_name(taxonomy_dataframe.columns, ["Fasta file", "FASTA file", "Fasta", "fasta file"])
	genus_col = find_column_name(taxonomy_dataframe.columns, ["Genus", "genus"])
	if fasta_col is None or genus_col is None:
		raise KeyError(
			f"Could not find required columns. Available: {list(taxonomy_dataframe.columns)}; "
			f"required like 'Fasta file' and 'Genus'."
		)

	# Normalize 'Fasta file' to base names without .gz for matching with filesystem.
	taxonomy_dataframe["_normalized_fasta"] = (
		taxonomy_dataframe[fasta_col].astype(str).map(normalize_fasta_basename)
	)

	# Choose first non-null genus per normalized fasta
	grouped = taxonomy_dataframe.groupby("_normalized_fasta")[genus_col].first()
	mapping: Dict[str, str] = {}
	for fasta_name, genus in grouped.items():
		if pd.isna(genus):
			continue
		mapping[str(fasta_name)] = str(genus)
	return mapping


def count_genera_for_files(
	file_paths: Iterable[str],
	fasta_to_genus: Dict[str, str],
	include_unknown: bool,
	unknown_label: str = "Unknown",
) -> Counter:
	"""
	Count genera for given file paths using the provided mapping.
	If include_unknown=False, files missing in mapping are skipped.
	"""
	counter: Counter = Counter()
	for path in file_paths:
		key = normalize_fasta_basename(path)
		genus = fasta_to_genus.get(key)
		if genus is None or (isinstance(genus, float) and pd.isna(genus)):
			if not include_unknown:
				continue
			genus = unknown_label
		counter[str(genus)] += 1
	return counter


def prepare_plot_matrix(
	dataset_to_counts: Dict[str, Counter],
	ordered_families: List[str],
	dataset_order: List[str],
	normalize: bool,
) -> List[List[float]]:
	"""
	Create a matrix [families x datasets] with counts or proportions.
	When normalize=True and the plot has families on the X-axis, we normalize rows
	(per family) so each stacked bar sums to 1.
	"""
	matrix: List[List[float]] = []
	for family in ordered_families:
		values: List[float] = []
		for dataset in dataset_order:
			values.append(float(dataset_to_counts.get(dataset, Counter()).get(family, 0)))
		matrix.append(values)

	if normalize:
		# Normalize each family (row) so the stacked bar sums to 1
		for i in range(len(ordered_families)):
			row_sum = sum(matrix[i][j] for j in range(len(dataset_order)))
			if row_sum > 0:
				for j in range(len(dataset_order)):
					matrix[i][j] = matrix[i][j] / row_sum
	return matrix


def choose_colors(num_families: int, special_labels: Optional[Dict[str, str]] = None) -> List:
	"""
	Choose colors for families using matplotlib colormap. Optionally supply special colors
	for labels like 'Other' or 'Unknown' in special_labels mapping.
	"""
	cmap = cm.get_cmap("tab20", max(20, num_families))
	colors = [cmap(i % cmap.N) for i in range(num_families)]
	return colors


def plot_stacked_bar(
	dataset_order: List[str],
	ordered_families: List[str],
	matrix: List[List[float]],
	title: str,
	ylabel: str,
	output_path: str,
) -> None:
	"""Plot and save a stacked bar chart with genera on X-axis and datasets as colors."""
	num_datasets = len(dataset_order)
	num_families = len(ordered_families)

	# Dynamic figure width for readability with many families
	fig_width = max(12.0, min(48.0, 0.18 * max(1, num_families)))
	fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=150)

	indices = list(range(num_families))
	bottom = [0.0] * num_families

	# Choose consistent colors per dataset
	cmap = cm.get_cmap("tab10", max(3, num_datasets))
	dataset_colors = [cmap(i % cmap.N) for i in range(num_datasets)]

	for j, dataset in enumerate(dataset_order):
		values = [matrix[i][j] for i in range(num_families)]
		ax.bar(
			indices,
			values,
			bottom=bottom,
			label=dataset,
			color=dataset_colors[j],
			edgecolor="white",
			linewidth=0.2,
		)
		bottom = [bottom[i] + values[i] for i in range(num_families)]

	ax.set_xticks(indices)
	ax.set_xticklabels(ordered_families, rotation=90, ha="right")
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, ncol=1)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)

	fig.tight_layout()
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	fig.savefig(output_path)
	plt.close(fig)


def main() -> None:
	args = parse_arguments()

	# Load mapping from Excel
	try:
		fasta_to_genus = load_fasta_to_genus_mapping(args.excel)
	except Exception as exc:
		print(f"[Error] Failed to load Excel mapping: {exc}", file=sys.stderr)
		sys.exit(1)

	# Gather FASTA files for each dataset
	dataset_dirs = {
		"Train": args.train_dir,
		"Validation": args.validation_dir,
		"Test": args.test_dir,
	}
	dataset_order = ["Train", "Validation", "Test"]

	dataset_to_files: Dict[str, List[str]] = {}
	for name, directory in dataset_dirs.items():
		files = find_fasta_files(directory, FASTA_EXTENSIONS)
		dataset_to_files[name] = files

	# Count genera
	dataset_to_counts: Dict[str, Counter] = {}
	for name, files in dataset_to_files.items():
		counts = count_genera_for_files(
			file_paths=files,
			fasta_to_genus=fasta_to_genus,
			include_unknown=(not args.exclude_unknown),
		)
		dataset_to_counts[name] = counts

	# Order genera by total count descending across datasets
	total_counts: Counter = Counter()
	for counts in dataset_to_counts.values():
		for family, cnt in counts.items():
			total_counts[family] += cnt
	ordered_families = [f for f, _ in total_counts.most_common()]

	# Build matrix and plot
	matrix = prepare_plot_matrix(
		dataset_to_counts=dataset_to_counts,
		ordered_families=ordered_families,
		dataset_order=dataset_order,
		normalize=bool(args.normalize),
	)

	ylabel = "Proportion" if args.normalize else "Count"
	title = "Genus distribution across datasets"

	os.makedirs(args.output_dir, exist_ok=True)
	output_path = os.path.join(args.output_dir, args.output_filename)
	plot_stacked_bar(
		dataset_order=dataset_order,
		ordered_families=ordered_families,
		matrix=matrix,
		title=title,
		ylabel=ylabel,
		output_path=output_path,
	)

	print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
	main()


