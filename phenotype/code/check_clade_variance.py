#!/usr/bin/env python3
"""Check, for each phenotype, whether there exists a sufficiently large
taxonomic clade that exhibits label variation within it.

Reads the widely used metadata Excel (sporulation/microbe.cards table S1.xlsx)
via phenotype_utils.read_metadata_table and prints results to console.

Defaults:
- Phenotypes: ['Spore formation']
- Taxonomic ranks inspected: ['Order'] (and tree plot uses Family→Genus→samples if available)
- Minimum clade size: 20 (total samples in a clade)
- Optional minimum per-label: 0 (disabled by default)
"""

from __future__ import annotations

import argparse
import re
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import io

try:
    # Common, journal-level library for phylogenetic tree handling/plotting
    from Bio import Phylo  # type: ignore
except Exception as _bio_exc:  # pragma: no cover
    Phylo = None  # type: ignore
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception as _mpl_exc:  # pragma: no cover
    plt = None  # type: ignore
try:
    import sourmash  # type: ignore
    from sourmash import MinHash  # type: ignore
except Exception:
    sourmash = None  # type: ignore
try:
    import skbio  # type: ignore
    from skbio import TreeNode  # type: ignore
    from skbio.tree import nj as neighbor_joining  # type: ignore
    from skbio.stats.distance import DistanceMatrix  # type: ignore
except Exception:
    skbio = None  # type: ignore

try:
    from phenotype_utils import (
        PHENOTYPE_COLUMNS,
        read_metadata_table,
        normalize_label_value,
        phenotype_to_slug,
        phenotype_to_slug,
        DATA_ROOT,
    )
except ImportError:  # pragma: no cover
    # package-style relative import fallback
    from .phenotype_utils import (  # type: ignore
        PHENOTYPE_COLUMNS,
        read_metadata_table,
        normalize_label_value,
        phenotype_to_slug,
        DATA_ROOT,
    )


DEFAULT_METADATA_XLSX = str(Path("sporulation") / "microbe.cards table S1.xlsx")
DEFAULT_TAXON_RANKS: List[str] = ["Order"]


def _normalize_labels(series: pd.Series) -> pd.Series:
    """Normalize phenotype labels to lowercase canonical strings."""
    return series.map(normalize_label_value).astype(str).str.strip()


def _keep_binary_labels(df: pd.DataFrame, phenotype_col: str) -> pd.DataFrame:
    """Keep only rows labeled 'true' or 'false' (drop others/unknown)."""
    allowed = {"true", "false"}
    mask = df[phenotype_col].isin(allowed)
    return df[mask]


def _select_present_ranks(df: pd.DataFrame, ranks: Iterable[str]) -> List[str]:
    return [r for r in ranks if r in df.columns]


def _has_label_variation(counts: pd.Series, min_per_label: int = 0) -> bool:
    if counts.empty:
        return False
    if min_per_label > 0:
        counts = counts[counts >= int(min_per_label)]
    return counts.nunique(dropna=False) >= 2 and counts.sum() >= counts.values.sum()


def evaluate_phenotype(
    df: pd.DataFrame,
    phenotype: str,
    tax_ranks: List[str],
    min_clade_size: int,
    min_per_label: int,
) -> Tuple[bool, Dict[str, List[Tuple[str, int, Dict[str, int]]]]]:
    """Return (exists, details) for a phenotype across taxonomic ranks.

    details[rank] = list of tuples: (clade_name, total_size, label_counts_dict)
    containing only clades that meet the criteria and have label variation.
    """
    col = phenotype
    if col not in df.columns:
        return False, {}

    work = df[[*tax_ranks, col]].copy()
    work[col] = _normalize_labels(work[col])
    work = work[work[col].str.len() > 0]
    # drop others/unknown
    work = _keep_binary_labels(work, col)
    exists_any = False
    details: Dict[str, List[Tuple[str, int, Dict[str, int]]]] = defaultdict(list)

    for rank in tax_ranks:
        if rank not in work.columns:
            continue
        grp = work.dropna(subset=[rank]).groupby(rank, dropna=True)
        for clade, sub in grp:
            total = int(len(sub))
            if total < int(min_clade_size):
                continue
            counts = sub[col].value_counts()
            if _has_label_variation(counts, min_per_label=min_per_label):
                exists_any = True
                details[rank].append((str(clade), total, {str(k): int(v) for k, v in counts.to_dict().items()}))

        if details.get(rank):
            # sort clades by size desc
            details[rank].sort(key=lambda x: x[1], reverse=True)

    return exists_any, details


def print_results(
    phenotype: str,
    exists_any: bool,
    details: Dict[str, List[Tuple[str, int, Dict[str, int]]]],
    top_k: int = 5,
) -> None:
    status = "YES" if exists_any else "NO"
    print(f"Phenotype: {phenotype} | any clade with label variation meeting size threshold? {status}")
    if not exists_any:
        return
    for rank, items in details.items():
        if not items:
            continue
        print(f"  Rank: {rank} | qualifying clades: {len(items)}")
        for name, size, counts in items[: int(top_k)]:
            counts_str = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            print(f"    - {name} | n={size} | labels={{ {counts_str} }}")


def _sanitize_newick_name(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.\-]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s or "NA"


def _build_taxonomy_newick_and_labels(
    df: pd.DataFrame,
    phenotype_col: str,
    order_name: str,
    max_leaves: Optional[int] = None,
    available_only: Optional[Set[str]] = None,
) -> Tuple[str, Dict[str, str]]:
    """Construct a taxonomy-based Newick tree within a given Order.

    Hierarchy: Family -> Genus -> Sample (FASTA basename). Returns (newick, leaf->label).
    """
    sub = df.copy()
    sub = sub[sub.get("Order").astype(str) == str(order_name)] if "Order" in sub.columns else sub.iloc[0:0]
    if phenotype_col not in sub.columns:
        return "();", {}
    # Normalize labels, keep only labeled rows
    sub[phenotype_col] = _normalize_labels(sub[phenotype_col])
    sub = sub[sub[phenotype_col].str.len() > 0]
    sub = _keep_binary_labels(sub, phenotype_col)
    if sub.empty:
        return "();", {}

    # Determine sample id
    sample_col = "Fasta file"
    if sample_col not in sub.columns:
        # Try normalized
        sample_col = "Fasta file_norm" if "Fasta file_norm" in sub.columns else sub.columns[0]

    # Restrict to genomes available in the current dataset (by basename, case-insensitive)
    # If available_only is provided (even if empty), restrict strictly to those basenames
    if available_only is not None:
        allowed = set(str(s).strip().lower() for s in available_only)
        basenames_lc = sub[sample_col].astype(str).map(lambda v: Path(v).name.strip().lower())
        sub = sub.loc[basenames_lc.isin(allowed)].copy()
        if sub.empty:
            return "();", {}

    # Build nested taxonomy
    families: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    leaf_to_label: Dict[str, str] = {}

    def _basename(val: object) -> str:
        return Path(str(val)).name

    for _, row in sub.iterrows():
        fam = str(row.get("Family", "NA"))
        gen = str(row.get("Genus", "NA"))
        sample = _basename(row.get(sample_col, ""))
        label = str(row.get(phenotype_col, ""))
        if not sample:
            continue
        leaf = _sanitize_newick_name(sample)
        families[fam][gen].append((leaf, label))
        leaf_to_label[leaf] = label

    # Optionally cap leaves
    if max_leaves is not None:
        remaining = int(max_leaves)
        for fam in list(families.keys()):
            for gen in list(families[fam].keys()):
                if remaining <= 0:
                    families[fam][gen] = []
                    continue
                lst = families[fam][gen]
                if len(lst) > remaining:
                    families[fam][gen] = lst[:remaining]
                    remaining = 0
                else:
                    remaining -= len(lst)

    # Compose Newick
    family_newicks: List[str] = []
    for fam, gen_map in families.items():
        genus_newicks: List[str] = []
        for gen, leaves in gen_map.items():
            if not leaves:
                continue
            leaf_names = [leaf for leaf, _lab in leaves]
            genus_newicks.append("(" + ",".join(leaf_names) + ")" + _sanitize_newick_name(gen))
        if not genus_newicks:
            continue
        family_newicks.append("(" + ",".join(genus_newicks) + ")" + _sanitize_newick_name(fam))

    if not family_newicks:
        return "();", {}
    newick = "(" + ",".join(family_newicks) + ")root;"
    return newick, leaf_to_label


def _plot_top_order_tree(
    df: pd.DataFrame,
    phenotype_col: str,
    order_name: str,
    out_dir: Path,
    available_only: Optional[Set[str]] = None,
    show_tip_labels: bool = False,
) -> Optional[Path]:
    if Phylo is None or plt is None:  # pragma: no cover
        print("[plot] Biopython (Bio.Phylo) and matplotlib are required for plotting. Please install biopython and matplotlib.")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    newick, leaf_labels = _build_taxonomy_newick_and_labels(
        df, phenotype_col, order_name, max_leaves=None, available_only=available_only
    )
    if not leaf_labels:
        print("[plot] No labeled leaves to plot for the selected Order.")
        return None

    import math
    from io import StringIO

    tree = Phylo.read(StringIO(newick), "newick")

    # Color mapping
    cmap = {
        "true": "#d62728",   # red
        "false": "#1f77b4",  # blue
    }
    default_color = "#7f7f7f"

    # Set branch colors on terminals
    for clade in tree.get_terminals():
        label = leaf_labels.get(str(clade.name), "")
        color = cmap.get(str(label).lower(), default_color)
        try:
            clade.color = color  # Biopython draws branch in this color
        except Exception:
            pass

    # Figure size + label policy
    n_leaves = max(1, len(tree.get_terminals()))
    height = min(30, max(8, int(math.ceil(n_leaves / 12.0))))
    # Increase width only when labels are shown
    max_label_len = max((len(str(c.name)) for c in tree.get_terminals()), default=10)
    width = 10 if not show_tip_labels else min(24, 10 + max_label_len * 0.08)
    plt.figure(figsize=(width, height))

    label_func = (lambda cl: str(cl.name) if (show_tip_labels and cl.is_terminal()) else None)
    Phylo.draw(tree, do_show=False, label_func=label_func)

    # Recolor terminal label texts to match branch color (when visible)
    ax = plt.gca()
    if show_tip_labels:
        texts = [obj for obj in ax.findobj() if getattr(obj, "get_text", None)]
        for t in texts:
            name = str(t.get_text()).strip()
            if not name:
                continue
            color = cmap.get(str(leaf_labels.get(_sanitize_newick_name(name), "")).lower(), default_color)
            try:
                t.set_color(color)
                t.set_fontsize(6)
                t.set_alpha(0.9)
            except Exception:
                pass

    # Legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=v, label=k) for k, v in (sorted(cmap.items()))]
    handles.append(mpatches.Patch(color=default_color, label="other/unknown"))
    plt.legend(handles=handles, loc="lower right", framealpha=0.9, fontsize=9)
    plt.title(f"{phenotype_col} | Order: {order_name}")
    # De-clutter axes
    try:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    except Exception:
        pass

    slug = phenotype_to_slug(phenotype_col)
    order_s = _sanitize_newick_name(order_name)
    out_path = out_dir / f"{slug}_order_{order_s}.png"
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=300)
    plt.close()
    print(f"[plot] Saved tree: {out_path}")
    return out_path

def _collect_available_basenames(seq_dirs_list: Iterable[str]) -> Set[str]:
    """Discover available genome basenames (case-insensitive) in provided directories.

    Includes both compressed and uncompressed basename variants to improve matching
    against metadata that might list either form (e.g., *.fna vs *.fna.gz).
    """
    allowed_exts = (".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz")
    available: Set[str] = set()
    for d in seq_dirs_list:
        try:
            p = Path(str(d).strip())
            if not p.exists() or not p.is_dir():
                continue
            for fn in p.iterdir():
                if not fn.is_file():
                    continue
                name_lc = fn.name.strip().lower()
                if name_lc.endswith(allowed_exts):
                    available.add(name_lc)
                    # add non-.gz variant if present to match metadata basenames
                    if name_lc.endswith(".gz"):
                        try:
                            no_gz = fn.stem.strip().lower()  # e.g., *.fna.gz -> *.fna
                            available.add(no_gz)
                        except Exception:
                            pass
        except Exception:
            continue
    return available


def _find_fasta_path(name: str, search_dirs: Iterable[str]) -> Optional[Path]:
    base = Path(name).name
    candidates = [base, base + ".gz"]
    for d in search_dirs:
        try:
            p = Path(d)
            if not p.exists() or not p.is_dir():
                continue
            for cand in candidates:
                fp = p / cand
                if fp.exists():
                    return fp
        except Exception:
            continue
    return None


def _minhash_from_fasta(path: Path, k: int, scaled: int) -> Optional[MinHash]:
    if sourmash is None:
        print("[phylo] sourmash is required for MinHash; please install sourmash")
        return None
    try:
        mh = MinHash(n=0, ksize=int(k), scaled=int(scaled))
        # streaming k-mer hashing
        with open(path, "rt", encoding="utf-8", errors="ignore") as f:
            seq = []
            for line in f:
                if line.startswith(">"):
                    if seq:
                        mh.add_sequence("".join(seq), force=True)
                        seq = []
                    continue
                seq.append(line.strip())
            if seq:
                mh.add_sequence("".join(seq), force=True)
        return mh
    except Exception as exc:
        print(f"[phylo] Failed to sketch {path.name}: {exc}")
        return None


def _jaccard_distance(a: MinHash, b: MinHash) -> float:
    try:
        sim = a.jaccard(b)
    except Exception:
        sim = 0.0
    sim = max(0.0, min(1.0, float(sim)))
    return float(1.0 - sim)


def _infer_and_plot_true_tree(
    df: pd.DataFrame,
    phenotype_col: str,
    order_name: str,
    out_dir: Path,
    k: int,
    scaled: int,
    seq_dirs: List[str],
    max_leaves: Optional[int],
    show_tip_labels: bool,
) -> Optional[Path]:
    if Phylo is None or plt is None or sourmash is None or skbio is None:  # pragma: no cover
        print("[phylo] biopython, matplotlib, sourmash, and scikit-bio are required. Please install them.")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    sub = df.copy()
    sub = sub[sub.get("Order").astype(str) == str(order_name)] if "Order" in sub.columns else sub.iloc[0:0]
    if sub.empty:
        print("[phylo] No samples found for the specified Order.")
        return None

    # Normalize labels
    sub[phenotype_col] = _normalize_labels(sub[phenotype_col])
    sub = sub[sub[phenotype_col].str.len() > 0]
    sub = _keep_binary_labels(sub, phenotype_col)
    if sub.empty:
        print("[phylo] No labeled samples for this Order.")
        return None

    # Prepare sample list
    sample_col = "Fasta file" if "Fasta file" in sub.columns else ("Fasta file_norm" if "Fasta file_norm" in sub.columns else None)
    if sample_col is None:
        print("[phylo] Missing FASTA file column in metadata.")
        return None

    # Restrict to genomes actually present in the provided sequence directories (by basename)
    seq_dirs_list = [str(d).strip() for d in seq_dirs if str(d).strip()]
    available: Set[str] = _collect_available_basenames(seq_dirs_list)
    norm_basenames = sub[sample_col].astype(str).map(lambda v: Path(v).name.strip().lower())
    # If sequence_dirs are provided but contain no genomes, do not fall back to the full dataset
    if not available:
        print("[phylo] No genomes found in the provided sequence_dirs.")
        return None
    sub = sub.loc[norm_basenames.isin(available)].copy()
    if sub.empty:
        print("[phylo] No available genomes found for this Order in provided sequence_dirs.")
        return None

    items: List[Tuple[str, str, str]] = []  # (leaf_name, label, display_name)
    for _, row in sub.iterrows():
        fname = str(row.get(sample_col, "")).strip()
        if not fname:
            continue
        leaf = _sanitize_newick_name(Path(fname).name)
        # build readable display name: Genus_species or truncated basename
        genus = str(row.get("Genus", "")).strip()
        species = str(row.get("Species", "")).strip()
        if genus and species:
            disp = _sanitize_newick_name(f"{genus}_{species}")
        elif genus:
            disp = _sanitize_newick_name(genus)
        else:
            base = Path(fname).stem
            disp = _sanitize_newick_name(base[:32])
        items.append((leaf, str(row.get(phenotype_col, "")), disp))

    # Deduplicate while preserving order
    seen = set()
    uniq_items: List[Tuple[str, str, str]] = []
    for leaf, lab, disp in items:
        if leaf in seen:
            continue
        seen.add(leaf)
        uniq_items.append((leaf, lab, disp))

    # Cap leaves for readability
    if max_leaves is not None and len(uniq_items) > int(max_leaves):
        # stratified cap by label to preserve both classes
        max_leaves = int(max_leaves)
        by_label: Dict[str, List[Tuple[str, str, str]]] = {"true": [], "false": []}
        for leaf, lab, disp in uniq_items:
            if lab in by_label:
                by_label[lab].append((leaf, lab, disp))
        n_true = len(by_label["true"]) or 1
        n_false = len(by_label["false"]) or 1
        # proportional allocation with floor, ensure at least 1 if present
        n_tgt_true = max(1 if n_true > 0 else 0, int(round(max_leaves * (n_true / (n_true + n_false)))))
        n_tgt_false = max(1 if n_false > 0 else 0, max_leaves - n_tgt_true)
        uniq_items = by_label["true"][:n_tgt_true] + by_label["false"][:n_tgt_false]

    # Locate FASTA files and sketch
    leaves = [leaf for leaf, _lab, _disp in uniq_items]
    labels = {leaf: lab for leaf, lab, _disp in uniq_items}
    display_map = {leaf: disp for leaf, _lab, disp in uniq_items}
    paths: List[Optional[Path]] = [_find_fasta_path(leaf, seq_dirs_list) for leaf in leaves]
    # If not found by basename, try direct names
    for i, p in enumerate(paths):
        if p is None:
            p2 = _find_fasta_path(leaves[i].replace("_", "."), seq_dirs_list)
            paths[i] = p2

    valid = [(leaf, labels[leaf], display_map[leaf], fp) for leaf, fp in zip(leaves, paths) if fp is not None]
    if len(valid) < 3:
        print("[phylo] Fewer than 3 genomes found; cannot infer NJ tree.")
        return None

    sketches: List[Tuple[str, str, str, MinHash]] = []
    for leaf, lab, disp, fp in valid:
        mh = _minhash_from_fasta(fp, k=k, scaled=scaled)
        if mh is not None:
            sketches.append((leaf, lab, disp, mh))

    if len(sketches) < 3:
        print("[phylo] Fewer than 3 successful sketches; cannot infer NJ tree.")
        return None

    # Distance matrix (Jaccard distance of MinHash sketches)
    names = [leaf for leaf, _lab, _disp, _mh in sketches]
    n = len(names)
    dmat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dist = _jaccard_distance(sketches[i][3], sketches[j][3])
            dmat[i, j] = dmat[j, i] = dist

    dm = DistanceMatrix(dmat, ids=names)
    tree_node: TreeNode = neighbor_joining(dm)
    # improve readability: ladderize the tree
    try:
        tree_node.ladderize()
    except Exception:
        pass

    # Convert to Biopython and plot
    from io import StringIO
    _buf = StringIO()
    # skbio's write requires a file-like and format specification
    tree_node.write(_buf, format="newick")
    newick = _buf.getvalue()
    tree = Phylo.read(StringIO(newick), "newick")

    cmap = {
        "true": "#d62728",
        "false": "#1f77b4",
    }
    default_color = "#7f7f7f"
    # Assign colors and replace names with readable display names
    display_to_color: Dict[str, str] = {}
    for clade in tree.get_terminals():
        orig = str(clade.name)
        color = cmap.get(str(labels.get(orig, "")).lower(), default_color)
        disp = display_map.get(orig, orig)
        try:
            clade.color = color
        except Exception:
            pass
        clade.name = disp
        display_to_color[disp] = color

    import math
    n_leaves = max(1, len(tree.get_terminals()))
    height = min(40, max(8, int(math.ceil(n_leaves / 10.0))))
    width = 16
    plt.figure(figsize=(width, height))
    label_func = (lambda cl: str(cl.name) if (show_tip_labels and cl.is_terminal()) else None)
    Phylo.draw(tree, do_show=False, label_func=label_func)
    ax = plt.gca()
    texts = [obj for obj in ax.findobj() if getattr(obj, "get_text", None)]
    for t in texts:
        nm = str(t.get_text()).strip()
        if not nm:
            continue
        if show_tip_labels:
            t.set_color(display_to_color.get(nm, default_color))
            try:
                t.set_fontsize(6)
            except Exception:
                pass
        else:
            # hide tip labels entirely
            try:
                t.set_visible(False)
            except Exception:
                pass

    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=v, label=k) for k, v in (sorted(cmap.items()))]
    handles.append(mpatches.Patch(color=default_color, label="other/unknown"))
    plt.legend(handles=handles, loc="lower right", framealpha=0.9, fontsize=9)
    n_plot = len(tree.get_terminals())
    # Compute full n and label counts from pre-filtered df for the selected Order
    full_sub = df.copy()
    full_sub = full_sub[full_sub.get("Order").astype(str) == str(order_name)] if "Order" in full_sub.columns else full_sub.iloc[0:0]
    full_sub[phenotype_col] = _normalize_labels(full_sub[phenotype_col])
    full_sub = full_sub[full_sub[phenotype_col].str.len() > 0]
    full_sub = _keep_binary_labels(full_sub, phenotype_col)
    full_n = int(len(full_sub))
    vc = full_sub[phenotype_col].value_counts()
    n_true = int(vc.get("true", 0))
    n_false = int(vc.get("false", 0))
    plt.title(f"{phenotype_col} | Order (true tree): {order_name}  |  n_total={full_n} (true:{n_true}, false:{n_false})  |  n_plotted={n_plot}")
    # De-clutter axes
    try:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    except Exception:
        pass

    slug = phenotype_to_slug(phenotype_col)
    order_s = _sanitize_newick_name(order_name)
    out_path = out_dir / f"{slug}_order_{order_s}_true.png"
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=300)
    plt.close()
    print(f"[phylo] Saved true tree: {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Check label variance within taxonomic clades per phenotype")
    ap.add_argument("--metadata_xlsx", type=str, default=DEFAULT_METADATA_XLSX, help="Path to metadata Excel")
    ap.add_argument(
        "--phenotypes",
        type=str,
        default="Spore formation",
        help="Comma-separated phenotype columns to evaluate or 'all' for defaults",
    )
    ap.add_argument(
        "--tax_ranks",
        type=str,
        default=",".join(DEFAULT_TAXON_RANKS),
        help="Comma-separated taxonomy ranks to consider (must exist in Excel)",
    )
    ap.add_argument("--min_clade_size", type=int, default=20, help="Minimum total samples per clade")
    ap.add_argument(
        "--min_per_label",
        type=int,
        default=0,
        help="Optional minimum samples per label within a clade (0 disables)",
    )
    ap.add_argument("--top_k", type=int, default=5, help="How many top clades per rank to print")
    ap.add_argument(
        "--order",
        type=str,
        default=None,
        help="If provided, force plotting this Order (e.g., 'Bacillales') instead of choosing top qualifying Order",
    )
    ap.add_argument(
        "--plot_top_order",
        action="store_true",
        help="If 'Order' is among ranks, plot the top qualifying Order tree colored by labels",
    )
    ap.set_defaults(plot_top_order=True)
    # True phylogeny inference options
    ap.add_argument(
        "--infer_true_tree",
        action="store_true",
        help="Infer a true phylogeny for the top Order using MinHash distances + NJ",
    )
    ap.add_argument(
        "--kmer_size",
        type=int,
        default=31,
        help="k-mer size for MinHash sketching",
    )
    ap.add_argument(
        "--scaled",
        type=int,
        default=1000,
        help="MinHash scaled parameter (downsampling factor)",
    )
    ap.add_argument(
        "--sequence_dirs",
        type=str,
        default=",".join([str(DATA_ROOT / d) for d in ["train", "validation", "test"]]),
        help="Comma-separated directories to search for FASTA files",
    )
    ap.add_argument(
        "--max_leaves_plot",
        type=int,
        default=2000,
        help="Cap number of leaves in the plotted tree for readability (None to disable)",
    )
    ap.add_argument(
        "--show_tip_labels",
        action="store_true",
        help="Show tip labels on the plotted tree (off by default)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    xlsx_path = os.path.abspath(str(args.metadata_xlsx))
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Metadata Excel not found: {xlsx_path}")

    md = read_metadata_table(Path(xlsx_path))

    # Restrict metadata to the provided Order (if any) before evaluation
    if args.order and ("Order" in md.columns):
        md = md[md["Order"].astype(str) == str(args.order)].copy()

    # Restrict metadata to genomes present under the provided sequence directories
    seq_dirs_list = [p for p in str(args.sequence_dirs).split(",") if p]
    available_basenames: Set[str] = _collect_available_basenames(seq_dirs_list)
    # If sequence_dirs are provided but contain no genomes, do not proceed with global metadata
    if not available_basenames:
        print("No genomes discovered under --sequence_dirs; aborting to avoid using the full dataset.")
        return
    # Determine sample column and filter by basename presence
    sample_col = "Fasta file" if "Fasta file" in md.columns else ("Fasta file_norm" if "Fasta file_norm" in md.columns else None)
    if sample_col is not None:
        basenames_lc = md[sample_col].astype(str).map(lambda v: Path(v).name.strip().lower())
        md = md.loc[basenames_lc.isin(available_basenames)].copy()
    # If no matching rows remain, exit early
    if md.empty:
        print("No metadata rows match genomes in --sequence_dirs (after optional Order filter). Nothing to evaluate.")
        return

    tax_ranks_req = [r.strip() for r in str(args.tax_ranks).split(",") if r.strip()]
    tax_ranks = _select_present_ranks(md, tax_ranks_req)
    if not tax_ranks:
        print("No requested taxonomy ranks present in metadata; nothing to evaluate.")
        return

    if str(args.phenotypes).strip().lower() == "all":
        phenotypes = [p for p in PHENOTYPE_COLUMNS if p in md.columns]
    else:
        phenotypes = [p.strip() for p in str(args.phenotypes).split(",") if p.strip()]
        phenotypes = [p for p in phenotypes if p in md.columns]

    if not phenotypes:
        print("No phenotype columns found to evaluate.")
        return

    print(
        f"Evaluating {len(phenotypes)} phenotypes across ranks {tax_ranks} | "
        f"min_clade_size={int(args.min_clade_size)} | min_per_label={int(args.min_per_label)}"
    )

    any_ok = False
    first_details: Optional[Dict[str, List[Tuple[str, int, Dict[str, int]]]]] = None
    first_pheno: Optional[str] = None
    for pheno in phenotypes:
        ok, details = evaluate_phenotype(
            md,
            phenotype=pheno,
            tax_ranks=tax_ranks,
            min_clade_size=int(args.min_clade_size),
            min_per_label=int(args.min_per_label),
        )
        if first_details is None:
            first_details = details
            first_pheno = pheno
        any_ok = any_ok or ok
        print_results(pheno, ok, details, top_k=int(args.top_k))

    print(f"Overall: any phenotype with qualifying clade? {'YES' if any_ok else 'NO'}")

    # Optional: plot the specified Order (or top Order) tree for the first phenotype evaluated
    if args.plot_top_order and first_pheno is not None and ("Order" in tax_ranks):
        try:
            # Determine which Order to plot
            selected_order: Optional[str] = None
            if args.order:
                selected_order = str(args.order)
            elif first_details is not None:
                order_items = first_details.get("Order", [])
                if order_items:
                    selected_order = order_items[0][0]
            if selected_order:
                out_dir = Path("phenotype/plots/clade_variance")
                # Compute available FASTA basenames for taxonomy-based plotting
                available: Set[str] = _collect_available_basenames(seq_dirs_list)
                if args.infer_true_tree:
                    _infer_and_plot_true_tree(
                        md,
                        phenotype_col=first_pheno,
                        order_name=selected_order,
                        out_dir=out_dir,
                        k=args.kmer_size,
                        scaled=args.scaled,
                        seq_dirs=seq_dirs_list,
                        max_leaves=int(args.max_leaves_plot) if args.max_leaves_plot is not None else None,
                        show_tip_labels=bool(args.show_tip_labels),
                    )
                else:
                    _plot_top_order_tree(
                        md,
                        phenotype_col=first_pheno,
                        order_name=selected_order,
                        out_dir=out_dir,
                        available_only=available,
                        show_tip_labels=bool(args.show_tip_labels),
                    )
        except Exception as exc:
            print(f"[plot] Warning: failed to produce Order tree plot: {exc}")


if __name__ == "__main__":
    main()


