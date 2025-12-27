#!/usr/bin/env python3
"""
Infer and/or plot a phylogenetic tree using PhyloPhlAn, rendered in a circular layout.

Two modes:
1) Run PhyloPhlAn on genomes/proteomes discovered in --sequence_dirs, then plot.
2) Plot an existing Newick tree passed via --tree_file.

Requirements:
- PhyloPhlAn 3.x CLI available in PATH if --run_phylophlan is used.
  Install via conda: `conda install -c bioconda phylophlan`
  Project page: https://huttenhower.sph.harvard.edu/phylophlan/
- ETE3 for circular plotting: `pip install ete3` (or conda: `conda install -c etetoolkit ete3`)

Notes:
- This is a lightweight wrapper; it does not manage PhyloPhlAn databases.
- If you already have a Newick file from PhyloPhlAn, pass it via --tree_file to only plot.
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Set, Tuple

# Optional imports
try:  # type: ignore
    from ete3 import Tree, TreeStyle
except Exception:  # pragma: no cover
    Tree = None  # type: ignore
    TreeStyle = None  # type: ignore


def _discover_sequences(sequence_dirs: List[str]) -> List[Path]:
    allowed_genome_exts = (".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz")
    allowed_protein_exts = (".faa", ".pep", ".faa.gz", ".pep.gz")
    allowed = allowed_genome_exts + allowed_protein_exts
    found: List[Path] = []
    seen: Set[str] = set()
    for d in sequence_dirs:
        p = Path(str(d).strip())
        if not p.exists() or not p.is_dir():
            continue
        for fn in p.iterdir():
            if not fn.is_file():
                continue
            low = fn.name.lower()
            if not low.endswith(allowed):
                continue
            # Deduplicate by lowercase basename
            if low in seen:
                continue
            seen.add(low)
            found.append(fn.resolve())
    return found


def _looks_like_proteins(paths: List[Path]) -> bool:
    protein_exts = (".faa", ".pep", ".faa.gz", ".pep.gz")
    return any(str(p).lower().endswith(protein_exts) for p in paths)


def _safe_symlink_or_copy(src: Path, dst: Path) -> None:
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
    except Exception:
        pass
    try:
        dst.symlink_to(src)
    except Exception:
        # Fallback to copy if symlink not permitted
        shutil.copy2(str(src), str(dst))


def _prepare_input_dir(files: List[Path], input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    for fp in files:
        # Preserve original basename; PhyloPhlAn uses filenames as identifiers
        target = input_dir / fp.name
        _safe_symlink_or_copy(fp, target)


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _find_newick_file(out_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for ext in (".nwk", ".tree", ".tre", ".newick"):
        candidates.extend(out_dir.rglob(f"*{ext}"))
    if not candidates:
        return None
    # Prefer the most recently modified candidate
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _candidate_configs_dirs(explicit_folder: Optional[str]) -> List[Path]:
    candidates: List[Path] = []
    # 1) explicit
    if explicit_folder:
        p = Path(str(explicit_folder)).expanduser().resolve()
        candidates.append(p)
    # 2) env var
    env = os.environ.get("PHYLOPHLAN_CONFIGS", "").strip()
    if env:
        candidates.append(Path(env).expanduser().resolve())
    # 3) common package locations along sys.path
    for sp in list(sys.path):
        try:
            base = Path(sp)
            if not base.exists() or not base.is_dir():
                continue
            # phylophlan/phylophlan_configs
            p1 = base / "phylophlan" / "phylophlan_configs"
            if p1.exists() and p1.is_dir():
                candidates.append(p1.resolve())
            # bare phylophlan_configs
            p2 = base / "phylophlan_configs"
            if p2.exists() and p2.is_dir():
                candidates.append(p2.resolve())
        except Exception:
            continue
    # 4) system share folder (some installs)
    sys_share = Path("/usr/share/phylophlan/phylophlan_configs")
    if sys_share.exists() and sys_share.is_dir():
        candidates.append(sys_share)
    # De-duplicate while preserving order
    seen: Set[str] = set()
    uniq: List[Path] = []
    for c in candidates:
        s = str(c)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(c)
    return uniq


def _autodetect_config(seq_type: str, explicit_folder: Optional[str]) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (config_file, configs_folder) if found; otherwise (None, None)."""
    cfg_name = "supertree_nt.cfg" if str(seq_type) == "n" else "supertree_aa.cfg"
    for d in _candidate_configs_dirs(explicit_folder):
        cand = d / cfg_name
        if cand.exists() and cand.is_file():
            return cand.resolve(), d.resolve()
    return None, None


def _run_phylophlan(
    input_dir: Path,
    work_dir: Path,
    threads: int,
    config_path: Optional[str],
    configs_folder: Optional[str],
    database: str,
    extra_args: Optional[str],
    diversity: str,
    seq_type: str,
) -> Path:
    if not _which("phylophlan"):
        raise RuntimeError(
            "PhylophlAn CLI not found in PATH. Install with 'conda install -c bioconda phylophlan'"
        )

    work_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        "phylophlan",
        "-i",
        str(input_dir),
        "-o",
        str(work_dir),
        "-d",
        str(database),
        "--diversity",
        str(diversity),
        "-t",
        str(seq_type),  # 'n' for nucleotides, 'a' for amino acids
        "--nproc",
        str(int(threads)),
    ]
    # Ensure a config file is provided, either explicit or auto-detected
    cfg_file: Optional[Path]
    cfg_dir: Optional[Path]
    if config_path:
        cfg_file = Path(config_path).expanduser().resolve()
        if not cfg_file.exists():
            raise RuntimeError(f"Provided PhyloPhlAn config not found: {cfg_file}")
        cfg_dir = cfg_file.parent
    else:
        cfg_file, cfg_dir = _autodetect_config(seq_type=seq_type, explicit_folder=configs_folder)
        if not cfg_file:
            raise RuntimeError(
                "Could not locate a PhyloPhlAn config file. Pass --config pointing to "
                "supertree_nt.cfg (genomes) or supertree_aa.cfg (proteomes), or set --configs_folder."
            )
    cmd.extend(["-f", str(cfg_file)])
    # If a configs folder is available, pass it to avoid internal warnings
    if configs_folder:
        cmd.extend(["--configs_folder", str(Path(configs_folder).expanduser().resolve())])
    elif cfg_dir:
        cmd.extend(["--configs_folder", str(cfg_dir)])
    if extra_args:
        # naive split, assume caller passes a simple space-delimited string
        cmd.extend(str(extra_args).split())

    print("[phylophlan] Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "PhyloPhlAn failed. Ensure configs are available and external tools in the config are installed. "
            "You can pass --config and --configs_folder explicitly."
        ) from exc

    tree_path = _find_newick_file(work_dir)
    if not tree_path:
        raise RuntimeError("PhylophlAn run completed, but no Newick file was found in output directory.")
    print(f"[phylophlan] Found tree: {tree_path}")
    return tree_path


def _plot_circular(newick_path: Path, out_png: Path, show_tip_labels: bool, width_px: int) -> None:
    if Tree is None or TreeStyle is None:
        raise RuntimeError(
            "ETE3 is required for circular plotting. Install with 'pip install ete3' or 'conda install -c etetoolkit ete3'"
        )
    t = Tree(str(newick_path))
    ts = TreeStyle()
    ts.mode = "c"  # circular
    ts.show_leaf_name = bool(show_tip_labels)
    ts.arc_start = 0
    ts.arc_span = 360
    # Reasonable scale; users can change with --image_width_px
    out_png.parent.mkdir(parents=True, exist_ok=True)
    t.render(str(out_png), tree_style=ts, w=int(width_px), units="px")
    print(f"[plot] Saved circular tree: {out_png}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run/plot PhyloPhlAn tree in circular layout")
    ap.add_argument(
        "--sequence_dirs",
        type=str,
        default="",
        help="Comma-separated directories containing genome/proteome FASTA files (used if --run_phylophlan)",
    )
    ap.add_argument(
        "--run_phylophlan",
        action="store_true",
        help="Run PhyloPhlAn on discovered inputs before plotting",
    )
    ap.add_argument(
        "--phylophlan_out_dir",
        type=str,
        default=str(Path("phenotype") / "phylophlan_workdir"),
        help="Working/output directory for PhyloPhlAn run",
    )
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional PhyloPhlAn config file (e.g., supertree_aa.cfg or supertree_nt.cfg)",
    )
    ap.add_argument(
        "--configs_folder",
        type=str,
        default=None,
        help="Optional folder containing PhyloPhlAn configs (used for autodetection and passed to CLI)",
    )
    ap.add_argument(
        "--database",
        type=str,
        default="phylophlan",
        help="PhylophlAn database to use (default: phylophlan)",
    )
    ap.add_argument("--threads", type=int, default=8, help="Number of CPU threads for PhyloPhlAn")
    ap.add_argument(
        "--phylophlan_args",
        type=str,
        default="",
        help="Extra args appended to the phylophlan command",
    )
    ap.add_argument(
        "--diversity",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="PhyloPhlAn diversity of input (required by PhyloPhlAn)",
    )
    ap.add_argument(
        "--seq_type",
        type=str,
        choices=["n", "a"],
        default=None,
        help="Sequence type: 'n' nucleotides (genomes) or 'a' amino acids (proteomes). Auto-detected if omitted.",
    )
    ap.add_argument(
        "--tree_file",
        type=str,
        default=None,
        help="Existing Newick tree to plot (skip running PhyloPhlAn)",
    )
    ap.add_argument(
        "--out_png",
        type=str,
        default=str(Path("phenotype") / "plots" / "phylophlan" / "phylophlan_tree_circular.png"),
        help="Output PNG path for the circular tree",
    )
    ap.add_argument("--show_tip_labels", action="store_true", help="Show leaf names on the plot")
    ap.add_argument("--image_width_px", type=int, default=2000, help="Rendered image width in pixels")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    tree_path: Optional[Path] = None

    if args.tree_file:
        tree_path = Path(args.tree_file)
        if not tree_path.exists():
            raise FileNotFoundError(f"Tree file not found: {tree_path}")
    elif args.run_phylophlan:
        seq_dirs = [p for p in str(args.sequence_dirs).split(",") if p.strip()]
        if not seq_dirs:
            raise ValueError("--sequence_dirs must be provided when using --run_phylophlan")
        files = _discover_sequences(seq_dirs)
        if not files:
            raise RuntimeError("No FASTA/FAA files found in provided --sequence_dirs")

        # Prepare inputs under a dedicated input directory within the workdir
        work_dir = Path(args.phylophlan_out_dir)
        input_dir = work_dir / "input"
        _prepare_input_dir(files, input_dir)

        # Determine sequence type if not provided
        seq_type = str(args.seq_type) if args.seq_type else ("a" if _looks_like_proteins(files) else "n")

        # Run PhyloPhlAn
        tree_path = _run_phylophlan(
            input_dir=input_dir,
            work_dir=work_dir,
            threads=int(args.threads),
            config_path=args.config,
            configs_folder=args.configs_folder,
            database=str(args.database),
            extra_args=args.phylophlan_args or None,
            diversity=str(args.diversity),
            seq_type=str(seq_type),
        )
    else:
        raise ValueError("Provide either --tree_file or --run_phylophlan with --sequence_dirs.")

    out_png = Path(args.out_png)
    _plot_circular(newick_path=tree_path, out_png=out_png, show_tip_labels=bool(args.show_tip_labels), width_px=int(args.image_width_px))


if __name__ == "__main__":
    main()


