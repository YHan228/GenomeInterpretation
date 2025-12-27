#!/usr/bin/env python3
"""
Run STREME discriminative motif discovery on sporulating vs non-sporulating genomes.
"""

import argparse
import subprocess
import random
from pathlib import Path
import pandas as pd

DATA_ROOT = Path("/vol/projects/BIFO/genomenet/yichen/phenotype/data")
METADATA_XLSX = Path("sporulation/microbe.cards table S1.xlsx")


def load_sequences_by_class(phenotype: str, max_per_class: int = 200):
    """Load sequences and separate by class."""
    # Read metadata
    df = pd.read_excel(METADATA_XLSX)

    # Build labels map
    labels_map = {}
    for _, row in df.iterrows():
        fname = str(row.get("Fasta file", "")).strip().lower()
        label_val = str(row.get(phenotype, "")).strip().lower()
        if fname and label_val in ("true", "false"):
            labels_map[fname] = 1 if label_val == "true" else 0

    positive_seqs = []
    negative_seqs = []

    train_dir = DATA_ROOT / "train"
    for fasta_file in train_dir.glob("*.fasta"):
        fname = fasta_file.name.strip().lower()
        label = labels_map.get(fname)
        if label is None:
            continue

        # Read sequence
        with open(fasta_file) as f:
            lines = f.readlines()
        seq = "".join(line.strip() for line in lines if not line.startswith(">"))

        if label == 1:
            positive_seqs.append((fasta_file.stem, seq))
        else:
            negative_seqs.append((fasta_file.stem, seq))

    # Subsample
    if len(positive_seqs) > max_per_class:
        positive_seqs = random.sample(positive_seqs, max_per_class)
    if len(negative_seqs) > max_per_class:
        negative_seqs = random.sample(negative_seqs, max_per_class)

    print(f"Loaded {len(positive_seqs)} positive, {len(negative_seqs)} negative genomes")
    return positive_seqs, negative_seqs


def extract_snippets(sequences, n_snippets: int = 10, snippet_len: int = 500):
    """Extract random snippets from sequences."""
    snippets = []
    for name, seq in sequences:
        seq_len = len(seq)
        if seq_len < snippet_len:
            continue
        for i in range(n_snippets):
            start = random.randint(0, seq_len - snippet_len)
            snippet = seq[start:start + snippet_len].upper()
            if snippet.count('N') / len(snippet) > 0.1:
                continue
            snippets.append((f"{name}_snip{i}", snippet))
    return snippets


def write_fasta(snippets, output_path: Path):
    """Write snippets to FASTA file."""
    with open(output_path, 'w') as f:
        for name, seq in snippets:
            f.write(f">{name}\n{seq}\n")
    print(f"Wrote {len(snippets)} sequences to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phenotype", default="Spore formation")
    parser.add_argument("--n_snippets", type=int, default=10)
    parser.add_argument("--snippet_len", type=int, default=500)
    parser.add_argument("--max_genomes", type=int, default=200)
    parser.add_argument("--minw", type=int, default=4)
    parser.add_argument("--maxw", type=int, default=8)
    parser.add_argument("--output_dir", default="sporulation/reports/streme")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prep_only", action="store_true", help="Only prepare FASTA, don't run STREME")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and extract
    print(f"Loading sequences for '{args.phenotype}'...")
    pos_seqs, neg_seqs = load_sequences_by_class(args.phenotype, args.max_genomes)

    print(f"Extracting snippets...")
    pos_snippets = extract_snippets(pos_seqs, args.n_snippets, args.snippet_len)
    neg_snippets = extract_snippets(neg_seqs, args.n_snippets, args.snippet_len)
    print(f"Total: {len(pos_snippets)} positive, {len(neg_snippets)} negative snippets")

    # Write FASTA
    pos_fasta = output_dir / "positive.fasta"
    neg_fasta = output_dir / "negative.fasta"
    write_fasta(pos_snippets, pos_fasta)
    write_fasta(neg_snippets, neg_fasta)

    if args.prep_only:
        print(f"\nFASTA files ready. Run STREME with:")
        print(f"  streme --p {pos_fasta} --n {neg_fasta} --oc {output_dir} --dna --minw {args.minw} --maxw {args.maxw}")
        return

    # Run STREME
    cmd = ["streme", "--p", str(pos_fasta), "--n", str(neg_fasta),
           "--oc", str(output_dir), "--dna",
           "--minw", str(args.minw), "--maxw", str(args.maxw), "--nmotifs", "20"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
    print(f"\nResults: {output_dir}/streme.html")


if __name__ == "__main__":
    main()
