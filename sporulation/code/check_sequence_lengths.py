import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

BASE_DIR = 'sporulation/data'
SEQ_LEN_THRESHOLD = 1_000_000

def parse_fasta_length(file_path):
    """A simple FASTA parser that returns the sequence length."""
    length = 0
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    continue
                length += len(line.strip())
        return file_path, length
    except Exception as e:
        return file_path, f"Error: {e}"

def analyze_directory(data_dir):
    """Analyzes all FASTA files in a given directory in parallel."""
    files_to_process = []
    if not os.path.isdir(data_dir):
        print(f"Directory not found: {data_dir}")
        return []

    for file_name in os.listdir(data_dir):
        if file_name.endswith(('.fasta', '.fa', '.fna')):
            files_to_process.append(os.path.join(data_dir, file_name))
            
    lengths = []
    with ProcessPoolExecutor() as executor:
        results = executor.map(parse_fasta_length, files_to_process)
        for _, length in results:
            if isinstance(length, int):
                lengths.append(length)
    return lengths

def main():
    """Main function to analyze all data splits."""
    all_lengths = []
    
    for split in ['train', 'validation', 'test']:
        print(f"--- Analyzing {split} set ---")
        data_dir = os.path.join(BASE_DIR, split)
        lengths = analyze_directory(data_dir)
        
        if not lengths:
            print(f"No FASTA files found or processed in {data_dir}")
            continue
            
        all_lengths.extend(lengths)
        
        df_split = pd.DataFrame(lengths, columns=['length'])
        print(df_split.describe())
        
        over_threshold = sum(1 for l in lengths if l > SEQ_LEN_THRESHOLD)
        print(f"Genomes over {SEQ_LEN_THRESHOLD} bp: {over_threshold} / {len(lengths)} ({over_threshold/len(lengths):.2%})")
        print("-" * 25)

    print("\n--- Overall Summary ---")
    if not all_lengths:
        print("No data to summarize.")
        return
        
    df_all = pd.DataFrame(all_lengths, columns=['length'])
    print(df_all.describe())
    
    total_over_threshold = sum(1 for l in all_lengths if l > SEQ_LEN_THRESHOLD)
    print(f"Total genomes over {SEQ_LEN_THRESHOLD} bp: {total_over_threshold} / {len(all_lengths)} ({total_over_threshold/len(all_lengths):.2%})")

if __name__ == "__main__":
    main()
