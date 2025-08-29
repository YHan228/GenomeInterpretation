import pandas as pd

def analyze_taxonomy(file_path):
    """Analyzes and prints the distribution of taxonomic ranks."""
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    taxonomic_ranks = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    
    for rank in taxonomic_ranks:
        if rank in df.columns:
            print(f"--- Distribution for {rank} ---")
            print(df[rank].value_counts())
            print(f"Number of unique {rank.lower()}s: {df[rank].nunique()}")
            print("-" * 30 + "\n")
        else:
            print(f"Column '{rank}' not found in the Excel file.")

if __name__ == "__main__":
    analyze_taxonomy('sporulation/sporulation.xlsx')
