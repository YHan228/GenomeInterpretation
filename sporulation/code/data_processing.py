import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil
import numpy as np

# --- Configuration ---
TAXONOMY_PATH = 'sporulation/sporulation.xlsx'
BASE_DATA_DIR = '/vol/projects/BIFO/genomenet/yichen/phenotype/data'
FASTA_DIR = os.path.join(BASE_DATA_DIR, 'fasta_combined')
TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DATA_DIR, 'validation')
TEST_DIR = os.path.join(BASE_DATA_DIR, 'test')
UNUSED_DIR = os.path.join(BASE_DATA_DIR, 'unused_fasta')
SUMMARY_CSV_PATH = os.path.join(BASE_DATA_DIR, 'data_split_summary.csv')

STRATIFY_COLUMN = 'Family'
MIN_SAMPLES_PER_GROUP = 10
RANDOM_STATE = 42

def consolidate_files():
    """Consolidates all FASTA files from split directories back into the combined directory."""
    print("Consolidating files...")
    os.makedirs(FASTA_DIR, exist_ok=True)
    for directory in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:
        if not os.path.isdir(directory):
            continue
        for file_name in os.listdir(directory):
            if file_name.endswith(('.fasta', '.fa', '.fna')):
                try:
                    shutil.move(os.path.join(directory, file_name), FASTA_DIR)
                except Exception as e:
                    print(f"Could not move {file_name}: {e}")
    print("Consolidation complete.")

def create_stratification_group(df, column, min_samples):
    """Creates a new column for stratification, grouping rare categories into 'other'."""
    value_counts = df[column].value_counts()
    rare_groups = value_counts[value_counts < min_samples].index
    
    df['stratify_group'] = df[column].apply(lambda x: 'other' if x in rare_groups else x)
    df['stratify_key'] = df['stratify_group'].astype(str) + '_' + df['Spore formation'].astype(str)
    
    return df

def move_and_report_files(file_list, destination_folder):
    """Moves files and returns a list of moved files."""
    os.makedirs(destination_folder, exist_ok=True)
    moved_files = []
    for file_name in file_list:
        source_path = os.path.join(FASTA_DIR, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        if os.path.exists(source_path):
            shutil.move(source_path, destination_path)
            moved_files.append(file_name)
        else:
            print(f"Warning: File {file_name} not found in {FASTA_DIR} and will be skipped.")
    return moved_files

def generate_summary_report(df, train_files, val_files, test_files):
    """Generates and saves a CSV report of the data split distribution."""
    df['split'] = 'unassigned'
    df.loc[df['file'].isin(train_files), 'split'] = 'train'
    df.loc[df['file'].isin(val_files), 'split'] = 'validation'
    df.loc[df['file'].isin(test_files), 'split'] = 'test'
    
    summary = df.groupby(['stratify_group', 'split']).size().unstack(fill_value=0)
    summary.to_csv(SUMMARY_CSV_PATH)
    print(f"\nGenerated data split summary at: {SUMMARY_CSV_PATH}")
    print(summary)

def main():
    """Main function to perform taxon-stratified data splitting."""
    consolidate_files()

    try:
        df_tax = pd.read_excel(TAXONOMY_PATH)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    df_tax.rename(columns={'Fasta file': 'file'}, inplace=True)
    df_tax.dropna(subset=['file', 'Spore formation', STRATIFY_COLUMN], inplace=True)

    df_processed = create_stratification_group(df_tax, STRATIFY_COLUMN, MIN_SAMPLES_PER_GROUP)
    
    key_counts = df_processed['stratify_key'].value_counts()
    single_member_keys = key_counts[key_counts == 1].index
    df_processed.loc[df_processed['stratify_key'].isin(single_member_keys), 'stratify_key'] = 'other_combined'

    X = df_processed['file']
    y = df_processed['stratify_key']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    temp_df = pd.DataFrame({'file': X_temp, 'stratify_key': y_temp})
    key_counts_temp = temp_df['stratify_key'].value_counts()
    single_member_keys_temp = key_counts_temp[key_counts_temp == 1].index
    temp_df.loc[temp_df['stratify_key'].isin(single_member_keys_temp), 'stratify_key'] = 'other_combined'

    X_val, X_test, _, _ = train_test_split(
        temp_df['file'], temp_df['stratify_key'], test_size=0.5, random_state=RANDOM_STATE, stratify=temp_df['stratify_key']
    )

    print("\nMoving files to respective directories...")
    train_files = move_and_report_files(X_train.tolist(), TRAIN_DIR)
    val_files = move_and_report_files(X_val.tolist(), VALIDATION_DIR)
    test_files = move_and_report_files(X_test.tolist(), TEST_DIR)
    
    print("\nData splitting and file moving complete.")
    generate_summary_report(df_processed, train_files, val_files, test_files)

    # Handle leftover files
    remaining_files = [f for f in os.listdir(FASTA_DIR) if f.endswith(('.fasta', '.fa', '.fna'))]
    if remaining_files:
        print(f"\nWarning: {len(remaining_files)} files were not in the metadata and were not split.")
        os.makedirs(UNUSED_DIR, exist_ok=True)
        for f in remaining_files:
            shutil.move(os.path.join(FASTA_DIR, f), UNUSED_DIR)
        print(f"These files have been moved to: {UNUSED_DIR}")

if __name__ == "__main__":
    main()
