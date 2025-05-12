"""
This script helps you load and explore your dataset in VS Code.
Run this script to verify that your dataset can be loaded correctly.
"""

import pandas as pd
import os
import sys

def load_dataset(file_path):
    """
    Load a dataset from various file formats.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        pandas DataFrame containing the dataset
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
        elif file_extension == '.pkl':
            df = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        
        print(f"Successfully loaded dataset from {file_path}")
        print(f"Dataset shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nDataset information:")
        print(df.info())
        
        print("\nSummary statistics:")
        print(df.describe())
        
        return df
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        df = load_dataset(file_path)
    else:
        print("Usage: python load_dataset.py <path_to_dataset_file>")
        print("\nSupported file formats: .csv, .xlsx, .xls, .json, .pkl")
