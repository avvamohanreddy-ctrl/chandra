"""
Data Loading Module for Bangalore Real Estate Prediction
=========================================================
Handles loading and initial parsing of Bengaluru House Data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the Bengaluru House Data CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Raw DataFrame with housing data
    """
    if filepath is None:
        filepath = Path(__file__).parent.parent / 'data' / 'Bengaluru_House_Data.csv'
    
    df = pd.read_csv(filepath)
    
    print(f"✓ Loaded {len(df):,} property records")
    print(f"  Columns: {list(df.columns)}")
    
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the dataset.
    
    Args:
        df: Raw housing DataFrame
        
    Returns:
        Dictionary with dataset statistics
    """
    summary = {
        'total_records': len(df),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'area_types': df['area_type'].value_counts().to_dict(),
        'unique_locations': df['location'].nunique(),
        'price_range': {
            'min': df['price'].min(),
            'max': df['price'].max(),
            'mean': df['price'].mean()
        }
    }
    
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Total Records: {summary['total_records']:,}")
    print(f"Unique Locations: {summary['unique_locations']}")
    print(f"\nMissing Values:")
    for col, count in summary['missing_values'].items():
        if count > 0:
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    print(f"\nArea Types:")
    for area_type, count in summary['area_types'].items():
        print(f"  {area_type}: {count}")
    print(f"\nPrice Range (in Lakhs):")
    print(f"  Min: ₹{summary['price_range']['min']:.2f}L")
    print(f"  Max: ₹{summary['price_range']['max']:.2f}L")
    print(f"  Mean: ₹{summary['price_range']['mean']:.2f}L")
    print("="*50 + "\n")
    
    return summary


if __name__ == "__main__":
    df = load_raw_data()
    summary = get_data_summary(df)
