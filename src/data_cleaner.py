"""
Data Cleaning Module for Bangalore Real Estate Prediction
==========================================================
Handles cleaning total_sqft, extracting BHK, normalizing locations,
and handling missing values.
"""

import pandas as pd
import numpy as np
import re


def clean_total_sqft(sqft_value) -> float:
    """
    Clean total_sqft values handling ranges and unit conversions.
    
    Examples:
        "1056" -> 1056.0
        "2100 - 2850" -> 2475.0 (average)
        "34.46Sq. Meter" -> 370.93 (convert to sqft)
        "1Acres" -> 43560.0
    """
    if pd.isna(sqft_value):
        return np.nan
    
    sqft_str = str(sqft_value).strip()
    
    # Handle ranges (e.g., "2100 - 2850")
    if '-' in sqft_str and 'Sq' not in sqft_str:
        try:
            parts = sqft_str.split('-')
            low = float(parts[0].strip())
            high = float(parts[1].strip())
            return (low + high) / 2
        except:
            pass
    
    # Handle Sq. Meter conversion
    if 'Sq. Meter' in sqft_str:
        try:
            value = float(re.search(r'[\d.]+', sqft_str).group())
            return value * 10.764  # 1 sq meter = 10.764 sq ft
        except:
            return np.nan
    
    # Handle Acres conversion
    if 'Acres' in sqft_str:
        try:
            value = float(re.search(r'[\d.]+', sqft_str).group())
            return value * 43560  # 1 acre = 43560 sq ft
        except:
            return np.nan
    
    # Handle Perch conversion (Sri Lankan unit sometimes used)
    if 'Perch' in sqft_str:
        try:
            value = float(re.search(r'[\d.]+', sqft_str).group())
            return value * 272.25  # 1 perch ≈ 272.25 sq ft
        except:
            return np.nan
    
    # Handle Guntha conversion
    if 'Guntha' in sqft_str:
        try:
            value = float(re.search(r'[\d.]+', sqft_str).group())
            return value * 1089  # 1 guntha = 1089 sq ft
        except:
            return np.nan
    
    # Handle Grounds conversion
    if 'Grounds' in sqft_str:
        try:
            value = float(re.search(r'[\d.]+', sqft_str).group())
            return value * 2400  # 1 ground = 2400 sq ft
        except:
            return np.nan
    
    # Try direct numeric conversion
    try:
        return float(sqft_str)
    except:
        return np.nan


def extract_bhk(size_value) -> int:
    """
    Extract BHK number from size column.
    
    Examples:
        "3 BHK" -> 3
        "4 Bedroom" -> 4
        "1 RK" -> 1
    """
    if pd.isna(size_value):
        return np.nan
    
    size_str = str(size_value).strip()
    
    try:
        # Extract the first number
        match = re.search(r'(\d+)', size_str)
        if match:
            return int(match.group(1))
    except:
        pass
    
    return np.nan


def normalize_location(location) -> str:
    """
    Normalize location names for consistency.
    """
    if pd.isna(location):
        return 'Unknown'
    
    loc = str(location).strip()
    # Remove extra whitespace
    loc = ' '.join(loc.split())
    # Title case
    loc = loc.title()
    
    return loc


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the entire dataset.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    print("Cleaning data...")
    
    # 1. Clean total_sqft
    print("  ✓ Cleaning total_sqft values...")
    df['total_sqft_clean'] = df['total_sqft'].apply(clean_total_sqft)
    
    # 2. Extract BHK
    print("  ✓ Extracting BHK from size column...")
    df['bhk'] = df['size'].apply(extract_bhk)
    
    # 3. Normalize locations
    print("  ✓ Normalizing location names...")
    df['location_clean'] = df['location'].apply(normalize_location)
    
    # 4. Handle missing values in bath and balcony
    print("  ✓ Handling missing values...")
    df['bath'] = df['bath'].fillna(df['bath'].median())
    df['balcony'] = df['balcony'].fillna(df['balcony'].median())
    
    # 5. Remove rows with missing critical values
    before_drop = len(df)
    df = df.dropna(subset=['total_sqft_clean', 'bhk', 'price'])
    after_drop = len(df)
    print(f"  ✓ Removed {before_drop - after_drop} rows with missing critical values")
    
    # 6. Filter unrealistic values
    # BHK should be between 1 and 16
    df = df[(df['bhk'] >= 1) & (df['bhk'] <= 16)]
    
    # Total sqft should be reasonable (100 to 100000)
    df = df[(df['total_sqft_clean'] >= 100) & (df['total_sqft_clean'] <= 100000)]
    
    # Bath should be reasonable
    df = df[(df['bath'] >= 1) & (df['bath'] <= 16)]
    
    print(f"  ✓ Final dataset: {len(df):,} records")
    
    return df


if __name__ == "__main__":
    from data_loader import load_raw_data
    
    df = load_raw_data()
    df_clean = clean_data(df)
    print(f"\nCleaned data shape: {df_clean.shape}")
