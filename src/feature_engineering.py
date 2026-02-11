"""
Feature Engineering Module for Bangalore Real Estate Prediction
================================================================
Calculates price_per_sqft, removes outliers, and creates derived features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def calculate_price_per_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price per square foot (target variable).
    
    Price is in Lakhs, so multiply by 100000 to get rupees.
    """
    df = df.copy()
    df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft_clean']
    return df


def remove_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers using IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    before = len(df)
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    after = len(df)
    
    print(f"  ✓ Removed {before - after} outliers from {column}")
    print(f"    Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    return df


def remove_bhk_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove properties where sqft per BHK is unrealistic.
    
    A typical BHK should have at least 300 sqft.
    """
    df = df.copy()
    df['sqft_per_bhk'] = df['total_sqft_clean'] / df['bhk']
    
    before = len(df)
    # Minimum 300 sqft per BHK is reasonable
    df = df[df['sqft_per_bhk'] >= 300]
    after = len(df)
    
    print(f"  ✓ Removed {before - after} properties with unrealistic sqft/BHK ratio")
    
    return df


def encode_locations(df: pd.DataFrame, min_count: int = 10) -> pd.DataFrame:
    """
    Encode location column using frequency-based approach.
    
    Locations with fewer than min_count properties are grouped as 'Other'.
    """
    df = df.copy()
    
    # Get location counts
    location_counts = df['location_clean'].value_counts()
    
    # Locations with fewer occurrences become 'Other'
    locations_less_than_threshold = location_counts[location_counts < min_count].index.tolist()
    df['location_encoded'] = df['location_clean'].apply(
        lambda x: 'Other' if x in locations_less_than_threshold else x
    )
    
    unique_locs = df['location_encoded'].nunique()
    print(f"  ✓ Encoded locations: {unique_locs} unique values")
    
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all derived features for modeling.
    """
    df = df.copy()
    print("\nEngineering features...")
    
    # 1. Calculate price per sqft
    print("  ✓ Calculating price_per_sqft...")
    df = calculate_price_per_sqft(df)
    
    # 2. Remove outliers by location
    print("  ✓ Removing location-based outliers...")
    df = remove_location_outliers(df)
    
    # 3. Remove BHK-based outliers
    df = remove_bhk_outliers(df)
    
    # 4. Remove overall price_per_sqft outliers
    df = remove_outliers_iqr(df, 'price_per_sqft', factor=1.5)
    
    # 5. Encode locations
    df = encode_locations(df, min_count=10)
    
    # 6. Create additional features
    print("  ✓ Creating derived features...")
    df['bath_per_bhk'] = df['bath'] / df['bhk']
    
    # Encode area_type
    area_type_map = {
        'Super built-up  Area': 1,
        'Built-up  Area': 2,
        'Plot  Area': 3,
        'Carpet  Area': 4
    }
    df['area_type_encoded'] = df['area_type'].map(area_type_map).fillna(1)
    
    print(f"\n  Final dataset: {len(df):,} records")
    print(f"  Features: bhk, bath, balcony, total_sqft_clean, area_type_encoded, location_encoded")
    
    return df


def remove_location_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove price per sqft outliers within each location.
    """
    df = df.copy()
    
    # Temporarily calculate price_per_sqft if not exists
    if 'price_per_sqft' not in df.columns:
        df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft_clean']
    
    # Remove outliers per location
    def remove_outliers_group(group):
        mean = group['price_per_sqft'].mean()
        std = group['price_per_sqft'].std()
        if std == 0 or pd.isna(std):
            return group
        return group[(group['price_per_sqft'] > (mean - std)) & 
                     (group['price_per_sqft'] < (mean + std))]
    
    before = len(df)
    df = df.groupby('location_clean', group_keys=False).apply(remove_outliers_group)
    after = len(df)
    
    print(f"  ✓ Removed {before - after} location-based outliers")
    
    return df


def prepare_model_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features and target for ML modeling.
    
    Returns:
        Tuple of (X, y, feature_names, label_encoders, scalers)
    """
    df = df.copy()
    
    # Define feature columns
    numerical_features = ['total_sqft_clean', 'bhk', 'bath', 'balcony', 'area_type_encoded']
    
    # Encode location
    le_location = LabelEncoder()
    df['location_label'] = le_location.fit_transform(df['location_encoded'])
    
    # Prepare feature matrix
    feature_cols = numerical_features + ['location_label']
    X = df[feature_cols].values
    y = df['price_per_sqft'].values
    
    # Scale features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    print(f"\n  Feature matrix shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    
    return X_scaled, y_scaled, feature_cols, le_location, scaler_X, scaler_y, df


if __name__ == "__main__":
    from data_loader import load_raw_data
    from data_cleaner import clean_data
    
    df = load_raw_data()
    df = clean_data(df)
    df = create_features(df)
    
    X, y, features, le, sx, sy, df_final = prepare_model_features(df)
    print(f"\nReady for modeling!")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
