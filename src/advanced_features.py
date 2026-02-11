"""
Advanced Feature Engineering Module
====================================
Creates interaction features, polynomial features, and location statistics
to improve model accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced derived features for better predictive accuracy.
    
    Adds:
    - Interaction features (BHK × sqft, etc.)
    - Ratio features
    - Location-based statistics
    - Polynomial features
    """
    df = df.copy()
    print("\n🔧 Creating advanced features...")
    
    # =========================================================================
    # 1. INTERACTION FEATURES
    # =========================================================================
    print("  ✓ Creating interaction features...")
    
    # BHK interactions
    df['bhk_sqft_interaction'] = df['bhk'] * df['total_sqft_clean']
    df['bhk_bath_interaction'] = df['bhk'] * df['bath']
    
    # Room density (how much space per room)
    df['sqft_per_room'] = df['total_sqft_clean'] / (df['bhk'] + df['bath'] + df['balcony'] + 1)
    
    # =========================================================================
    # 2. RATIO FEATURES
    # =========================================================================
    print("  ✓ Creating ratio features...")
    
    # Bath to BHK ratio (luxury indicator)
    df['bath_bhk_ratio'] = df['bath'] / df['bhk']
    
    # Balcony per BHK
    df['balcony_bhk_ratio'] = df['balcony'] / df['bhk']
    
    # Sqft per BHK (spaciousness)
    df['sqft_per_bhk'] = df['total_sqft_clean'] / df['bhk']
    
    # =========================================================================
    # 3. BINNED/CATEGORICAL FEATURES
    # =========================================================================
    print("  ✓ Creating binned features...")
    
    # Sqft bins (studio, small, medium, large, luxury)
    df['sqft_bin'] = pd.cut(df['total_sqft_clean'], 
                            bins=[0, 500, 1000, 1500, 2500, float('inf')],
                            labels=[0, 1, 2, 3, 4]).astype(float)
    
    # BHK category (compact vs spacious)
    df['bhk_category'] = np.where(df['bhk'] <= 2, 0, 
                                  np.where(df['bhk'] <= 4, 1, 2))
    
    # =========================================================================
    # 4. LOCATION-BASED STATISTICS
    # =========================================================================
    print("  ✓ Computing location statistics...")
    
    # Mean price per sqft by location
    location_stats = df.groupby('location_encoded').agg({
        'price_per_sqft': ['mean', 'std', 'median', 'count']
    }).reset_index()
    location_stats.columns = ['location_encoded', 'loc_price_mean', 'loc_price_std', 
                              'loc_price_median', 'loc_count']
    
    # Fill NaN std with 0 (single-property locations)
    location_stats['loc_price_std'] = location_stats['loc_price_std'].fillna(0)
    
    # Merge back
    df = df.merge(location_stats, on='location_encoded', how='left')
    
    # Location price tier (percentile-based)
    df['loc_price_tier'] = pd.qcut(df['loc_price_mean'], q=5, labels=[0, 1, 2, 3, 4]).astype(float)
    
    # =========================================================================
    # 5. AREA TYPE ENHANCED FEATURES
    # =========================================================================
    print("  ✓ Creating area type features...")
    
    # One-hot encode area types
    area_dummies = pd.get_dummies(df['area_type'], prefix='area')
    for col in area_dummies.columns:
        df[col] = area_dummies[col].astype(float)
    
    # =========================================================================
    # 6. RELATIVE FEATURES
    # =========================================================================
    print("  ✓ Computing relative features...")
    
    # Price deviation from location mean
    df['price_vs_loc_mean'] = df['price_per_sqft'] / (df['loc_price_mean'] + 1)
    
    # Property quality score (composite)
    df['quality_score'] = (
        (df['bath'] / df['bhk']) * 0.3 +
        (df['balcony'] / df['bhk']) * 0.2 +
        (df['sqft_per_bhk'] / 500) * 0.5
    ).clip(0, 3)
    
    print(f"  ✓ Total features created: {len(df.columns)}")
    
    return df


def prepare_advanced_features(df: pd.DataFrame) -> tuple:
    """
    Prepare the full feature matrix with all advanced features.
    
    Returns:
        Tuple of (X, y, feature_names, label_encoders, scalers, df)
    """
    df = df.copy()
    
    # Encode location as label
    le_location = LabelEncoder()
    df['location_label'] = le_location.fit_transform(df['location_encoded'])
    
    # Define all feature columns (expanded list) - only numeric!
    feature_cols = [
        # Original numeric features
        'total_sqft_clean', 'bhk', 'bath', 'balcony', 'area_type_encoded', 'location_label',
        # Interaction features
        'bhk_sqft_interaction', 'bhk_bath_interaction', 'sqft_per_room',
        # Ratio features
        'bath_bhk_ratio', 'balcony_bhk_ratio', 'sqft_per_bhk',
        # Binned features
        'sqft_bin', 'bhk_category',
        # Location statistics
        'loc_price_mean', 'loc_price_std', 'loc_price_median', 'loc_count', 'loc_price_tier',
        # Quality
        'quality_score'
    ]
    
    # Add area type dummies if they exist (these are numeric 0/1)
    area_cols = [col for col in df.columns if col.startswith('area_') and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
    feature_cols.extend(area_cols)
    
    # Filter to only existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Prepare matrices
    X = df[feature_cols].values
    y = df['price_per_sqft'].values
    
    # Handle any NaN values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    print(f"\n📊 Advanced Feature Summary:")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    
    return X_scaled, y_scaled, feature_cols, le_location, scaler_X, scaler_y, df


if __name__ == "__main__":
    from data_loader import load_raw_data
    from data_cleaner import clean_data
    from feature_engineering import create_features
    
    # Load and process
    df = load_raw_data()
    df = clean_data(df)
    df = create_features(df)
    df = create_advanced_features(df)
    
    X, y, features, le, sx, sy, df_final = prepare_advanced_features(df)
    
    print(f"\nAdvanced features ready!")
    print(f"  Features: {features}")
