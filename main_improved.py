"""
Improved Training Pipeline
==========================
Uses advanced features and ensemble methods for maximum accuracy.

Usage:
    python main_improved.py
"""

import os
import sys
import warnings
import numpy as np
import torch
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader import load_raw_data, get_data_summary
from src.data_cleaner import clean_data
from src.feature_engineering import create_features
from src.advanced_features import create_advanced_features, prepare_advanced_features
from src.ensemble_model import train_ensemble_pipeline
from src.graph_builder import create_graph_data, create_train_val_test_masks
from src.evaluate import calculate_metrics, plot_model_comparison


def main():
    """
    Main improved training pipeline.
    """
    print("\n" + "="*70)
    print("  IMPROVED BANGALORE REAL ESTATE PRICE PREDICTION")
    print("  With Advanced Features & Ensemble Methods")
    print("="*70)
    
    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    print("\n📂 STEP 1: Loading data...")
    
    data_path = Path(__file__).parent / 'data' / 'Bengaluru_House_Data.csv'
    df = load_raw_data(str(data_path))
    _ = get_data_summary(df)
    
    # =========================================================================
    # STEP 2: CLEAN DATA
    # =========================================================================
    print("\n🧹 STEP 2: Cleaning data...")
    
    df_clean = clean_data(df)
    
    # =========================================================================
    # STEP 3: BASIC FEATURE ENGINEERING
    # =========================================================================
    print("\n⚙️ STEP 3: Basic feature engineering...")
    
    df_features = create_features(df_clean)
    
    # =========================================================================
    # STEP 4: ADVANCED FEATURE ENGINEERING
    # =========================================================================
    print("\n🚀 STEP 4: Advanced feature engineering...")
    
    df_advanced = create_advanced_features(df_features)
    
    # =========================================================================
    # STEP 5: PREPARE MODEL FEATURES
    # =========================================================================
    print("\n🔧 STEP 5: Preparing advanced features...")
    
    X, y, feature_names, le_location, scaler_X, scaler_y, df_final = prepare_advanced_features(df_advanced)
    
    # Create train/val/test split
    train_mask, val_mask, test_mask = create_train_val_test_masks(len(X))
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"\n  Data splits:")
    print(f"    Train: {len(X_train):,}")
    print(f"    Val:   {len(X_val):,}")
    print(f"    Test:  {len(X_test):,}")
    print(f"    Features: {len(feature_names)}")
    
    # =========================================================================
    # STEP 6: TRAIN ENSEMBLE MODELS
    # =========================================================================
    print("\n📊 STEP 6: Training ensemble models with hyperparameter tuning...")
    
    checkpoints_dir = Path(__file__).parent / 'checkpoints'
    
    results, base_models, stacking = train_ensemble_pipeline(
        X_train, y_train, 
        X_val, y_val, 
        X_test, y_test,
        save_dir=str(checkpoints_dir)
    )
    
    # =========================================================================
    # STEP 7: SAVE ARTIFACTS
    # =========================================================================
    print("\n💾 STEP 7: Saving artifacts...")
    
    joblib.dump(scaler_X, checkpoints_dir / 'scaler_X_advanced.joblib')
    joblib.dump(scaler_y, checkpoints_dir / 'scaler_y_advanced.joblib')
    joblib.dump(le_location, checkpoints_dir / 'le_location_advanced.joblib')
    
    import json
    with open(checkpoints_dir / 'feature_names_advanced.json', 'w') as f:
        json.dump(feature_names, f)
    
    # =========================================================================
    # STEP 8: GENERATE COMPARISON PLOTS
    # =========================================================================
    print("\n📈 STEP 8: Generating comparison plots...")
    
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Prepare results for plotting
    plot_results = {}
    for name, data in results.items():
        plot_results[name.replace('_', ' ').title()] = {
            'r2': data['r2'],
            'mae': data['mae'],
            'rmse': np.sqrt(data['mae'])  # Approximation
        }
    
    plot_model_comparison(plot_results, str(results_dir / 'improved_model_comparison.png'))
    
    # Save metrics
    with open(results_dir / 'improved_metrics.json', 'w') as f:
        json.dump({k: {'r2': v['r2'], 'mae': v['mae']} for k, v in results.items()}, f, indent=2)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("  IMPROVED TRAINING COMPLETE - FINAL SUMMARY")
    print("="*70)
    print(f"\n  Dataset: {len(df_final):,} properties")
    print(f"  Features: {len(feature_names)} (advanced)")
    print(f"\n  Test Set Performance:")
    print(f"    {'Model':<25} {'R² Score':<12} {'Accuracy %':<12}")
    print(f"    {'-'*49}")
    
    for model_name, metrics in sorted(results.items(), key=lambda x: -x[1]['r2']):
        accuracy_pct = metrics['r2'] * 100
        print(f"    {model_name:<25} {metrics['r2']:.4f}       {accuracy_pct:.1f}%")
    
    # Best model
    best_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_r2 = results[best_name]['r2']
    
    print(f"\n  🏆 BEST: {best_name.upper()} with {best_r2*100:.1f}% accuracy!")
    print("="*70 + "\n")
    
    return results, base_models, stacking


if __name__ == "__main__":
    results, models, ensemble = main()
