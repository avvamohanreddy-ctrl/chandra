"""
Main Pipeline for Bangalore Real Estate Price Prediction
=========================================================
Orchestrates the complete pipeline from data loading to model evaluation.

Usage:
    python main.py                  # Run full pipeline
    python main.py --quick-test     # Quick test with subset
"""

import os
import sys
import argparse
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
from src.feature_engineering import create_features, prepare_model_features
from src.geocoder import add_coordinates
from src.graph_builder import create_graph_data, create_train_val_test_masks
from src.models.gnn_model import BangaloreGAT, count_parameters
from src.models.baseline_models import train_baseline_models
from src.train import train
from src.evaluate import generate_all_plots, calculate_metrics, plot_model_comparison


def main(quick_test: bool = False):
    """
    Main pipeline function.
    
    Args:
        quick_test: If True, use subset of data for quick testing
    """
    print("\n" + "="*70)
    print("  BANGALORE REAL ESTATE PRICE PREDICTION SYSTEM")
    print("  Using Graph Attention Networks (GAT)")
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
    # STEP 3: FEATURE ENGINEERING
    # =========================================================================
    print("\n⚙️ STEP 3: Feature engineering...")
    
    df_features = create_features(df_clean)
    
    # =========================================================================
    # STEP 4: GEOCODING
    # =========================================================================
    print("\n📍 STEP 4: Geocoding locations...")
    
    df_geo = add_coordinates(df_features, 'location_clean')
    
    # Quick test: use subset
    if quick_test:
        print("\n⚡ Quick test mode: using 1000 samples...")
        df_geo = df_geo.sample(n=min(1000, len(df_geo)), random_state=RANDOM_SEED)
    
    # =========================================================================
    # STEP 5: PREPARE ML FEATURES
    # =========================================================================
    print("\n🔧 STEP 5: Preparing features for modeling...")
    
    X, y, feature_names, le_location, scaler_X, scaler_y, df_final = prepare_model_features(df_geo)
    
    # Get coordinates
    coordinates = df_geo[['latitude', 'longitude']].values
    
    # Create train/val/test masks
    train_mask, val_mask, test_mask = create_train_val_test_masks(len(X))
    
    # =========================================================================
    # STEP 6: TRAIN BASELINE MODELS
    # =========================================================================
    print("\n📊 STEP 6: Training baseline ML models...")
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    baseline_results, baseline_models = train_baseline_models(
        X_train, y_train, X_val, y_val,
        save_dir=str(Path(__file__).parent / 'checkpoints')
    )
    
    # Evaluate baselines on test set
    print("\n📈 Baseline models on test set:")
    for name, model in baseline_models.items():
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        baseline_results[name.replace('_', ' ').title()] = metrics
        print(f"  {name}: R² = {metrics['r2']:.4f}, MAE = {metrics['mae']:.4f}")
    
    # =========================================================================
    # STEP 7: BUILD GRAPH AND TRAIN GNN
    # =========================================================================
    print("\n🕸️ STEP 7: Building graph and training GNN...")
    
    # Create graph data
    data = create_graph_data(
        features=X,
        target=y,
        coordinates=coordinates,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        k=10  # k-nearest neighbors
    )
    
    # Initialize GAT model
    model = BangaloreGAT(
        in_channels=data.num_node_features,
        hidden_channels=64,
        out_channels=1,
        heads=4,
        dropout=0.3
    )
    
    print(f"\nModel: BangaloreGAT")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Train
    history = train(
        model=model,
        data=data,
        epochs=500 if not quick_test else 100,
        lr=0.01,
        weight_decay=5e-4,
        patience=50,
        save_dir=str(Path(__file__).parent / 'checkpoints'),
        verbose=True
    )
    
    # =========================================================================
    # STEP 8: EVALUATE AND COMPARE
    # =========================================================================
    print("\n📊 STEP 8: Evaluating and comparing models...")
    
    # Get GNN predictions on test set
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        y_pred_gnn = out[data.test_mask].numpy().ravel()
    
    y_true_test = data.y[data.test_mask].numpy().ravel()
    
    gnn_metrics = calculate_metrics(y_true_test, y_pred_gnn)
    baseline_results['GAT (GNN)'] = gnn_metrics
    
    # Generate plots
    results_dir = Path(__file__).parent / 'results'
    
    generate_all_plots(
        history=history,
        y_true=y_true_test,
        y_pred=y_pred_gnn,
        model_name="Graph Attention Network",
        save_dir=str(results_dir)
    )
    
    # Model comparison plot
    plot_model_comparison(baseline_results, str(results_dir / 'model_comparison.png'))
    
    # =========================================================================
    # STEP 9: SAVE ARTIFACTS
    # =========================================================================
    print("\n💾 STEP 9: Saving artifacts...")
    
    # Save scalers and encoders
    checkpoints_dir = Path(__file__).parent / 'checkpoints'
    joblib.dump(scaler_X, checkpoints_dir / 'scaler_X.joblib')
    joblib.dump(scaler_y, checkpoints_dir / 'scaler_y.joblib')
    joblib.dump(le_location, checkpoints_dir / 'le_location.joblib')
    
    # Save feature names
    import json
    with open(checkpoints_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    # Save processed data for API
    df_final.to_csv(checkpoints_dir / 'processed_data.csv', index=False)
    
    print(f"  ✓ Scalers saved to {checkpoints_dir}")
    print(f"  ✓ Model saved to {checkpoints_dir}/best_gat_model.pt")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("  TRAINING COMPLETE - FINAL SUMMARY")
    print("="*70)
    print(f"\n  Dataset: {len(df_final):,} properties")
    print(f"  Graph: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
    print(f"\n  Test Set Performance:")
    print(f"    {'Model':<25} {'R²':>10} {'MAE':>12} {'RMSE':>12}")
    print(f"    {'-'*59}")
    
    for model_name, metrics in sorted(baseline_results.items(), key=lambda x: -x[1]['r2']):
        print(f"    {model_name:<25} {metrics['r2']:>10.4f} {metrics['mae']:>12.4f} {metrics['rmse']:>12.4f}")
    
    print(f"\n  Outputs:")
    print(f"    - Model: checkpoints/best_gat_model.pt")
    print(f"    - Plots: results/")
    print("="*70 + "\n")
    
    return model, data, history, baseline_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Bangalore Real Estate Price Predictor')
    parser.add_argument('--quick-test', action='store_true', help='Quick test with subset of data')
    args = parser.parse_args()
    
    model, data, history, results = main(quick_test=args.quick_test)
