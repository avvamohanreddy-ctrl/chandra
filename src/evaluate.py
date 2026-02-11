"""
Evaluation Module for Bangalore Real Estate Prediction
=======================================================
Compares baseline models with GNN, generates visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import json


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate all evaluation metrics."""
    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
        'accuracy_10pct': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)) < 0.1) * 100,
        'accuracy_20pct': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)) < 0.2) * 100
    }


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray,
                             title: str = "Actual vs Predicted",
                             save_path: str = None):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 8))
    
    plt.scatter(y_true, y_pred, alpha=0.5, s=10, c='#6366f1')
    
    # Perfect prediction line
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate R²
    r2 = r2_score(y_true, y_pred)
    
    plt.xlabel('Actual Price (₹/sqft)', fontsize=12)
    plt.ylabel('Predicted Price (₹/sqft)', fontsize=12)
    plt.title(f'{title}\nR² = {r2:.4f}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    plt.close()


def plot_training_history(history: dict, save_path: str = None):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', color='#3b82f6')
    axes[0].plot(history['val_loss'], label='Val Loss', color='#ef4444')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # R² curves
    axes[1].plot(history['train_r2'], label='Train R²', color='#3b82f6')
    axes[1].plot(history['val_r2'], label='Val R²', color='#ef4444')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('R² Score Over Training')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    plt.close()


def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                            save_path: str = None):
    """Plot distribution of prediction errors."""
    errors = y_pred - y_true
    pct_errors = (y_pred - y_true) / (y_true + 1e-8) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute error distribution
    axes[0].hist(errors, bins=50, color='#6366f1', alpha=0.7, edgecolor='white')
    axes[0].axvline(0, color='red', linestyle='--', lw=2)
    axes[0].set_xlabel('Error (₹/sqft)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Prediction Errors')
    axes[0].grid(True, alpha=0.3)
    
    # Percentage error distribution
    axes[1].hist(pct_errors, bins=50, color='#10b981', alpha=0.7, edgecolor='white',
                 range=(-100, 100))
    axes[1].axvline(0, color='red', linestyle='--', lw=2)
    axes[1].set_xlabel('Percentage Error (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Percentage Errors')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    plt.close()


def plot_model_comparison(results: dict, save_path: str = None):
    """Compare performance of different models."""
    models = list(results.keys())
    r2_scores = [results[m]['r2'] for m in models]
    mae_scores = [results[m]['mae'] for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
    
    # R² comparison
    bars1 = axes[0].bar(models, r2_scores, color=colors[:len(models)], edgecolor='white')
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('Model Comparison: R² Score')
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars1, r2_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{score:.3f}', ha='center', fontsize=10)
    
    # MAE comparison
    bars2 = axes[1].bar(models, mae_scores, color=colors[:len(models)], edgecolor='white')
    axes[1].set_ylabel('MAE (₹/sqft)')
    axes[1].set_title('Model Comparison: MAE (Lower is Better)')
    axes[1].tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars2, mae_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{score:.1f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")
    
    plt.close()


def generate_all_plots(history: dict, y_true: np.ndarray, y_pred: np.ndarray,
                       model_name: str = "GAT", save_dir: str = "results"):
    """Generate all evaluation plots."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating evaluation plots...")
    
    # Training history
    plot_training_history(history, save_path / 'training_history.png')
    
    # Actual vs Predicted
    plot_actual_vs_predicted(y_true, y_pred, f"{model_name}: Actual vs Predicted",
                             save_path / 'actual_vs_predicted.png')
    
    # Error distribution
    plot_error_distribution(y_true, y_pred, save_path / 'error_distribution.png')
    
    # Save metrics
    metrics = calculate_metrics(y_true, y_pred)
    with open(save_path / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n  All plots saved to {save_dir}/")
    
    return metrics


if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    
    y_true = np.random.uniform(5000, 15000, 100)
    y_pred = y_true + np.random.normal(0, 500, 100)
    
    history = {
        'train_loss': np.linspace(1, 0.1, 100).tolist(),
        'val_loss': np.linspace(1.2, 0.15, 100).tolist(),
        'train_r2': np.linspace(0.5, 0.95, 100).tolist(),
        'val_r2': np.linspace(0.4, 0.9, 100).tolist()
    }
    
    metrics = generate_all_plots(history, y_true, y_pred, save_dir='../results')
    print("\nMetrics:", metrics)
