"""
Baseline Machine Learning Models for Bangalore Real Estate Prediction
======================================================================
Implements Linear Regression, Random Forest, and XGBoost baselines.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path

try:
    import xgboost as xgb
    # Disable XGBoost temporarily due to segfault issues on macOS
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost disabled for stability")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, will skip XGBoost model")


def train_baseline_models(X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          save_dir: str = None) -> dict:
    """
    Train all baseline models and return results.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        save_dir: Directory to save trained models
        
    Returns:
        Dictionary with model results
    """
    results = {}
    models = {}
    
    print("\n" + "="*60)
    print("TRAINING BASELINE MODELS")
    print("="*60)
    
    # 1. Linear Regression
    print("\n1. Linear Regression...")
    lr = Ridge(alpha=1.0)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_val)
    
    results['Linear Regression'] = evaluate_model(y_val, y_pred_lr)
    models['linear_regression'] = lr
    print(f"   R² = {results['Linear Regression']['r2']:.4f}")
    
    # 2. Random Forest
    print("\n2. Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_val)
    
    results['Random Forest'] = evaluate_model(y_val, y_pred_rf)
    models['random_forest'] = rf
    print(f"   R² = {results['Random Forest']['r2']:.4f}")
    
    # 3. Gradient Boosting (sklearn)
    print("\n3. Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_val)
    
    results['Gradient Boosting'] = evaluate_model(y_val, y_pred_gb)
    models['gradient_boosting'] = gb
    print(f"   R² = {results['Gradient Boosting']['r2']:.4f}")
    
    # 4. XGBoost (if available)
    if XGBOOST_AVAILABLE:
        print("\n4. XGBoost...")
        try:
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            y_pred_xgb = xgb_model.predict(X_val)
            
            results['XGBoost'] = evaluate_model(y_val, y_pred_xgb)
            models['xgboost'] = xgb_model
            print(f"   R² = {results['XGBoost']['r2']:.4f}")
        except Exception as e:
            print(f"   ⚠ XGBoost failed: {e}")
            print("   Skipping XGBoost model...")
    
    # Print comparison
    print("\n" + "="*60)
    print("BASELINE MODEL COMPARISON")
    print("="*60)
    print(f"{'Model':<25} {'R²':>10} {'MAE':>12} {'RMSE':>12}")
    print("-"*60)
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['r2']:>10.4f} {metrics['mae']:>12.2f} {metrics['rmse']:>12.2f}")
    
    # Save models
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for name, model in models.items():
            joblib.dump(model, save_path / f'{name}.joblib')
        print(f"\n✓ Models saved to {save_dir}")
    
    return results, models


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate evaluation metrics.
    """
    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    }


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Get feature importance from tree-based models.
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        return None
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return df


if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    n_samples = 1000
    n_features = 6
    
    X = np.random.randn(n_samples, n_features)
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(n_samples) * 0.5
    
    # Split
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    results, models = train_baseline_models(X_train, y_train, X_val, y_val)
