"""
Ensemble Model Module
=====================
Implements stacking ensemble with hyperparameter-tuned base models
for maximum predictive accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    StackingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def create_optimized_models():
    """
    Create base models with optimized hyperparameters.
    """
    models = {
        'random_forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        ),
        'extra_trees': ExtraTreesRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42
        ),
        'ridge': Ridge(alpha=1.0),
        'knn': KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1),
    }
    return models


def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Perform hyperparameter tuning on key models.
    """
    print("\n🔧 Tuning hyperparameters...")
    
    best_models = {}
    
    # 1. Random Forest tuning
    print("  Tuning Random Forest...")
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [15, 20, 25],
        'min_samples_split': [2, 5],
    }
    rf = RandomForestRegressor(n_jobs=-1, random_state=42)
    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='r2', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_models['random_forest'] = rf_grid.best_estimator_
    print(f"    Best RF params: {rf_grid.best_params_}")
    print(f"    RF R² on val: {rf_grid.best_estimator_.score(X_val, y_val):.4f}")
    
    # 2. Gradient Boosting tuning
    print("  Tuning Gradient Boosting...")
    gb_params = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
    }
    gb = GradientBoostingRegressor(random_state=42)
    gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='r2', n_jobs=-1)
    gb_grid.fit(X_train, y_train)
    best_models['gradient_boosting'] = gb_grid.best_estimator_
    print(f"    Best GB params: {gb_grid.best_params_}")
    print(f"    GB R² on val: {gb_grid.best_estimator_.score(X_val, y_val):.4f}")
    
    # 3. Extra Trees
    print("  Training Extra Trees...")
    et = ExtraTreesRegressor(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
    et.fit(X_train, y_train)
    best_models['extra_trees'] = et
    print(f"    ET R² on val: {et.score(X_val, y_val):.4f}")
    
    return best_models


def create_stacking_ensemble(base_models, X_train, y_train):
    """
    Create a stacking ensemble with the best base models.
    """
    print("\n📦 Building stacking ensemble...")
    
    estimators = [
        ('rf', base_models['random_forest']),
        ('gb', base_models['gradient_boosting']),
        ('et', base_models['extra_trees']),
    ]
    
    # Meta-learner
    meta_learner = Ridge(alpha=1.0)
    
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        passthrough=True  # Include original features for meta-learner
    )
    
    print("  Training stacking ensemble...")
    stacking.fit(X_train, y_train)
    
    return stacking


def train_ensemble_pipeline(X_train, y_train, X_val, y_val, X_test, y_test, save_dir=None):
    """
    Full ensemble training pipeline with hyperparameter tuning.
    
    Returns:
        dict: Results with metrics and trained models
    """
    print("\n" + "="*60)
    print("🚀 ENSEMBLE TRAINING PIPELINE")
    print("="*60)
    
    results = {}
    
    # 1. Tune hyperparameters
    best_models = tune_hyperparameters(X_train, y_train, X_val, y_val)
    
    # 2. Evaluate individual tuned models on test set
    print("\n📊 Individual Model Performance (Test Set):")
    print("-"*50)
    
    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        results[name] = {'r2': r2, 'mae': mae, 'model': model}
        print(f"  {name:20s} | R²: {r2:.4f} | MAE: {mae:.4f}")
    
    # 3. Create and train stacking ensemble
    stacking = create_stacking_ensemble(best_models, X_train, y_train)
    
    # 4. Evaluate ensemble
    y_pred_stack = stacking.predict(X_test)
    stack_r2 = r2_score(y_test, y_pred_stack)
    stack_mae = mean_absolute_error(y_test, y_pred_stack)
    results['stacking_ensemble'] = {'r2': stack_r2, 'mae': stack_mae, 'model': stacking}
    
    print(f"\n  {'STACKING ENSEMBLE':20s} | R²: {stack_r2:.4f} | MAE: {stack_mae:.4f}")
    
    # 5. Simple averaging ensemble
    print("\n📊 Averaging Ensemble:")
    predictions = np.column_stack([
        best_models['random_forest'].predict(X_test),
        best_models['gradient_boosting'].predict(X_test),
        best_models['extra_trees'].predict(X_test),
    ])
    y_pred_avg = predictions.mean(axis=1)
    avg_r2 = r2_score(y_test, y_pred_avg)
    avg_mae = mean_absolute_error(y_test, y_pred_avg)
    results['averaging_ensemble'] = {'r2': avg_r2, 'mae': avg_mae}
    print(f"  {'AVERAGING ENSEMBLE':20s} | R²: {avg_r2:.4f} | MAE: {avg_mae:.4f}")
    
    # 6. Find best model
    best_name = max(results.keys(), key=lambda k: results[k]['r2'])
    best_r2 = results[best_name]['r2']
    
    print("\n" + "="*60)
    print(f"🏆 BEST MODEL: {best_name.upper()}")
    print(f"   R² Score: {best_r2:.4f} ({best_r2*100:.1f}%)")
    print("="*60)
    
    # 7. Save models
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for name, data in results.items():
            if 'model' in data:
                joblib.dump(data['model'], save_path / f'{name}_optimized.joblib')
        
        print(f"\n✓ Models saved to {save_dir}")
    
    return results, best_models, stacking


if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5
    
    # Split
    train_end = int(0.7 * n_samples)
    val_end = int(0.85 * n_samples)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    results, models, ensemble = train_ensemble_pipeline(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
