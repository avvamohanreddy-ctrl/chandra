"""
Maximum Accuracy GNN Training Pipeline
======================================
Trains optimized GNN models with advanced features for maximum accuracy.

Target: 80% R² Score
"""

import os
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch_geometric.data import Data
from sklearn.metrics import r2_score, mean_absolute_error
import json
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader import load_raw_data
from src.data_cleaner import clean_data
from src.feature_engineering import create_features
from src.advanced_features import create_advanced_features, prepare_advanced_features
from src.graph_builder import create_train_val_test_masks
from src.models.optimized_gnn import create_optimized_gnn, count_parameters


def create_enhanced_graph(X: np.ndarray, y: np.ndarray, 
                          coords: np.ndarray = None, 
                          k_neighbors: int = 30) -> Data:
    """
    Create graph with more connections for better information flow.
    Uses scipy for stability.
    """
    from scipy.spatial import cKDTree
    
    n_samples = X.shape[0]
    k = min(k_neighbors + 1, n_samples)
    
    # Use KDTree for neighbor finding (more stable than sklearn)
    tree = cKDTree(X)
    distances, indices = tree.query(X, k=k)
    
    # Create edge list (excluding self-loops)
    edge_list = []
    for i in range(n_samples):
        for j in indices[i, 1:]:  # Skip first (self)
            edge_list.append([i, j])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Make bidirectional
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    
    # Create graph data
    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32).view(-1, 1),
        edge_index=edge_index
    )
    
    print(f"  Graph: {n_samples} nodes, {edge_index.size(1)} edges")
    print(f"  Avg degree: {edge_index.size(1) / n_samples:.1f}")
    
    return data


def train_epoch(model, data, optimizer, criterion, train_mask):
    """Single training epoch with gradient clipping."""
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    """Evaluate model on a subset of data."""
    model.eval()
    out = model(data.x, data.edge_index)
    
    predictions = out[mask].cpu().numpy().flatten()
    targets = data.y[mask].cpu().numpy().flatten()
    
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    
    return r2, mae


def train_optimized_gnn(data, model_type='optimized_gat', epochs=1000, lr=0.001, 
                        patience=100, device='cpu', verbose=True):
    """
    Train GNN with advanced optimization techniques.
    
    Features:
    - Cosine annealing with warm restarts
    - Early stopping with patience
    - Gradient clipping
    - Best model checkpointing
    """
    in_channels = data.x.size(1)
    
    # Create model
    model = create_optimized_gnn(in_channels, model_type=model_type)
    model = model.to(device)
    data = data.to(device)
    
    print(f"\n{'='*60}")
    print(f"  TRAINING {model_type.upper()}")
    print(f"{'='*60}")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}, LR: {lr}")
    
    # Create masks
    n_nodes = data.x.size(0)
    train_mask, val_mask, test_mask = create_train_val_test_masks(n_nodes)
    
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)
    
    # Loss, optimizer, scheduler
    criterion = nn.HuberLoss(delta=1.0)  # Robust to outliers
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    
    # Training loop
    best_val_r2 = -float('inf')
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_r2': [], 'val_mae': [], 'train_r2': []}
    
    iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
    
    for epoch in iterator:
        # Train
        train_loss = train_epoch(model, data, optimizer, criterion, train_mask)
        scheduler.step()
        
        # Evaluate
        train_r2, _ = evaluate(model, data, train_mask)
        val_r2, val_mae = evaluate(model, data, val_mask)
        
        history['train_loss'].append(train_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['val_mae'].append(val_mae)
        
        # Check for improvement
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Update progress bar
        if verbose and epoch % 10 == 0:
            iterator.set_postfix({
                'loss': f'{train_loss:.4f}',
                'train_r2': f'{train_r2:.4f}',
                'val_r2': f'{val_r2:.4f}',
                'best': f'{best_val_r2:.4f}'
            })
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    test_r2, test_mae = evaluate(model, data, test_mask)
    
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE - {model_type.upper()}")
    print(f"{'='*60}")
    print(f"  Best Val R²:  {best_val_r2:.4f} ({best_val_r2*100:.1f}%)")
    print(f"  Test R²:      {test_r2:.4f} ({test_r2*100:.1f}%)")
    print(f"  Test MAE:     {test_mae:.4f}")
    print(f"{'='*60}")
    
    return model, history, {'r2': test_r2, 'mae': test_mae, 'best_val_r2': best_val_r2}


def main():
    """Main training pipeline for maximum GNN accuracy."""
    
    print("\n" + "="*70)
    print("  MAXIMUM ACCURACY GNN TRAINING")
    print("  Target: 80% R² Score")
    print("="*70)
    
    # Device - Force CPU to avoid MPS segfaults
    device = 'cpu'  # MPS causes segfaults with complex GNN operations
    print(f"\n  Using device: {device}")
    
    # Load and process data
    print("\n📂 Loading and processing data...")
    
    data_path = Path(__file__).parent / 'data' / 'Bengaluru_House_Data.csv'
    df = load_raw_data(str(data_path))
    df = clean_data(df)
    df = create_features(df)
    df = create_advanced_features(df)
    
    X, y, feature_names, le_location, scaler_X, scaler_y, df_final = prepare_advanced_features(df)
    
    print(f"\n  Dataset: {len(X):,} properties")
    print(f"  Features: {len(feature_names)}")
    
    # Create enhanced graph
    print("\n🕸️ Building enhanced graph...")
    data = create_enhanced_graph(X, y, k_neighbors=40)  # More neighbors
    
    # Train multiple GNN variants
    results = {}
    best_model = None
    best_r2 = 0
    best_history = None
    
    model_configs = [
        ('optimized_gat', {'epochs': 1000, 'lr': 0.001, 'patience': 100}),
        ('hybrid', {'epochs': 800, 'lr': 0.001, 'patience': 80}),
        ('ultra_deep', {'epochs': 1000, 'lr': 0.0005, 'patience': 100}),
    ]
    
    for model_type, config in model_configs:
        print(f"\n{'='*70}")
        print(f"  Training: {model_type.upper()}")
        print(f"{'='*70}")
        
        try:
            model, history, metrics = train_optimized_gnn(
                data, 
                model_type=model_type,
                device=device,
                **config
            )
            results[model_type] = metrics
            
            if metrics['r2'] > best_r2:
                best_r2 = metrics['r2']
                best_model = model
                best_model_type = model_type
                best_history = history
                
        except Exception as e:
            print(f"  Error training {model_type}: {e}")
            continue
    
    # Save best model
    checkpoints_dir = Path(__file__).parent / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    
    torch.save(best_model.state_dict(), checkpoints_dir / 'best_optimized_gnn.pt')
    
    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'gnn_optimization_results.json', 'w') as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
    
    # Final summary
    print("\n" + "="*70)
    print("  FINAL GNN OPTIMIZATION RESULTS")
    print("="*70)
    print(f"\n  {'Model':<25} {'Test R²':<12} {'Accuracy %':<12}")
    print(f"  {'-'*49}")
    
    for model_name, metrics in sorted(results.items(), key=lambda x: -x[1]['r2']):
        accuracy = metrics['r2'] * 100
        marker = " 🏆" if model_name == best_model_type else ""
        print(f"  {model_name:<25} {metrics['r2']:.4f}       {accuracy:.1f}%{marker}")
    
    print(f"\n  🎯 Best Model: {best_model_type.upper()}")
    print(f"  📊 Accuracy: {best_r2*100:.1f}%")
    
    if best_r2 >= 0.8:
        print(f"\n  ✅ TARGET ACHIEVED: 80% accuracy reached!")
    else:
        print(f"\n  ⚠️ Current accuracy: {best_r2*100:.1f}%")
        print(f"     Gap to 80%: {(0.8 - best_r2)*100:.1f}%")
    
    print("="*70 + "\n")
    
    return results, best_model, best_history


if __name__ == "__main__":
    results, model, history = main()
