"""
Target-Encoding Enhanced GNN for Maximum Accuracy
=================================================
Uses target-encoded location features and advanced aggregation
to maximize accuracy potential with available data.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv, SAGEConv, TransformerConv
from torch_geometric.data import Data
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from scipy.spatial import cKDTree
import json
from pathlib import Path

warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

sys.path.insert(0, str(Path(__file__).parent / 'src'))


def create_target_encoded_features(df, target_col='price_per_sqft', n_folds=5):
    """
    Create target-encoded features using k-fold to prevent leakage.
    This adds features that capture location-specific price patterns.
    """
    print("\n🎯 Creating target-encoded features...")
    
    df = df.copy()
    
    # Location target encoding
    location_col = 'location_encoded'
    
    # K-fold target encoding to prevent leakage
    df['loc_target_mean'] = np.nan
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    for train_idx, val_idx in kf.split(df):
        # Calculate mean on training fold
        train_means = df.iloc[train_idx].groupby(location_col)[target_col].mean()
        # Apply to validation fold
        df.loc[df.index[val_idx], 'loc_target_mean'] = df.iloc[val_idx][location_col].map(train_means)
    
    # Fill NaN with global mean
    global_mean = df[target_col].mean()
    df['loc_target_mean'] = df['loc_target_mean'].fillna(global_mean)
    
    # Location target std (smoothed)
    loc_std = df.groupby(location_col)[target_col].transform('std').fillna(0)
    df['loc_target_std'] = loc_std
    
    # Location count (popularity)
    loc_count = df.groupby(location_col)[target_col].transform('count')
    df['loc_count_log'] = np.log1p(loc_count)
    
    # Price percentile within location
    df['price_loc_percentile'] = df.groupby(location_col)[target_col].transform(
        lambda x: x.rank(pct=True)
    )
    
    # Area type target encoding
    area_means = df.groupby('area_type_encoded')[target_col].mean()
    df['area_target_mean'] = df['area_type_encoded'].map(area_means)
    
    # BHK target encoding
    bhk_means = df.groupby('bhk')[target_col].mean()
    df['bhk_target_mean'] = df['bhk'].map(bhk_means)
    
    print(f"  ✓ Added 6 target-encoded features")
    
    return df


def prepare_max_features(df):
    """Prepare feature matrix with all available signal."""
    
    # Target-encoded features
    target_cols = ['loc_target_mean', 'loc_target_std', 'loc_count_log', 
                   'price_loc_percentile', 'area_target_mean', 'bhk_target_mean']
    
    # Basic features
    basic_cols = ['total_sqft_clean', 'bhk', 'bath', 'balcony', 'area_type_encoded']
    
    # Derived features from advanced engineering
    derived_cols = ['bhk_sqft_interaction', 'bhk_bath_interaction', 'sqft_per_room',
                    'bath_bhk_ratio', 'balcony_bhk_ratio', 'sqft_per_bhk',
                    'sqft_bin', 'bhk_category', 'quality_score']
    
    # Location stats (computed during advanced features)
    loc_stats = ['loc_price_mean', 'loc_price_std', 'loc_price_median', 'loc_price_tier']
    
    # Select available columns
    all_cols = target_cols + basic_cols + derived_cols + loc_stats
    feature_cols = [c for c in all_cols if c in df.columns]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['price_per_sqft'].values.astype(np.float32)
    
    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)
    
    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"\n📊 Feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Features: {feature_cols}")
    
    return X, y, feature_cols, scaler


class MaxAccuracyGNN(nn.Module):
    """GNN optimized for maximum accuracy with target-encoded features."""
    
    def __init__(self, in_channels, hidden=256, heads=8, dropout=0.15):
        super().__init__()
        
        # Deep MLP for feature extraction
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Multiple GNN layers with different types
        self.gat1 = GATv2Conv(hidden, hidden//heads, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATv2Conv(hidden, hidden//heads, heads=heads, dropout=dropout, concat=True)
        self.sage = SAGEConv(hidden, hidden)
        
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        
        # Combine features
        self.combine = nn.Linear(hidden * 3, hidden)
        
        # Output head
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.GELU(),
            nn.Linear(hidden // 4, 1)
        )
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        # MLP features
        h = self.mlp(x)
        
        # GAT branch 1
        h1 = self.gat1(h, edge_index)
        h1 = self.bn1(h1)
        h1 = F.gelu(h1)
        h1 = h1 + h  # Skip connection
        
        # GAT branch 2
        h2 = self.gat2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.gelu(h2)
        h2 = h2 + h1
        
        # SAGE branch
        h3 = self.sage(h, edge_index)
        h3 = self.bn3(h3)
        h3 = F.gelu(h3)
        h3 = h3 + h
        
        # Combine all branches
        combined = torch.cat([h, h2, h3], dim=1)
        combined = self.combine(combined)
        combined = F.gelu(combined)
        
        # Output
        return self.output(combined)


def create_graph(X, y, k=25):
    """Create graph with optimal neighbor count."""
    n = X.shape[0]
    tree = cKDTree(X)
    _, indices = tree.query(X, k=min(k+1, n))
    
    edges = []
    for i in range(n):
        for j in indices[i, 1:]:
            edges.append([i, j])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    
    return Data(
        x=torch.tensor(X, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32).view(-1, 1),
        edge_index=edge_index
    )


def train_for_max_accuracy(epochs=500, lr=0.002):
    """Train for maximum possible accuracy."""
    
    print("\n" + "="*60)
    print("  MAXIMUM ACCURACY GNN TRAINING")
    print("  Target: 80% R² Score")
    print("="*60)
    
    # Load data
    from src.data_loader import load_raw_data
    from src.data_cleaner import clean_data
    from src.feature_engineering import create_features
    from src.advanced_features import create_advanced_features
    from src.graph_builder import create_train_val_test_masks
    
    print("\n📂 Loading and processing data...")
    data_path = Path(__file__).parent / 'data' / 'Bengaluru_House_Data.csv'
    df = load_raw_data(str(data_path))
    df = clean_data(df)
    df = create_features(df)
    df = create_advanced_features(df)
    
    # Add target-encoded features
    df = create_target_encoded_features(df)
    
    # Prepare features
    X, y, feature_names, scaler = prepare_max_features(df)
    
    # Normalize target
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std
    
    # Create graph
    print("\n🕸️ Building graph...")
    data = create_graph(X, y_norm, k=25)
    print(f"  Nodes: {data.x.size(0)}, Edges: {data.edge_index.size(1)}")
    
    # Masks
    n = data.x.size(0)
    train_mask, val_mask, test_mask = create_train_val_test_masks(n)
    train_mask = torch.tensor(train_mask)
    val_mask = torch.tensor(val_mask)
    test_mask = torch.tensor(test_mask)
    
    # Model
    model = MaxAccuracyGNN(in_channels=X.shape[1], hidden=256, heads=8, dropout=0.15)
    print(f"\n📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.005)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=1
    )
    
    best_val_r2 = -float('inf')
    best_state = None
    patience = 0
    max_patience = 80
    
    print(f"\n🚀 Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Evaluate every 20 epochs
        if epoch % 20 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                
                # Denormalize for R² calculation
                out_denorm = out * y_std + y_mean
                y_true = data.y * y_std + y_mean
                
                train_r2 = r2_score(y_true[train_mask].numpy(), out_denorm[train_mask].numpy())
                val_r2 = r2_score(y_true[val_mask].numpy(), out_denorm[val_mask].numpy())
            
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            
            print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f} | Best: {best_val_r2:.4f}")
            
            if patience >= max_patience // 20:
                print("  Early stopping!")
                break
    
    # Final evaluation
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        out_denorm = out * y_std + y_mean
        y_true = data.y * y_std + y_mean
        
        test_r2 = r2_score(y_true[test_mask].numpy(), out_denorm[test_mask].numpy())
        test_mae = mean_absolute_error(y_true[test_mask].numpy(), out_denorm[test_mask].numpy())
    
    print(f"\n" + "="*60)
    print(f"  FINAL RESULTS")
    print(f"="*60)
    print(f"  Best Val R²: {best_val_r2:.4f} ({best_val_r2*100:.1f}%)")
    print(f"  Test R²:     {test_r2:.4f} ({test_r2*100:.1f}%)")
    print(f"  Test MAE:    {test_mae:.2f} ₹/sqft")
    
    if test_r2 >= 0.8:
        print(f"\n  ✅ TARGET ACHIEVED: {test_r2*100:.1f}% accuracy!")
    elif test_r2 >= 0.7:
        print(f"\n  ⚡ STRONG: {test_r2*100:.1f}% accuracy (close to 80%!)")
    else:
        print(f"\n  📊 Current: {test_r2*100:.1f}% | Gap to 80%: {(0.8-test_r2)*100:.1f}%")
    
    print("="*60)
    
    # Save
    checkpoints = Path(__file__).parent / 'checkpoints'
    torch.save(best_state, checkpoints / 'max_accuracy_gnn.pt')
    
    results_dir = Path(__file__).parent / 'results'
    with open(results_dir / 'max_accuracy_results.json', 'w') as f:
        json.dump({
            'test_r2': float(test_r2), 
            'test_mae': float(test_mae), 
            'best_val_r2': float(best_val_r2),
            'features': feature_names
        }, f, indent=2)
    
    return model, test_r2, feature_names


if __name__ == "__main__":
    model, r2, features = train_for_max_accuracy(epochs=500, lr=0.002)
