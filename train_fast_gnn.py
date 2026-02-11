"""
Fast GNN Training for Maximum Accuracy
======================================
Optimized for speed with effective training.
"""

import os
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv, SAGEConv
from torch_geometric.data import Data
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.spatial import cKDTree
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

sys.path.insert(0, str(Path(__file__).parent / 'src'))


class FastGAT(nn.Module):
    """Fast GAT optimized for performance."""
    def __init__(self, in_channels, hidden=128, out_channels=1, heads=4, dropout=0.2):
        super().__init__()
        
        # Feature preprocessing
        self.pre = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout)
        )
        
        # GAT layers
        self.gat1 = GATv2Conv(hidden, hidden//heads, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATv2Conv(hidden, hidden//heads, heads=heads, dropout=dropout, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        
        # Output
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden//2, out_channels)
        )
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        # Preprocess
        x = self.pre(x)
        h = x
        
        # GAT 1 + skip
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.1)
        x = F.dropout(x, self.dropout, self.training)
        x = x + h
        
        h = x
        # GAT 2 + skip
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.1)
        x = F.dropout(x, self.dropout, self.training)
        x = x + h
        
        return self.out(x)


def create_graph(X, y, k=20):
    """Fast graph creation with fewer edges."""
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


def train_fast(epochs=300, lr=0.005):
    """Fast training pipeline."""
    
    print("\n" + "="*60)
    print("  FAST GNN TRAINING FOR 80% ACCURACY")
    print("="*60)
    
    # Load data
    from src.data_loader import load_raw_data
    from src.data_cleaner import clean_data
    from src.feature_engineering import create_features
    from src.advanced_features import create_advanced_features, prepare_advanced_features
    from src.graph_builder import create_train_val_test_masks
    
    print("\n📂 Loading data...")
    data_path = Path(__file__).parent / 'data' / 'Bengaluru_House_Data.csv'
    df = load_raw_data(str(data_path))
    df = clean_data(df)
    df = create_features(df)
    df = create_advanced_features(df)
    X, y, features, _, _, _, _ = prepare_advanced_features(df)
    
    print(f"\n  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    
    # Create graph
    print("\n🕸️ Building graph...")
    data = create_graph(X, y, k=20)
    print(f"  Nodes: {data.x.size(0)}, Edges: {data.edge_index.size(1)}")
    
    # Create masks
    n = data.x.size(0)
    train_mask, val_mask, test_mask = create_train_val_test_masks(n)
    train_mask = torch.tensor(train_mask)
    val_mask = torch.tensor(val_mask)
    test_mask = torch.tensor(test_mask)
    
    # Model
    model = FastGAT(in_channels=X.shape[1], hidden=128, heads=4, dropout=0.2)
    print(f"\n📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_val_r2 = -float('inf')
    best_state = None
    patience = 0
    max_patience = 50
    
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
        
        # Eval
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                train_r2 = r2_score(data.y[train_mask].numpy(), out[train_mask].numpy())
                val_r2 = r2_score(data.y[val_mask].numpy(), out[val_mask].numpy())
            
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_state = model.state_dict().copy()
                patience = 0
            else:
                patience += 1
            
            print(f"  Epoch {epoch:4d} | Loss: {loss:.4f} | Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f} | Best: {best_val_r2:.4f}")
            
            if patience >= max_patience // 10:
                print("  Early stopping!")
                break
    
    # Load best and evaluate on test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_r2 = r2_score(data.y[test_mask].numpy(), out[test_mask].numpy())
        test_mae = mean_absolute_error(data.y[test_mask].numpy(), out[test_mask].numpy())
    
    print(f"\n" + "="*60)
    print(f"  RESULTS")
    print(f"="*60)
    print(f"  Best Val R²: {best_val_r2:.4f} ({best_val_r2*100:.1f}%)")
    print(f"  Test R²:     {test_r2:.4f} ({test_r2*100:.1f}%)")
    print(f"  Test MAE:    {test_mae:.4f}")
    
    if test_r2 >= 0.8:
        print(f"\n  ✅ TARGET ACHIEVED: 80%+ accuracy!")
    else:
        print(f"\n  Gap to 80%: {(0.8 - test_r2)*100:.1f}%")
    
    print("="*60)
    
    # Save model
    checkpoints = Path(__file__).parent / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    torch.save(best_state, checkpoints / 'fast_gat_best.pt')
    
    # Save results
    results_dir = Path(__file__).parent / 'results'
    with open(results_dir / 'fast_gat_results.json', 'w') as f:
        json.dump({'test_r2': float(test_r2), 'test_mae': float(test_mae), 'best_val_r2': float(best_val_r2)}, f)
    
    return model, test_r2


if __name__ == "__main__":
    model, r2 = train_fast(epochs=300, lr=0.005)
