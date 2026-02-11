"""
Training Pipeline for Bangalore Real Estate Prediction
=======================================================
Handles model training with early stopping and metric tracking.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data
from pathlib import Path
import json
from datetime import datetime


def train_epoch(model: nn.Module, data: Data, optimizer: optim.Optimizer,
                criterion: nn.Module) -> float:
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(model: nn.Module, data: Data, criterion: nn.Module, 
             mask: torch.Tensor) -> tuple:
    """Evaluate model on given mask."""
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        loss = criterion(out[mask], data.y[mask])
        
        # Calculate R² score
        y_true = data.y[mask].numpy()
        y_pred = out[mask].numpy()
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # MAE
        mae = np.mean(np.abs(y_true - y_pred))
        
    return loss.item(), r2, mae


def train(model: nn.Module, data: Data, epochs: int = 500,
          lr: float = 0.01, weight_decay: float = 5e-4,
          patience: int = 50, save_dir: str = 'checkpoints',
          verbose: bool = True) -> dict:
    """
    Train the GNN model with early stopping.
    
    Args:
        model: PyTorch model
        data: PyTorch Geometric Data object
        epochs: Maximum training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        patience: Early stopping patience
        save_dir: Directory for saving checkpoints
        verbose: Print training progress
        
    Returns:
        Training history dictionary
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_r2': [],
        'val_r2': [],
        'train_mae': [],
        'val_mae': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    no_improve = 0
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("\n" + "="*60)
        print("TRAINING GNN MODEL")
        print("="*60)
        print(f"Epochs: {epochs}, LR: {lr}, Weight Decay: {weight_decay}")
        print(f"Device: {device}")
        print("-"*60)
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, data, optimizer, criterion)
        
        # Evaluate
        _, train_r2, train_mae = evaluate(model, data, criterion, data.train_mask)
        val_loss, val_r2, val_mae = evaluate(model, data, criterion, data.val_mask)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            
            # Save best model
            torch.save(model.state_dict(), save_path / 'best_gat_model.pt')
        else:
            no_improve += 1
        
        # Print progress
        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:4d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val R²: {val_r2:.4f}")
        
        # Early stopping
        if no_improve >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(save_path / 'best_gat_model.pt'))
    
    # Final evaluation
    _, test_r2, test_mae = evaluate(model, data, criterion, data.test_mask)
    
    if verbose:
        print("-"*60)
        print(f"Best epoch: {best_epoch + 1}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print("="*60)
    
    # Save training history
    history['best_epoch'] = best_epoch
    history['test_r2'] = test_r2
    history['test_mae'] = test_mae
    
    with open(save_path / 'training_history.json', 'w') as f:
        serializable_history = {}
        for k, v in history.items():
            if isinstance(v, list):
                serializable_history[k] = [float(x) for x in v]
            elif isinstance(v, (np.floating, np.integer)):
                serializable_history[k] = float(v)
            else:
                serializable_history[k] = v
        json.dump(serializable_history, f, indent=2)
    
    return history


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from models.gnn_model import BangaloreGAT
    from graph_builder import create_graph_data, create_train_val_test_masks
    
    # Test with dummy data
    np.random.seed(42)
    n_samples = 100
    n_features = 6
    
    features = np.random.randn(n_samples, n_features)
    target = np.random.randn(n_samples)
    coords = np.column_stack([
        np.random.uniform(12.8, 13.2, n_samples),
        np.random.uniform(77.4, 77.8, n_samples)
    ])
    
    train_mask, val_mask, test_mask = create_train_val_test_masks(n_samples)
    data = create_graph_data(features, target, coords, train_mask, val_mask, test_mask)
    
    model = BangaloreGAT(in_channels=n_features)
    history = train(model, data, epochs=100, verbose=True)
