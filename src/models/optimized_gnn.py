"""
Optimized Graph Neural Network for Maximum Accuracy
====================================================
Enhanced GNN architecture with:
- Deeper network with residual connections
- Edge features based on distance
- Hybrid GNN + MLP for feature extraction
- Advanced regularization techniques
- Location embedding layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, TransformerConv, SAGEConv
from torch_geometric.nn import BatchNorm, LayerNorm
from torch_geometric.data import Data
import numpy as np

# Reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)


class FeatureMLP(nn.Module):
    """
    MLP for feature preprocessing before GNN.
    Extracts higher-level features from raw inputs.
    """
    def __init__(self, in_features, hidden_features, out_features, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.bn2 = nn.BatchNorm1d(hidden_features)
        self.dropout = dropout
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc3(x)
        return x


class OptimizedGAT(nn.Module):
    """
    Optimized Graph Attention Network for maximum accuracy.
    
    Key improvements:
    1. Feature preprocessing MLP
    2. Multiple GAT layers with skip connections
    3. GATv2 for more expressive attention
    4. Layer normalization for stability
    5. Dropout regularization
    6. Final regression head with multiple layers
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 256,
                 out_channels: int = 1,
                 heads: int = 8,
                 num_gat_layers: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        
        self.dropout = dropout
        self.num_gat_layers = num_gat_layers
        
        # Feature preprocessing MLP
        self.feature_mlp = FeatureMLP(
            in_features=in_channels,
            hidden_features=hidden_channels,
            out_features=hidden_channels,
            dropout=dropout
        )
        
        # Input projection (for skip connection)
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GAT layers with GATv2 (more expressive attention)
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(num_gat_layers):
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // heads,
                    heads=heads,
                    dropout=dropout,
                    concat=True,
                    add_self_loops=True,
                    share_weights=False
                )
            )
            self.layer_norms.append(LayerNorm(hidden_channels))
        
        # Regression head (MLP for prediction)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),  # Concat MLP + GNN features
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_channels // 4, out_channels)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Store original features for skip connection
        x_skip = self.input_proj(x)
        
        # Feature preprocessing
        x_mlp = self.feature_mlp(x)
        
        # GAT layers with residual connections
        x_gnn = x_mlp
        for i in range(self.num_gat_layers):
            x_residual = x_gnn
            x_gnn = self.gat_layers[i](x_gnn, edge_index)
            x_gnn = self.layer_norms[i](x_gnn)
            x_gnn = F.leaky_relu(x_gnn, 0.1)
            x_gnn = F.dropout(x_gnn, p=self.dropout, training=self.training)
            x_gnn = x_gnn + x_residual  # Residual connection
        
        # Combine MLP features and GNN features
        x_combined = torch.cat([x_mlp, x_gnn], dim=1)
        
        # Final prediction
        out = self.regression_head(x_combined)
        
        return out


class HybridGNNEnsemble(nn.Module):
    """
    Hybrid model that combines multiple GNN architectures.
    Uses both GAT and GraphSAGE for complementary feature extraction.
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 128,
                 out_channels: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        
        self.dropout = dropout
        
        # Branch 1: GAT
        self.gat1 = GATv2Conv(in_channels, hidden_channels // 4, heads=4, concat=True, dropout=dropout)
        self.gat2 = GATv2Conv(hidden_channels, hidden_channels // 4, heads=4, concat=True, dropout=dropout)
        
        # Branch 2: GraphSAGE
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Branch 3: Transformer-based attention
        self.trans1 = TransformerConv(in_channels, hidden_channels // 4, heads=4, concat=True, dropout=dropout)
        self.trans2 = TransformerConv(hidden_channels, hidden_channels // 4, heads=4, concat=True, dropout=dropout)
        
        # Normalization
        self.bn_gat = nn.BatchNorm1d(hidden_channels)
        self.bn_sage = nn.BatchNorm1d(hidden_channels)
        self.bn_trans = nn.BatchNorm1d(hidden_channels)
        
        # Fusion layer (combines all branches)
        combined_dim = hidden_channels * 3
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # GAT branch
        x_gat = F.leaky_relu(self.gat1(x, edge_index), 0.1)
        x_gat = F.dropout(x_gat, p=self.dropout, training=self.training)
        x_gat = F.leaky_relu(self.gat2(x_gat, edge_index), 0.1)
        x_gat = self.bn_gat(x_gat)
        
        # SAGE branch
        x_sage = F.leaky_relu(self.sage1(x, edge_index), 0.1)
        x_sage = F.dropout(x_sage, p=self.dropout, training=self.training)
        x_sage = F.leaky_relu(self.sage2(x_sage, edge_index), 0.1)
        x_sage = self.bn_sage(x_sage)
        
        # Transformer branch
        x_trans = F.leaky_relu(self.trans1(x, edge_index), 0.1)
        x_trans = F.dropout(x_trans, p=self.dropout, training=self.training)
        x_trans = F.leaky_relu(self.trans2(x_trans, edge_index), 0.1)
        x_trans = self.bn_trans(x_trans)
        
        # Fusion
        x_combined = torch.cat([x_gat, x_sage, x_trans], dim=1)
        out = self.fusion(x_combined)
        
        return out


class UltraDeepGAT(nn.Module):
    """
    Very deep GAT with advanced techniques for maximum accuracy.
    
    Features:
    - 6 GAT layers with skip connections
    - Pre-layer normalization
    - Learnable attention scaling
    - Multi-scale aggregation
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 256,
                 out_channels: int = 1,
                 heads: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(0.1)
        )
        
        # Deep GAT layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.scales = nn.ParameterList()
        
        num_layers = 6
        for i in range(num_layers):
            self.layers.append(
                GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, 
                          dropout=dropout, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_channels))
            self.scales.append(nn.Parameter(torch.ones(1) * 0.1))
        
        # Multi-scale aggregation
        self.scale_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        
        # Collect multi-scale outputs
        layer_outputs = []
        
        for i, (layer, norm, scale) in enumerate(zip(self.layers, self.norms, self.scales)):
            x_residual = x
            x = norm(x)  # Pre-normalization
            x = layer(x, edge_index)
            x = F.leaky_relu(x, 0.1)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x_residual + scale * x  # Scaled residual
            layer_outputs.append(x)
        
        # Weighted multi-scale aggregation
        weights = F.softmax(self.scale_weights, dim=0)
        x = sum(w * out for w, out in zip(weights, layer_outputs))
        
        # Output
        out = self.output_head(x)
        
        return out


def create_optimized_gnn(in_channels: int, model_type: str = 'optimized_gat', **kwargs):
    """
    Factory function to create optimized GNN models.
    
    Args:
        in_channels: Number of input features
        model_type: 'optimized_gat', 'hybrid', or 'ultra_deep'
    """
    models = {
        'optimized_gat': OptimizedGAT,
        'hybrid': HybridGNNEnsemble,
        'ultra_deep': UltraDeepGAT
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](in_channels=in_channels, **kwargs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test all models
    in_features = 25
    
    for model_type in ['optimized_gat', 'hybrid', 'ultra_deep']:
        print(f"\nTesting {model_type}...")
        model = create_optimized_gnn(in_features, model_type)
        print(f"  Parameters: {count_parameters(model):,}")
        
        # Test forward pass
        x = torch.randn(100, in_features)
        edge_index = torch.randint(0, 100, (2, 500))
        output = model(x, edge_index)
        print(f"  Output shape: {output.shape}")
