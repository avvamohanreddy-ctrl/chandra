"""
Graph Attention Network Model for Bangalore Real Estate Prediction
===================================================================
Implements GAT and GCN models using PyTorch Geometric.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

# Reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)


class BangaloreGAT(nn.Module):
    """
    Graph Attention Network for real estate price prediction.
    
    Architecture:
    - Input: Node features (property attributes + encoded location)
    - GAT Layer 1: Multi-head attention (4 heads, 64 hidden units)
    - GAT Layer 2: Multi-head attention (4 heads, 32 hidden units)
    - Output: Predicted price_per_sqft
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 out_channels: int = 1,
                 heads: int = 4,
                 dropout: float = 0.3):
        super(BangaloreGAT, self).__init__()
        
        self.dropout = dropout
        
        # First GAT layer
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        
        # Second GAT layer
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels // 2,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        
        # Final prediction layer
        self.fc = nn.Linear((hidden_channels // 2) * heads, out_channels)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.bn2 = nn.BatchNorm1d((hidden_channels // 2) * heads)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GAT model.
        """
        # First GAT layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GAT layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final prediction
        x = self.fc(x)
        
        return x
    
    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Extract attention weights for interpretability."""
        _, attention_weights = self.conv1(x, edge_index, return_attention_weights=True)
        return attention_weights


class BangaloreGCN(nn.Module):
    """
    Graph Convolutional Network for comparison with GAT.
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 out_channels: int = 1,
                 dropout: float = 0.3):
        super(BangaloreGCN, self).__init__()
        
        self.dropout = dropout
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.fc = nn.Linear(hidden_channels // 2, out_channels)
        
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels // 2)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.fc(x)
        return x


class DeepGAT(nn.Module):
    """
    Deeper GAT with residual connections for potentially better performance.
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 128,
                 out_channels: int = 1,
                 heads: int = 8,
                 dropout: float = 0.4,
                 num_layers: int = 3):
        super(DeepGAT, self).__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(
                GATConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // heads,
                    heads=heads,
                    dropout=dropout,
                    concat=True
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.fc = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Project input
        x = self.input_proj(x)
        x = F.elu(x)
        
        # GAT layers with residual connections
        for i in range(self.num_layers):
            residual = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual  # Residual connection
        
        # Output
        x = self.fc(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(model_type: str, in_channels: int, **kwargs) -> nn.Module:
    """
    Factory function to get model by type.
    
    Args:
        model_type: One of 'gat', 'gcn', 'deep_gat'
        in_channels: Number of input features
        
    Returns:
        Model instance
    """
    models = {
        'gat': BangaloreGAT,
        'gcn': BangaloreGCN,
        'deep_gat': DeepGAT
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type](in_channels=in_channels, **kwargs)


if __name__ == "__main__":
    # Test models
    in_features = 6
    
    print("Testing BangaloreGAT...")
    gat_model = BangaloreGAT(in_channels=in_features)
    print(f"  Parameters: {count_parameters(gat_model):,}")
    
    # Test forward pass
    dummy_x = torch.randn(100, in_features)
    dummy_edge = torch.randint(0, 100, (2, 500))
    output = gat_model(dummy_x, dummy_edge)
    print(f"  Output shape: {output.shape}")
    
    print("\nTesting BangaloreGCN...")
    gcn_model = BangaloreGCN(in_channels=in_features)
    print(f"  Parameters: {count_parameters(gcn_model):,}")
    
    print("\nTesting DeepGAT...")
    deep_model = DeepGAT(in_channels=in_features)
    print(f"  Parameters: {count_parameters(deep_model):,}")
