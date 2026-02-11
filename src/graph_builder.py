"""
Graph Construction Module for Bangalore Real Estate Prediction
===============================================================
Builds spatial proximity graph using k-NN on geographic coordinates.
"""

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data


def build_knn_graph(coordinates: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Build k-NN graph based on geographic coordinates.
    
    Args:
        coordinates: (N, 2) array of [latitude, longitude]
        k: Number of nearest neighbors
        
    Returns:
        Edge index array of shape (2, num_edges)
    """
    n_samples = len(coordinates)
    k = min(k, n_samples - 1)  # Can't have more neighbors than samples
    
    # Fit k-NN model
    knn = NearestNeighbors(n_neighbors=k + 1, metric='haversine')
    
    # Convert to radians for haversine
    coords_rad = np.radians(coordinates)
    knn.fit(coords_rad)
    
    # Get neighbors (includes self, so we use k+1)
    distances, indices = knn.kneighbors(coords_rad)
    
    # Build edge list (excluding self-loops)
    edges_source = []
    edges_target = []
    
    for i in range(n_samples):
        for j in range(1, k + 1):  # Skip index 0 (self)
            neighbor_idx = indices[i, j]
            edges_source.append(i)
            edges_target.append(neighbor_idx)
            # Add reverse edge for undirected graph
            edges_source.append(neighbor_idx)
            edges_target.append(i)
    
    edge_index = np.array([edges_source, edges_target])
    
    # Remove duplicate edges
    edge_index = np.unique(edge_index, axis=1)
    
    return edge_index


def create_graph_data(
    features: np.ndarray,
    target: np.ndarray,
    coordinates: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    k: int = 10
) -> Data:
    """
    Create PyTorch Geometric Data object for GNN training.
    
    Args:
        features: Node feature matrix (N, F)
        target: Target values (N,)
        coordinates: Lat/lng coordinates (N, 2)
        train_mask, val_mask, test_mask: Boolean masks for splits
        k: Number of nearest neighbors for graph construction
        
    Returns:
        PyTorch Geometric Data object
    """
    print("Constructing spatial proximity graph...")
    
    # Build k-NN graph
    edge_index = build_knn_graph(coordinates, k)
    
    # Convert to tensors
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(target, dtype=torch.float32).view(-1, 1)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    
    train_mask_tensor = torch.tensor(train_mask, dtype=torch.bool)
    val_mask_tensor = torch.tensor(val_mask, dtype=torch.bool)
    test_mask_tensor = torch.tensor(test_mask, dtype=torch.bool)
    
    # Create Data object
    data = Data(
        x=x,
        y=y,
        edge_index=edge_index_tensor,
        train_mask=train_mask_tensor,
        val_mask=val_mask_tensor,
        test_mask=test_mask_tensor
    )
    
    # Store coordinates for visualization
    data.coordinates = torch.tensor(coordinates, dtype=torch.float32)
    
    print(f"  ✓ Graph created:")
    print(f"    Nodes: {data.num_nodes:,}")
    print(f"    Edges: {data.num_edges:,}")
    print(f"    Features: {data.num_node_features}")
    print(f"    Avg degree: {data.num_edges / data.num_nodes:.2f}")
    
    return data


def create_train_val_test_masks(num_nodes: int, 
                                 train_ratio: float = 0.7,
                                 val_ratio: float = 0.15) -> tuple:
    """
    Create train/validation/test masks for node-level prediction.
    """
    indices = np.random.permutation(num_nodes)
    
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    print(f"\nData split:")
    print(f"  Train: {train_mask.sum():,} ({train_ratio*100:.0f}%)")
    print(f"  Val:   {val_mask.sum():,} ({val_ratio*100:.0f}%)")
    print(f"  Test:  {test_mask.sum():,} ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    return train_mask, val_mask, test_mask


def insert_temporary_node(
    data: Data,
    new_features: np.ndarray,
    new_coordinates: np.ndarray,
    k: int = 5
) -> Data:
    """
    Insert a temporary node for real-time prediction.
    
    Finds k nearest neighbors in the existing graph and connects
    the new node to them.
    
    Args:
        data: Existing graph data
        new_features: Features for new node (1, F)
        new_coordinates: Lat/lng for new node (1, 2)
        k: Number of neighbors to connect to
        
    Returns:
        New Data object with temporary node added
    """
    # Get existing coordinates
    existing_coords = data.coordinates.numpy()
    
    # Find k nearest neighbors in existing graph
    knn = NearestNeighbors(n_neighbors=k, metric='haversine')
    knn.fit(np.radians(existing_coords))
    
    _, neighbor_indices = knn.kneighbors(np.radians(new_coordinates))
    neighbor_indices = neighbor_indices[0]
    
    # New node index
    new_idx = data.num_nodes
    
    # Create edges to neighbors
    new_edges_src = []
    new_edges_tgt = []
    for neighbor_idx in neighbor_indices:
        new_edges_src.extend([new_idx, neighbor_idx])
        new_edges_tgt.extend([neighbor_idx, new_idx])
    
    # Combine with existing edges
    new_edge_index = torch.cat([
        data.edge_index,
        torch.tensor([[new_edges_src], [new_edges_tgt]], dtype=torch.long).view(2, -1)
    ], dim=1)
    
    # Add new node features
    new_x = torch.cat([
        data.x,
        torch.tensor(new_features, dtype=torch.float32).view(1, -1)
    ], dim=0)
    
    # Create new Data object
    new_data = Data(
        x=new_x,
        edge_index=new_edge_index,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask
    )
    
    return new_data, new_idx


if __name__ == "__main__":
    # Test graph construction
    np.random.seed(42)
    
    # Generate random coordinates around Bangalore
    n_samples = 100
    coords = np.column_stack([
        np.random.uniform(12.8, 13.2, n_samples),  # Latitude
        np.random.uniform(77.4, 77.8, n_samples)   # Longitude
    ])
    
    # Random features and target
    features = np.random.randn(n_samples, 6)
    target = np.random.randn(n_samples)
    
    # Create masks
    train_mask, val_mask, test_mask = create_train_val_test_masks(n_samples)
    
    # Create graph
    data = create_graph_data(features, target, coords, train_mask, val_mask, test_mask, k=5)
    print(f"\nGraph data: {data}")
