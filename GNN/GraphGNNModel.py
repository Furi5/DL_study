import torch_geometric.nn as geom_nn
from torch import nn

from GNN.GNNModel import GNNModel


class GraphGNNModel(nn.Module):

    def __init__(self, D_in: int, D_hidden: int, D_out: int, dp_rate_linear: float = 0.5, device: str = 'cpu',
                 **kwargs):
        """
        Graph GCN model

        Args:
            c_in (int): Dimension of input features
            c_hidden (int): Dimension of hidden features
            c_out (int): Dimension of output features (usually number of classes)
            dp_rate_linear (float): Dropout rate before the linear layer (usually much higher than inside the GNN)
            device (str)
            **kwargs: Additional arguments for the GNNModel object
        """

        super().__init__()
        self.GNN = GNNModel(D_in=D_in,
                                   D_hidden=D_hidden,
                                   D_out=D_hidden,  # Not our prediction output yet!
                                   **kwargs).to(device)
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(D_hidden, D_out)
        )

    def forward(self, x, edge_index, batch_idx):
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            batch_idx: Index of batch element for each node

        Returns:
            x: Output
        """
        x = self.GNN(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx)  # Average pooling
        x = self.head(x)
        return x