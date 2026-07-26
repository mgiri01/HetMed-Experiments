"""PyTorch Geometric graph-convolution adapter.

The implementation uses :class:`torch_geometric.nn.GCNConv`, distributed by
the PyG Team under the MIT License. See ``PYG_LICENSE`` in this directory.
"""

import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Apply a PyG GCN convolution to the project's single-graph batches."""

    def __init__(self, in_ft, out_ft, act="relu", drop_prob=0.0, isBias=False):
        super().__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.conv = GCNConv(in_ft, out_ft, bias=isBias, add_self_loops=False)
        activations = {
            "relu": nn.ReLU,
            "prelu": nn.PReLU,
            "leakyrelu": nn.LeakyReLU,
            "relu6": nn.ReLU6,
            "rrelu": nn.RReLU,
            "selu": nn.SELU,
            "celu": nn.CELU,
            "sigmoid": nn.Sigmoid,
            "identity": nn.Identity,
        }
        if act not in activations:
            raise ValueError(f"Unsupported activation: {act}")
        self.activation = activations[act]()

    def forward(self, features, adjacency, sparse=True):
        del sparse
        if features.ndim != 3 or features.size(0) != 1:
            raise ValueError("GCN expects features shaped [1, nodes, channels]")
        adjacency = adjacency.coalesce()
        encoded = self.conv(
            self.dropout(features.squeeze(0)),
            adjacency.indices(),
            adjacency.values(),
        )
        return self.activation(encoded).unsqueeze(0)
