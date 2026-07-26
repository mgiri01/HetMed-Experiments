"""Graph-level pooling helpers."""

from torch import nn


class AvgReadout(nn.Module):
    def forward(self, node_embeddings):
        return node_embeddings.mean(dim=1)
