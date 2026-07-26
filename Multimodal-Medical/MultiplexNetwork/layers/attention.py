"""Learned fusion of multiplex relation embeddings."""

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.query = nn.Parameter(torch.empty(args.nb_graphs, args.hid_units))
        self.offset = nn.Parameter(torch.zeros(args.nb_graphs))
        nn.init.xavier_uniform_(self.query)

    @staticmethod
    def _stack(values):
        return torch.stack([value.squeeze(0) for value in values]) if isinstance(values, list) else values

    def _fuse_nodes(self, values):
        scores = torch.einsum("gnd,gd->gn", values, self.query) + self.offset[:, None]
        weights = scores.transpose(0, 1).softmax(dim=-1)
        fused = torch.einsum("gnd,ng->nd", values, weights).unsqueeze(0)
        return fused, weights

    def _fuse_summaries(self, values):
        scores = torch.einsum("gd,gd->g", values, self.query) + self.offset
        weights = scores.softmax(dim=0)
        return torch.einsum("gd,g->d", values, weights).unsqueeze(0)

    def forward(self, positive, negative, summaries):
        positive, negative, summaries = map(self._stack, (positive, negative, summaries))
        fused_positive, weights = self._fuse_nodes(positive)
        fused_negative, _ = self._fuse_nodes(negative)
        return fused_positive, fused_negative, self._fuse_summaries(summaries), weights.mean(dim=0)
