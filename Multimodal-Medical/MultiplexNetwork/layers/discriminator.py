"""Bilinear discriminator for relation-level contrastive learning."""

import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.score = nn.Bilinear(hidden_channels, hidden_channels, 1)
        nn.init.xavier_uniform_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, summary, positive, negative, positive_bias=None, negative_bias=None):
        context = summary.unsqueeze(1).expand_as(positive)
        positive_scores = self.score(positive, context).squeeze(-1)
        negative_scores = self.score(negative, context).squeeze(-1)
        if positive_bias is not None:
            positive_scores = positive_scores + positive_bias
        if negative_bias is not None:
            negative_scores = negative_scores + negative_bias
        return torch.cat((positive_scores, negative_scores), dim=1)
