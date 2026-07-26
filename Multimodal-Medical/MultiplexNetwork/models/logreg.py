"""Linear node classifier used for embedding evaluation."""

from torch import nn


class LogReg(nn.Module):
    def __init__(self, input_channels, classes):
        super().__init__()
        self.classifier = nn.Linear(input_channels, classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, embeddings):
        return self.classifier(embeddings)
