import torch.nn as nn
import torch.nn.functional as F

class SimpleProbe(nn.Module):
    def __init__(
        self,
        embedding_dim,
        out_dim
    ):
        super().__init__()
        self.h2out = nn.Linear(embedding_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        logits = self.h2out(X)

        return self.softmax(logits)
