import torch.nn as nn
import torch.nn.functional as F

class ML1Probe(nn.Module):
    def __init__(
        self,
        embedding_dim,
        out_dim
    ):
        super().__init__()
        self.h1 = nn.Linear(embedding_dim, 300)
        self.output = nn.Linear(300, out_dim)

    def forward(self, X):
        X = F.relu(self.h1(X))
        X = F.softmax(self.output(X))
        return X
