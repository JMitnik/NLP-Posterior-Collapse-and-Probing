from collections import OrderedDict
import torch
import torch.nn as nn

class RNNLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        hidden_size: int,
    ):
        super().__init__()

        self.vocab_size: int = vocab_size
        self.hidden_size: int = hidden_size
        self.embedding_size: int = embedding_size

        # Layers
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn: nn.GRU = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.fc1: nn.Linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor):
        """
        Given a sub-sentence, we predict the next word that follows.

        Arguments:
            x {torch.Tensor} -- Current word
            h {torch.Tensor} -- Previous hidden state

        Returns:
            {torch.Tensor} --
            {torch.Tensor} -- NextP
        """
        embeddings: torch.Tensor = self.embedding(x)

        # Pass embeddings through rnn
        states, last = self.rnn(embeddings)

        # TODO: Confirm if RELU must be put here

        # Classify by passing into fc-layer, and activate
        classification = self.fc1(states)

        # Return classification and hidden as state up til now
        return classification

    def init_hidden(self, batch):
        return torch.zeros(batch.shape[0], self.hidden_size)
