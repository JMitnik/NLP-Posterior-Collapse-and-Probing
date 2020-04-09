import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VAE(nn.Module):
    """
    Our VAE
    """
    def __init__(
        self,
        encoder_hidden_size,
        decoder_hidden_size,
        latent_size,
        vocab_size,
        embedding_size
    ):
        super().__init__()
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)


        self.encoder: Encoder = Encoder(vocab_size, embedding_size, encoder_hidden_size, latent_size)
        self.decoder: Decoder = Decoder(vocab_size, embedding_size, latent_size, decoder_hidden_size)

    def forward(self, x):
        z, distribution = self.encoder(x)
        reconstruction = self.decoder(x, z)

        return reconstruction, distribution

class Encoder(nn.Module):
    """
    Encoder
    """
    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        latent_size
    ):
        super().__init__()

        # Times two because we encapsulate bidirectionality this way
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size * 2
        self.latent_size = latent_size

        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, bidirectional=True, batch_first=True)

        self.hidden2mu = nn.Linear(self.hidden_size, latent_size)
        self.hidden2sigma = nn.Linear(self.hidden_size, latent_size)

        self.softplus = nn.Softplus()

    def make_distribution(self, mu, sigma):
        # TODO: Fix this
        return torch.distributions.MultivariateNormal(mu, sigma)

    def forward(self, x):
        embeddings: torch.Tensor = self.embeddings(x)
        _, final_hidden = self.rnn(embeddings)

        # Concat final hidden state from forward (0) and backward (1)
        hidden = torch.cat((final_hidden[0, :, :], final_hidden[1, :, :]), dim=1)

        mu = self.hidden2mu(hidden)
        sigma = self.softplus(self.hidden2sigma(hidden))

        distribution: torch.distributions.Distribution = self.make_distribution(mu, sigma)
        z = distribution.rsample((x.shape[0], self.latent_size))

        return z, distribution

class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        latent_size: int,
        hidden_size: int,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.embedding: nn.Embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)
        self.latent2hidden = nn.Linear(latent_size, hidden_size)

        self.hidden2out = nn.Linear(hidden_size, vocab_size)

        self.tanh = nn.Tanh()

    def forward(self, x, z):
        global_hidden = self.tanh(self.latent2hidden(z))
        states, _ = self.rnn(x, global_hidden)

        out = self.hidden2out(states)
        return out
