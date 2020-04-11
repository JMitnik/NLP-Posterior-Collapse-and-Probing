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

        self.embeddings = nn.Embedding(vocab_size, embedding_size)

        self.encoder: Encoder = Encoder(vocab_size, embedding_size, encoder_hidden_size, latent_size)
        self.decoder: Decoder = Decoder(vocab_size, embedding_size, latent_size, decoder_hidden_size)

    def make_distribution(self, mu, sigma):
        return torch.distributions.Normal(mu, sigma)

    def forward(self, x):
        embeds = self.embeddings(x)
        mu, sigma = self.encoder(embeds)

        # Sample latent variable
        distribution: torch.distributions.Distribution = self.make_distribution(
            mu.clone().cpu(),
            sigma.clone().cpu()
        )

        # Send to same device as where the input is
        z = distribution.rsample().to(x.device)

        pred = self.decoder(embeds, z)

        return pred, distribution

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

        self.rnn = nn.GRU(embedding_size, hidden_size, bidirectional=True, batch_first=True)

        self.hidden2mu = nn.Linear(self.hidden_size, latent_size)
        self.hidden2sigma = nn.Linear(self.hidden_size, latent_size)

        self.softplus = nn.Softplus()

    def forward(self, x):
        _, final_hidden = self.rnn(x)

        # Concat final hidden state from forward (0) and backward (1)
        hidden = torch.cat((final_hidden[0, :, :], final_hidden[1, :, :]), dim=1)

        mu = self.hidden2mu(hidden)
        sigma = self.softplus(self.hidden2sigma(hidden))

        return mu, sigma

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

        self.rnn = nn.GRU(self.embedding_size, self.hidden_size, batch_first=True)
        self.latent2hidden = nn.Linear(latent_size, hidden_size)

        self.hidden2out = nn.Linear(hidden_size, vocab_size)

        self.tanh = nn.Tanh()

    def forward(self, x, z):
        global_hidden = self.tanh(self.latent2hidden(z))
        states, _ = self.rnn(x, global_hidden.view(1, global_hidden.shape[0], -1))

        out = self.hidden2out(states)
        return out
