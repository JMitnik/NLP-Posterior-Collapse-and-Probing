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
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        latent_size: int,
        vocab_size: int,
        embedding_size: int,
        param_wdropout_k: float = 1,
        token_unknown_index: int = 0
    ):
        super().__init__()
        self.latent_size: int = latent_size
        self.vocab_size: int = vocab_size
        self.embedding_size: int = embedding_size

        self.embeddings: nn.Embedding = nn.Embedding(vocab_size, embedding_size)

        self.encoder: Encoder = Encoder(vocab_size, embedding_size, encoder_hidden_size, latent_size)
        self.decoder: Decoder = Decoder(vocab_size, embedding_size, latent_size, decoder_hidden_size)

        self.param_wdropout_k: float = param_wdropout_k
        self.token_unknown_index: int = token_unknown_index

        self.graph_mode: bool = False

    def make_distribution(self, mu, sigma):
        return torch.distributions.Normal(mu, sigma)

    def apply_word_dropout(self, input_seq):
        # Make bernoulli distribution that takes k as its probability

        bern = torch.distributions.Bernoulli(torch.tensor([self.param_wdropout_k]))

        # Sample mask that is (batch_size x nr_words)
        mask = bern.sample(torch.tensor([
            input_seq.shape[0],
            input_seq.shape[1]
        ])).to(input_seq.device)

        # Apply mask
        # TODO: Do we need to somehow prevent the gradient from being set?
        return input_seq * mask

    def forward(self, x, nr_multi_sample=1):
        embeds = self.embeddings(x)
        mu, sigma = self.encoder(embeds)

        batch_size = x.shape[0]

        # Sample latent variable
        distribution: torch.distributions.Distribution = self.make_distribution(
            mu.clone().cpu(),
            sigma.clone().cpu()
        )

        # Apply some dropout here
        if self.training and self.param_wdropout_k < 1:
            """
            Only problem still is the the BOS will also be turned
            into an UNK token sometimes
            """
            mask = torch.full_like(x, fill_value=self.param_wdropout_k, dtype=float)
            mask = torch.distributions.Bernoulli(mask)
            mask = mask.sample().eq(1)
            pad_mask = x.ne(0)
            indexes = pad_mask.__and__(mask) # Compare two vector and do an logical AND operator
            masked_x = x.masked_fill(indexes, value=3) # Replace all value for 3(UNK) when true
            x = masked_x
            embeds = self.embeddings(x)

        if nr_multi_sample == 1:
            z = distribution.rsample().to(x.device)
        else:
            z = distribution.rsample(torch.tensor([nr_multi_sample])).reshape((nr_multi_sample * batch_size, -1)).to(x.device)

        pred = self.decoder(embeds, z, nr_multi_sample)

        if self.graph_mode:
            return pred

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

    def forward(self, x, z, nr_multi_sample = 1):
        # If so, copy x to be `nr_multi_sample` to match
        if nr_multi_sample > 1:
            x = x.unsqueeze(1).repeat(1, nr_multi_sample, 1, 1).flatten(0, 1)

        global_hidden = self.tanh(self.latent2hidden(z))
        states, _ = self.rnn(x, global_hidden.view(1, global_hidden.shape[0], -1))

        out = self.hidden2out(states)
        return out
