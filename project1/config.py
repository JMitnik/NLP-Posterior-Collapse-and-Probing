from dataclasses import dataclass

@dataclass
class Config:
    # Sizes in general
    batch_size: int
    embedding_size: int
    vocab_size: int
    nr_epochs: int

    # Paths
    train_path: str
    valid_path: str
    test_path: str
    device: str

    # RNN sizes (optionally 0)
    rnn_hidden_size: int = 0

    # VAE sizes
    vae_encoder_hidden_size: int = 0
    vae_decoder_hidden_size: int = 0
    vae_latent_size: int = 0
