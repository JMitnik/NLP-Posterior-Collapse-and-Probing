from dataclasses import dataclass

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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

    # Hyperparameters
    param_wdropout_k: int = 1

    def __post_init__(self):
        if self.param_wdropout_k < 1:
            print(f"{bcolors.WARNING}❗Word-dropout active, is set to {self.param_wdropout_k} {bcolors.ENDC}")
