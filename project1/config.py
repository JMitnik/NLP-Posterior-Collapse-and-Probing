from dataclasses import dataclass, field
from typing import List, Union

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
    results_path: str

    device: str

    # Run label
    run_label: str

    # RNN sizes (optionally 0)
    rnn_hidden_size: int = 0

    # Commands
    will_train_rnn: bool = False
    will_train_vae: bool = True

    # VAE sizes
    vae_encoder_hidden_size: int = 0
    vae_decoder_hidden_size: int = 0
    vae_latent_size: int = 0

    # Hyperparameters
    param_wdropout_k: Union[int, float, List[int] ,List[float]] = 1
    freebits_param:  Union[int, float, List[int] ,List[float]] = -1
    mu_force_beta_param: Union[int, List[int]] = 0

    # Training configurations
    # How often we validate our model, in iterations
    validate_every: int = 1000

    # How often we print our results
    print_every: int = 100

    # Multiple hyperparameters
