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
    param_wdropout_k: float = 1
    freebits_param: int = -1
    mu_force_beta_param: int = 0 # Possible values should be [0, 2, 3, 5, 10]

    def __post_init__(self):
        if self.param_wdropout_k < 1:
            print(f"{bcolors.WARNING}❗Word-dropout active, is set to {self.param_wdropout_k} {bcolors.ENDC}")

        if self.freebits_param > -1:
            print(f"{bcolors.WARNING}❗Freebits active, is set to {self.freebits_param} {bcolors.ENDC}")

        if self.mu_force_beta_param > 0:
            print(f"{bcolors.WARNING}❗Mu-force beta param active, is set to {self.mu_force_beta_param} {bcolors.ENDC}")
