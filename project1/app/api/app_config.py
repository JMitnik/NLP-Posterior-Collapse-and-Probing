from config import Config
import torch

config = Config(
    run_label='',
    batch_size=64,
    vae_latent_size=16,
    embedding_size=256,
    rnn_hidden_size=256,
    vae_encoder_hidden_size=320,
    param_wdropout_k=[0, 0.5, 1],
    vae_decoder_hidden_size=320,
    vocab_size=10000,
    validate_every=50,
    print_every=10,
    mu_force_beta_param=[0, 2, 3, 5, 10],
    freebits_param=[-1, 0.25, 0.5, 1, 2, 8],
    will_train_rnn=False,
    will_train_vae=True,
    nr_epochs=5,
    results_path = 'results',
    train_path = '/data/02-21.10way.clean',
    valid_path = '/data/22.auto.clean',
    test_path  = '/data/23.auto.clean',
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
)
