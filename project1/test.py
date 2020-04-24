# %% [markdown]
# ## Imports
from metrics import make_elbo_criterion
from models.VAE import VAE
import importlib

# Model-related imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.RNNLM
importlib.reload(models.RNNLM)
from models.RNNLM import RNNLM

import utils
importlib.reload(utils)

from config import Config
from torch.utils.tensorboard.writer import SummaryWriter
from inference.evaluate_vae import evaluate_vae
from tools.customdata import CustomData

#%%
import argparse

parser = argparse.ArgumentParser()

# Parse arguments
parser.add_argument('--run_label', type=str, help='label for run')
parser.add_argument('--nr_epochs', type=int, help='nr epochs to run for')
parser.add_argument('--wdropout_k', type=float, help='dropout to apply')
parser.add_argument('--mu_beta', type=float, help='mu_beta parameter')
parser.add_argument('--freebits', type=float, help='freebits to apply')
parser.add_argument('-f', type=str, help='Path to kernel json')

# Extract args
ARGS, unknown = parser.parse_known_args()

# %%
###
### Data definition
###
chosen_wdropout_params = [0, 0.5, 1]
chosen_mu_force_beta_params = [0, 2, 3, 5, 10]
chosen_freebits_params = [-1]


config = Config(
    run_label=ARGS.run_label or '',
    batch_size=64,
    vae_latent_size=16,
    embedding_size=256,
    rnn_hidden_size=256,
    vae_encoder_hidden_size=320,
    param_wdropout_k=ARGS.wdropout_k or [0, 0.5, 1],
    vae_decoder_hidden_size=320,
    vocab_size=10000,
    validate_every=50,
    print_every=10,
    mu_force_beta_param=ARGS.mu_beta or [0, 2, 3, 5, 10],
    freebits_param=ARGS.freebits or [-1, 0.25, 0.5, 1, 2, 8],
    will_train_rnn=False,
    will_train_vae=True,
    nr_epochs=ARGS.nr_epochs or 20,
    results_path = 'results',
    train_path = '/data/02-21.10way.clean',
    valid_path = '/data/22.auto.clean',
    test_path  = '/data/23.auto.clean',
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
)

cd = CustomData(config)
test_loader = cd.get_data_loader(type='test', shuffle=False)
# %%
best_saved_models = [
    'vae_best_mu5_wd1_fb-1.pt',
    'vae_best_mu0_wd1_fb-1.pt',
    'vae_best_mu0_wd1_fb0.pt',
    'vae_best_mu3_wd1_fb0.25.pt',
    # 'vae_best_mu5_wd1.0_fb0.pt',
]

for model_path in best_saved_models:
    if 'vae' in model_path:
        model = VAE(
            encoder_hidden_size=config.vae_encoder_hidden_size,
            decoder_hidden_size=config.vae_decoder_hidden_size,
            latent_size=config.vae_latent_size,
            vocab_size=config.vocab_size,
            param_wdropout_k=-1,
            embedding_size=config.embedding_size
        ).to(config.device)

        loss_fn = make_elbo_criterion(config.vocab_size, config.vae_latent_size, -1, 0)


        prior = torch.distributions.Normal(
            torch.zeros(model.latent_size),
            torch.ones(model.latent_size)
        )

        path_to_model = f'results/saved_models/{model_path}'
        model, _, _ = utils.load_model(path_to_model, model, config.device)
        model = model.to(config.device)
        vae_results_writer: SummaryWriter = SummaryWriter(comment=f"EVAL_{config.run_label}--{model_path}")
        (test_total_loss, test_total_kl_loss, test_total_nlll, test_total_mu_loss), test_perp = evaluate_vae(
            model,
            test_loader,
            -1,
            config.device,
            loss_fn,
            0,
            vae_results_writer,
            'test'
        )

        print(f'For model {model_path}: \n')
        print(f'Test Results || Elbo loss: {test_total_loss} || KL loss: {test_total_kl_loss} || NLLL {test_total_nlll} || Perp: {test_perp} ||MU loss {test_total_mu_loss}')

