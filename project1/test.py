# %% [markdown]
# ## Imports
from dataclasses import asdict
from losses import make_elbo_criterion
from models.VAE import VAE
import os
import numpy as np
import importlib
from collections import OrderedDict
from operator import iconcat
from trainers import train_rnn, train_vae
from nltk import Tree
from nltk.treeprettyprinter import TreePrettyPrinter

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
from evaluations import evaluate_VAE, evaluate_rnn
from customdata import CustomData

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
    batch_size=16,
    vae_latent_size=16,
    embedding_size=384,
    rnn_hidden_size=192,
    vae_encoder_hidden_size=320,
    param_wdropout_k=ARGS.wdropout_k or chosen_wdropout_params,
    vae_decoder_hidden_size=320,
    vocab_size=10000,
    validate_every=1000,
    print_every=500,
    freebits_param=chosen_freebits_params,
    mu_force_beta_param=ARGS.mu_beta or chosen_mu_force_beta_params,
    will_train_rnn=False,
    will_train_vae=True,
    nr_epochs=ARGS.nr_epochs or 5,
    results_path = 'results',
    train_path = '/data/02-21.10way.clean',
    valid_path = '/data/22.auto.clean',
    test_path  = '/data/23.auto.clean',
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
)

cd = CustomData(config)
test_loader = cd.get_data_loader(type='test', shuffle=False)

loss_fn = make_elbo_criterion(config.vocab_size, -1, 0)

# %%
best_saved_models = ['vae_best_mu5_wd1.0_fb-1.pt']

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

        prior = torch.distributions.Normal(
            torch.zeros(model.latent_size),
            torch.ones(model.latent_size)
        )

        path_to_model = f'models/saved_models/{model_path}'
        model, _, _ = utils.load_model(path_to_model, model, config.device)
        model = model.to(config.device)
        vae_results_writer: SummaryWriter = SummaryWriter(comment=f"EVAL_{config.run_label}--{model_path}")
        evaluate_VAE(model, test_loader, -1, config.device, loss_fn, 0, prior, vae_results_writer, 'test')
