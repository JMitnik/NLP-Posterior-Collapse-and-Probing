# %% [markdown]
# ## Imports
from tools.results_writer import ResultsWriter
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from trainers.trainer_rnn import train_rnn
from trainers.trainer_vae import train_vae

# Model-related imports
import torch
import torch.nn as nn
from utils import make_param_grid
from models.RNNLM import RNNLM
from models.VAE import VAE
from torch.utils.tensorboard.writer import SummaryWriter

import utils
from config import Config
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
    nr_epochs=ARGS.nr_epochs or 5,
    results_path = 'results',
    train_path = '/data/02-21.10way.clean',
    valid_path = '/data/22.auto.clean',
    test_path  = '/data/23.auto.clean',
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
)

print(f'Running on {config.device}.')

cd = CustomData(config)
train_loader = cd.get_data_loader(type="train", shuffle=True)
valid_loader = cd.get_data_loader(type='valid', shuffle=False)
test_loader = cd.get_data_loader(type='test', shuffle=False)

# %%
###
###-------------------- START RNN -----------------------##
###
# Define our model, optimizer and loss function
rnn_lm = RNNLM(config.vocab_size, config.embedding_size, config.rnn_hidden_size).to(config.device)
criterion = nn.CrossEntropyLoss(
    ignore_index=0,
    reduction='sum'
)
optim = torch.optim.Adam(rnn_lm.parameters(), lr=0.001)

path_to_results = f'{config.results_path}/rnn'
rnn_results_writer = ResultsWriter(label=f'{config.run_label}--rnn')

if config.will_train_rnn:
    train_rnn(
        rnn_lm,
        optim,
        train_loader,
        valid_loader,
        config=config,
        nr_epochs=config.nr_epochs,
        device=config.device,
        results_writer=rnn_results_writer
)
else:
    rnn_lm, optim, cp = utils.load_model('models/saved_models/rnn_best.pt', rnn_lm, config.device, optim)

rnn_results_writer.tensorboard_writer.close()

###
###-------------------- START VAE -----------------------##
###

potential_params = OrderedDict({
    'free_bits_param': np.hstack([config.freebits_param]),
    'param_wdropout_k': np.hstack([config.param_wdropout_k]),
    'mu_force_beta_param': np.hstack([config.mu_force_beta_param]),
})
param_grid = make_param_grid(potential_params)

# %%
for param_setting in param_grid:

    # Copy config, set new variables
    run_config = deepcopy(config)
    run_config.freebits_param = param_setting['free_bits_param']
    run_config.mu_force_beta_param = param_setting['mu_force_beta_param']
    run_config.param_wdropout_k = param_setting['param_wdropout_k']

    vae = VAE(
        encoder_hidden_size=run_config.vae_encoder_hidden_size,
        decoder_hidden_size=run_config.vae_decoder_hidden_size,
        latent_size=run_config.vae_latent_size,
        vocab_size=run_config.vocab_size,
        param_wdropout_k=run_config.param_wdropout_k,
        embedding_size=run_config.embedding_size
    ).to(run_config.device)

    optimizer = torch.optim.Adam(params=vae.parameters())

    # Initalize results writer
    path_to_results = f'{run_config.results_path}/vae'
    params2string = '-'.join([f"{i}:{param_setting[i]}" for i in param_setting.keys()])

    results_writer = ResultsWriter(
        label=f'{run_config.run_label}--vae-{params2string}',
    )

    sentence_decoder = utils.make_sentence_decoder(cd.tokenizer, 1)

    if run_config.will_train_vae:
        print(f"Training params: {params2string}")
        train_vae(
            vae,
            optimizer,
            train_loader,
            valid_loader,
            nr_epochs=run_config.nr_epochs,
            device=run_config.device,
            results_writer=results_writer,
            config=run_config,
            decoder=sentence_decoder,
        )

    results_writer.tensorboard_writer.close()

    print(f"Finished training for {params2string}!!!")


# %%
