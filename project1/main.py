# %% [markdown]
# ## Imports
from dataclasses import asdict
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
from evaluations import generate_next_words
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
    validate_every=100,
    print_every=10,
    mu_force_beta_param=ARGS.mu_beta or [0, 2, 3, 5, 10],
    freebits_param=ARGS.freebits or [0, 0.125, 0.25, 0.5, 1, 2, 4, 8],
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
optim = torch.optim.Adam(rnn_lm.parameters(), lr=0.00001)

path_to_results = f'{config.results_path}/rnn'
rnn_results_writer = SummaryWriter()

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


# %%
###
### Evaluation of RNN
###
generated_sentence = generate_next_words(rnn_lm, cd, device=config.device)
print(cd.tokenizer.decode(generated_sentence))

# %%
import models.VAE
import importlib
importlib.reload(models.VAE)
###
###-------------------- START VAE -----------------------##
###
# Playing around with VAEs now
from models.VAE import VAE
import itertools
from functools import reduce
import operator

def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def flatten(L):
    for item in L:
        try:
            yield from flatten(item)
        except TypeError:
            yield item

potential_params = OrderedDict({
    'free_bits_param': np.hstack([config.freebits_param]),
    'param_wdropout_k': np.hstack([config.param_wdropout_k]),
    'mu_force_beta_param': np.hstack([config.mu_force_beta_param]),
})

# Add VAE to tensorboard
def make_sentence_decoder(tokenizer, temperature=1):
    def sentence_decoder(encoded_sentences):
        # If its an embedding (predictions)
        if len(encoded_sentences.shape) == 3:
            sentence = encoded_sentences[0]
            output_idxs = []

            for word in sentence:
                predicted_word_vector = F.softmax(word / temperature, 0)
                vector_sampler = torch.distributions.Categorical(predicted_word_vector)
                output_idxs.append(int(vector_sampler.sample()))

            return tokenizer.decode(output_idxs)

        # Else, its just the indices (targets)
        return tokenizer.decode(encoded_sentences[0])

    return sentence_decoder

param_grid = list(dict_product(potential_params))
param_grid

# %%
for param_setting in param_grid:
    vae = VAE(
        encoder_hidden_size=config.vae_encoder_hidden_size,
        decoder_hidden_size=config.vae_decoder_hidden_size,
        latent_size=config.vae_latent_size,
        vocab_size=config.vocab_size,
        param_wdropout_k=param_setting['param_wdropout_k'],
        embedding_size=config.embedding_size
    ).to(config.device)

    optimizer = torch.optim.Adam(params=vae.parameters())

    # Initalize results writer
    path_to_results = f'{config.results_path}/vae'
    params2string = '-'.join([f"{i}:{param_setting[i]}" for i in param_setting.keys()])
    vae_results_writer: SummaryWriter = SummaryWriter(comment=f"{config.run_label}--{params2string}")

    sentence_decoder = make_sentence_decoder(cd.tokenizer, 1)

    if config.will_train_vae:
        print(f"Training params: {params2string}")
        train_vae(
            vae,
            optimizer,
            train_loader,
            valid_loader,
            nr_epochs=config.nr_epochs,
            device=config.device,
            results_writer=vae_results_writer,
            config=config,
            decoder=sentence_decoder,
            freebits_param=param_setting['free_bits_param'],
            mu_force_beta_param=param_setting['mu_force_beta_param']
        )

    vae_results_writer.close()


# %%
