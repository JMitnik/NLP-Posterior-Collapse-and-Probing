# %% [markdown]
# ## Imports

from dataclasses import asdict
import os
import importlib
from trainers import train_rnn, train_vae
from nltk import Tree
from nltk.treeprettyprinter import TreePrettyPrinter

# Model-related imports
import torch
import torch.nn as nn
import torch.functional as F
import models.RNNLM
importlib.reload(models.RNNLM)
from models.RNNLM import RNNLM

import utils
importlib.reload(utils)

from config import Config
from torch.utils.tensorboard.writer import SummaryWriter
from evaluations import generate_next_words
from customdata import CustomData

# %%
###
### Data definition
###

config = Config(
    batch_size=16,
    embedding_size=50,
    rnn_hidden_size=50,
    vae_encoder_hidden_size=128,
    vae_decoder_hidden_size=1281,
    param_wdropout_k=0.5,
    vae_latent_size=128,
    vocab_size=10000,
    will_train_rnn=False,
    will_train_vae=True,
    nr_epochs=1,
    results_path = '/results/',
    train_path = '/data/02-21.10way.clean',
    valid_path = '/data/22.auto.clean',
    test_path  = '/data/23.auto.clean',
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
)

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
optim = torch.optim.Adam(rnn_lm.parameters())

path_to_results = f'{config.results_path}/rnn'
rnn_results_writer = SummaryWriter()

if config.will_train_rnn:
    train_rnn(
        rnn_lm,
        optim,
        train_loader,
        config.nr_epochs,
        config.device,
        rnn_results_writer
)


# %%
###
### Evaluation of RNN
###
generated_sentence = generate_next_words(rnn_lm, cd, device=config.device)
print(cd.tokenizer.decode(generated_sentence))

# %%
###
###-------------------- START VAE -----------------------##
###
# Playing around with VAEs now
from models.VAE import VAE
vae = VAE(
    encoder_hidden_size=config.vae_encoder_hidden_size,
    decoder_hidden_size=config.vae_decoder_hidden_size,
    latent_size=config.vae_latent_size,
    vocab_size=config.vocab_size,
    param_wdropout_k=config.param_wdropout_k,
    embedding_size=config.embedding_size
).to(config.device)

optimizer = torch.optim.Adam(params=vae.parameters())

# Initalize results writer
path_to_results = f'{config.results_path}/vae'
vae_results_writer: SummaryWriter = SummaryWriter(path_to_results)
vae_results_writer.add_graph(vae)

if config.will_train_vae:
    train_vae(
        vae,
        optimizer,
        train_loader,
        nr_epochs=config.nr_epochs,
        device=config.device,
        results_writer=vae_results_writer,
        freebits_param=config.freebits_param
)
