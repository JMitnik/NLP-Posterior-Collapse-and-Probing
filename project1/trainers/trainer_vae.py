from tools.results_writer import ResultsWriter
from metrics import calc_batch_perplexity, calc_mu_loss
from config import Config
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from metrics import make_elbo_criterion
import pandas as pd
from models.VAE import VAE
from models.RNNLM import RNNLM
import torch
import os
from inference.evaluate_vae import evaluate_vae
from utils import save_model
import numpy as np

def make_vae_results_dict(losses, perp, model, config: Config, epoch, it):
    elbo_loss, kl_loss, nll_loss, mu_loss = losses

    return {
        'run_label': config.run_label,
        'model_name': type(model).__name__,
        'word_dropout': model.param_wdropout_k,
        'mu_force_beta_param': config.mu_force_beta_param,
        'freebits_param': config.freebits_param,
        'elbo_loss': elbo_loss,
        'kl_loss': kl_loss,
        'nll_loss': nll_loss,
        'mu_loss': mu_loss,
        'perp_metric': perp,
        'epoch': epoch,
        'iteration': it
    }

def train_vae(
    model: VAE,
    optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    nr_epochs: int,
    device: str,
    results_writer: ResultsWriter,
    config: Config,
    decoder,
):
    """
    Trains VAE, bases on a config file
    """

    # Define highest values
    best_valid_loss = np.inf
    previous_valid_loss = np.inf

    # Create elbo loss function
    loss_fn = make_elbo_criterion(
        vocab_size=model.vocab_size,
        latent_size=model.latent_size,
        freebits_param=config.freebits_param,
        mu_force_beta_param=config.mu_force_beta_param
    )

    for epoch in range(nr_epochs):
        for idx, (train_batch, batch_sent_lengths) in enumerate(train_loader):
            it = epoch * len(train_loader) + idx

            batch_loss, preds = train_batch_vae(
                model,
                optimizer,
                loss_fn,
                train_batch,
                device,
                config.mu_force_beta_param,
                results_writer,
                it
            )
            elbo_loss, kl_loss, nlll, mu_loss = batch_loss

            # Calculate metrics
            perp = calc_batch_perplexity(elbo_loss, batch_sent_lengths)

            if idx % config.print_every == 0:
                print(f'Iteration: {it} || NLLL: {nlll} || Perp: {perp} || KL Loss: {kl_loss} || MuLoss: {mu_loss} || Total: {elbo_loss}')
                decoded_first_pred = decoder(preds)
                decoded_first_true = decoder(train_batch[:, 1:])
                results_writer.add_sentence_predictions(decoded_first_pred, decoded_first_true, it)

                # Store in the table
                train_vae_results = make_vae_results_dict(batch_loss, perp, model, config, epoch, it)
                results_writer.add_train_batch_results(train_vae_results)

            if idx % config.validate_every == 0 and it != 0:
                print('Validating model')
                valid_losses, valid_perp = evaluate_vae(model,
                    valid_loader,
                    epoch,
                    device,
                    loss_fn,
                    config.mu_force_beta_param,
                    results_writer,
                    iteration = it
                )

                # Store validation results
                valid_vae_results = make_vae_results_dict(valid_losses, valid_perp, model, config, epoch, it)
                results_writer.add_valid_results(valid_vae_results)

                valid_elbo_loss, valid_kl_loss, valid_nll_loss, valid_mu_loss = valid_losses
                print(f'Validation Results || Elbo loss: {valid_elbo_loss} || KL loss: {valid_kl_loss} || NLLL {valid_nll_loss} || Perp: {valid_perp} ||MU loss {valid_mu_loss}')

                # Check if the model is better and save
                previous_valid_loss = valid_elbo_loss
                if previous_valid_loss < best_valid_loss:
                    print(f'New Best Validation score of {previous_valid_loss}!')
                    best_valid_loss = previous_valid_loss
                    save_model(f'vae_best_mu{config.mu_force_beta_param}_wd{model.param_wdropout_k}_fb{config.freebits_param}', model, optimizer, it)

                model.train()

    results_writer.save_train_results()
    results_writer.save_valid_results()

    print('Done training the VAE')


def train_batch_vae(model, optimizer, criterion, train_batch, device, mu_force_beta_param, writer: ResultsWriter, it):
    """
    Trains single batch of VAE
    """
    optimizer.zero_grad()

    # Current assumption: to not predict past final token, we dont include the EOS tag in the input
    inp = train_batch[:, 0:-1].to(device)

    # Creat both prediction of next word and the posterior of which we sample Z.
    preds, posterior = model(inp)

    # Define target as the next word to predict
    target = train_batch[:, 1:].to(device)

    # Calc loss by using the ELBO-criterion
    loss, kl_loss, nlll = criterion(
        preds,
        target,
        posterior
    )

    # Take mean of mini-batch loss
    loss = loss.mean()
    kl_loss = kl_loss.mean()
    nlll = nlll.mean()

    # Add Mu density plot to tensorboard
    flattened_posterior = posterior.loc.flatten()
    writer.tensorboard_writer.add_histogram('train-vae/mu', flattened_posterior, it)

    # Now calc loss
    mu_force_loss = calc_mu_loss(posterior, train_batch.shape[0], mu_force_beta_param).to(device)
    loss = loss + mu_force_loss

    # Backprop and gradient descent
    loss.backward()
    optimizer.step()

    # Define all losses for reporting
    losses = loss.item(), kl_loss.item(), nlll.item(), mu_force_loss.item()

    return losses, preds
