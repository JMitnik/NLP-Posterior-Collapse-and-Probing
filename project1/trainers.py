from config import Config
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from losses import make_elbo_criterion
import pandas as pd
from models.VAE import VAE
from models.RNNLM import RNNLM
import torch
import os
from evaluations import evaluate_VAE, evaluate_rnn
from utils import save_model
import numpy as np

def train_batch_rnn(model, optimizer, criterion, train_batch, device):
    inp = train_batch[:, 0:-1].to(device)

    # Current assumption: to not predikct past final token, we dont include the EOS tag in the input
    output = model(inp)

    # One hot target and shift by one (so first word matches second token)
    # target = torch.nn.functional.one_hot(input_tensor, num_classes=vocab_size)
    target = train_batch[:, 1:].to(device)

    # Calc loss and perform backprop
    loss = criterion(output.reshape(-1, model.vocab_size), target.reshape(-1))
    loss.backward()
    optimizer.step()

    return loss


def train_rnn(
    model: RNNLM,
    optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config,
    nr_epochs: int,
    device: str,
    results_writer: SummaryWriter,
):
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=0,
        reduction='sum'
    )
    best_valid_loss = 1000
    previous_valid_loss = 1000
    for epoch in range(nr_epochs):
        print(f'Epoch: {epoch + 1} / {nr_epochs}')
        
        epoch_perp = 0
        epoch_loss = 0

        for idx, (batch, sl) in enumerate(train_loader):
            model.train()

            loss = train_batch_rnn(model, optimizer, loss_fn, batch, device)
        
            all_words = torch.sum(sl).item()
            perplexity = np.exp(loss.item() / all_words) / batch.size(0)

            loss = loss / batch.shape[0]
          
            
            epoch_perp += perplexity
            epoch_loss += loss

            it = epoch * len(train_loader) + idx
            
            # print and log every 50 iterations
            if it % config.print_every == 0:
                print(f'Iteration: {it} || Loss: {loss} || Perplexity {perplexity}')
                results_writer.add_scalar('train-rnn/loss', loss, it)
                results_writer.add_scalar('train-rnn/ppl', perplexity, it)
            
            # Save the most recent model
            save_model('rnn_recent', model, optimizer, it)
            # Validate the model and save if the model is better
            if it % config.validate_every == 0 and it != 0:
                print("Validating model")
                valid_loss, valid_perp = evaluate_rnn(model, valid_loader, it, device, loss_fn, results_writer, it=it)
                previous_valid_loss = valid_loss

                if previous_valid_loss < best_valid_loss:
                    print('New Best Validation score!')
                    best_valid_loss = previous_valid_loss
                    save_model('rnn_best', model, optimizer, it)

                print(f'Validation results || Loss: {valid_loss} || Perplexity {valid_perp}')
                print()
                model.train()
 
        print('\n\n')
        epoch_perp = epoch_perp / len(train_loader)
        epoch_loss = epoch_loss / len(train_loader)
        print(f'The Perplexity of Epoch {epoch+1}: {epoch_perp}')
        print(f'The Loss of Epoch {epoch+1}: {epoch_loss}')
        print('\n\n')
    print("Done with training!")


def train_batch_vae(model, optimizer, criterion, train_batch, prior, device, mu_force_beta_param, writer, it):
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
        prior,
        posterior
    )

    # Take mean of mini-batch loss
    loss = loss.mean()
    kl_loss = kl_loss.mean()
    nlll = nlll.mean()
    flattend_post = posterior.loc.flatten()
 
    writer.add_histogram('train-vae/mu', flattend_post, it)

    # Now add to the loss mu force loss
    batch_mean_vectors = posterior.loc # mu(n) mu vector of nth sample
    avg_batch_mean_vector = batch_mean_vectors.mean(0) # mu(stripe) mean of vectors mu
    # Shouldn't it be tensordot.... / (batch.shape[0] * 2) ??
    mu_force_loss_var = torch.tensordot(batch_mean_vectors - avg_batch_mean_vector, batch_mean_vectors - avg_batch_mean_vector, 2) / train_batch.shape[0] / 2
    mu_force_loss = torch.max(torch.tensor([0.0]), mu_force_beta_param - mu_force_loss_var).to(device)

    loss = loss + mu_force_loss

    # Backprop and gradient descent
    loss.backward()
    optimizer.step()

    return (loss.item(), kl_loss.item(), nlll.item(), mu_force_loss_var.item()), preds

def train_vae(
    model: VAE,
    optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    nr_epochs: int,
    device: str,
    results_writer: SummaryWriter,
    config: Config,
    decoder,
    freebits_param=-1,
    mu_force_beta_param=1,
):
    best_valid_loss = 1000
    previous_valid_loss = 1000

    vocab_size = model.vocab_size
    loss_fn = make_elbo_criterion(vocab_size, freebits_param, mu_force_beta_param)

    prior = torch.distributions.Normal(
        torch.zeros(model.latent_size),
        torch.ones(model.latent_size)
    )

    train_results = pd.DataFrame(columns=['model_name', 'word', 'word_dropout'
    'mu_force_beta_param', 'freebits_param', 'elbo', 'kl', 'nll',
    'mu', 'ppl', 'epoch', 'iteration',])

    valid_results = pd.DataFrame(columns=['model_name', 'word', 'word_dropout'
    'mu_force_beta_param', 'freebits_param', 'elbo', 'kl', 'nll',
    'mu', 'ppl', 'epoch', 'iteration', 'best_model'])

    train_results_filename = 'results/vae_training.csv'
    valid_results_filename = 'results/vae_valid_scores.csv'

    for epoch in range(nr_epochs):
        print (epoch)

        for idx, train_batch in enumerate(train_loader):
            it = epoch * len(train_loader) + idx

            loss, preds = train_batch_vae(
                model,
                optimizer,
                loss_fn,
                train_batch,
                prior,
                device,
                mu_force_beta_param,
                results_writer,
                it
            )

            loss, kl_loss, nlll, mu_loss = loss
            sentence_length = train_batch[0].size()[0]
            perplexity = np.exp(loss /sentence_length)
            # print and log every 50 iterations
            if idx % config.print_every == 0:
                print(f'Iteration: {it} || NLLL: {nlll} || Perp: {perplexity} || KL Loss: {kl_loss} || MuLoss: {mu_loss} || Total: {loss}')
                results_writer.add_scalar('train-vae/elbo-loss', loss, it)
                results_writer.add_scalar('train-vae/ppl', perplexity, it)
                results_writer.add_scalar('train-vae/kl-loss', kl_loss, it)
                results_writer.add_scalar('train-vae/nll-loss', nlll, it)
                results_writer.add_scalar('train-vae/mu-loss', mu_loss, it)

            # Every 50 iterations, predict a sentence and check the truth
            if idx % config.print_every == 0:
                decoded_first_pred = decoder(preds) # TODO: Cutt-off after sentence length?
                decoded_first_true = decoder(train_batch[:, 1:])
                results_writer.add_text(f'it {it}: prediction', decoded_first_pred)
                results_writer.add_text(f'it {it}: truth', decoded_first_true)

                # # Store in the table
                train_results = train_results.append({
                    'run_label': config.run_label,
                    'model_name': type(model).__name__,
                    'word_dropout': model.param_wdropout_k,
                    'mu_force_beta_param': mu_force_beta_param,
                    'freebits_param': freebits_param,
                    'elbo': loss,
                    'kl': kl_loss,
                    'nll-loss': nlll,
                    'mu-loss': mu_loss,
                    'ppl': torch.log(torch.tensor(loss)),
                    'epoch': epoch,
                    'iteration': it
                }, ignore_index=True)

            if idx % config.validate_every == 0 and it != 0:
                print('Validating model')
                valid_total_loss, valid_total_kl_loss, valid_total_nlll, valid_perp, valid_total_mu_loss = evaluate_VAE(model,
                    valid_loader,
                    epoch,
                    device,
                    loss_fn,
                    mu_force_beta_param,
                    prior,
                    results_writer,
                    iteration = epoch * len(train_loader) + idx
                )

                # # Store in the table
                valid_results = valid_results.append({
                    'run_label': config.run_label,
                    'model_name': type(model).__name__,
                    'word_dropout': model.param_wdropout_k,
                    'mu_force_beta_param': mu_force_beta_param,
                    'freebits_param': freebits_param,
                    'elbo': valid_total_loss,
                    'kl': valid_total_kl_loss,
                    'nll-loss': valid_total_nlll,
                    'mu-loss': valid_total_mu_loss,
                    'ppl': torch.log(torch.tensor(loss)),
                    'epoch': epoch,
                    'iteration': it
                }, ignore_index=True)

                previous_valid_loss = valid_total_loss

                # Check if the model is better and save
                if previous_valid_loss < best_valid_loss:
                    print('New Best Validation score!')
                    best_valid_loss = previous_valid_loss
                    save_model(f'vae_best_mu{mu_force_beta_param}_wd{model.param_wdropout_k}_fb{freebits_param}', model, optimizer, it)
                    print(f'Validation Results || Elbo loss: {valid_total_loss} || KL loss: {valid_total_kl_loss} || NLLL {valid_total_nlll} || Perp: {valid_perp} ||MU loss {valid_total_mu_loss}')
                    print()
                model.train()

    # Save train
    if not os.path.exists(train_results_filename):
        os.makedirs(os.path.dirname(train_results_filename), exist_ok=True)
        train_results.to_csv(train_results_filename)
    else:
        train_results.to_csv(train_results_filename, mode='a', header=False)

    # Save valid
    if not os.path.exists(valid_results_filename):
        os.makedirs(os.path.dirname(valid_results_filename), exist_ok=True)
        valid_results.to_csv(valid_results_filename)
    else:
        valid_results.to_csv(valid_results_filename, mode='a', header=False)

    print('Done training the VAE')
