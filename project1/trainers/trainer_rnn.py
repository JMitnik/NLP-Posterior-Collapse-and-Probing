from tools.results_writer import ResultsWriter
from metrics import calc_batch_perplexity
from config import Config
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import pandas as pd
from models.VAE import VAE
from models.RNNLM import RNNLM
from inference.evaluate_rnn import evaluate_rnn
from utils import save_model
import numpy as np

def make_rnn_results_dict(nll_loss, perp, model, config: Config, epoch, it):
    return {
        'run_label': config.run_label,
        'model_name': type(model).__name__,
        'nll_loss': nll_loss,
        'perp_metric': perp,
        'epoch': epoch,
        'iteration': it
    }

def train_rnn(
    model: RNNLM,
    optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    config,
    nr_epochs: int,
    device: str,
    results_writer: ResultsWriter,
):
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=0,
        reduction='sum'
    )
    best_valid_loss = np.inf
    previous_valid_loss = np.inf

    for epoch in range(nr_epochs):
        print(f'Epoch: {epoch + 1} / {nr_epochs}')

        epoch_perp = 0
        epoch_loss = 0

        for idx, (batch, sentence_lengths) in enumerate(train_loader):
            model.train()
            batch_loss = train_batch_rnn(model, optimizer, loss_fn, batch, device)

            # Calculate average over batch-size
            avg_batch_perp = calc_batch_perplexity(batch_loss, sentence_lengths)
            avg_batch_loss = batch_loss / batch.shape[0]

            # Calculate epoch statistics
            epoch_perp += avg_batch_perp
            epoch_loss += avg_batch_loss

            it = epoch * len(train_loader) + idx

            # Report and print every X iterations
            if it % config.print_every == 0:
                print(f'Iteration: {it} || Loss: {avg_batch_loss} || Perplexity {avg_batch_perp}')
                results_dict = make_rnn_results_dict(avg_batch_loss, avg_batch_perp, model, config, epoch, it)
                results_writer.add_train_batch_results(results_dict)

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

def train_batch_rnn(model, optimizer, criterion, train_batch, device):
    """
    Trains a single batch for an RNNLM
    """
    optimizer.zero_grad()
    inp = train_batch[:, 0:-1].to(device)

    output = model(inp)
    target = train_batch[:, 1:].to(device)

    loss = criterion(output.reshape(-1, model.vocab_size), target.reshape(-1))

    loss.backward()
    optimizer.step()

    return loss
