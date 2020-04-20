from config import Config
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from losses import make_elbo_criterion
from models.VAE import VAE
from models.RNNLM import RNNLM
import torch
from evaluations import evaluate_VAE, evaluate_rnn

validate_every = 250 # how often we want to validate our models
print_every = 100 # how often we want some results

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

        for idx, train_batch in enumerate(train_loader):
            model.train()
            loss = train_batch_rnn(model, optimizer, loss_fn, train_batch, device)
            loss = loss / train_batch.shape[0]
            perplexity = torch.log(loss)

            it = epoch * len(train_loader) + idx
            results_writer.add_scalar('train-rnn/loss', loss, it)
            results_writer.add_scalar('train-rnn/ppl', perplexity, it)


            if it % print_every == 0:
                print(f'Iter: {it}, {round(idx/len(train_loader)*100)}/100% || Loss: {loss} || Perplexity {perplexity}')
            
            if it % validate_every == 0:
                print("Validating model")
                valid_loss, valid_perp = evaluate_rnn(model, valid_loader, it, device, loss_fn)
                previous_valid_loss = valid_loss
                if previous_valid_loss < best_valid_loss:
                    print('New Best Validation score!')
                    best_valid_loss = previous_valid_loss
                    # have to save model

                print(f'Validation results || Loss: {valid_loss} || Perplexity {valid_perp}')
                results_writer.add_scalar('valid-rnn/loss' , valid_loss, it)
                results_writer.add_scalar('valid-rnn/ppl' , valid_perp, it)
                print()
                model.train()
        print('\n\n')
    print("Done with training!")


def train_batch_vae(model, optimizer, criterion, train_batch, prior, device, mu_force_beta_param, writer):
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

    # Now add to the loss mu force loss
    batch_mean_vectors = posterior.loc
    avg_batch_mean_vector = batch_mean_vectors.mean(0)
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
    vocab_size = model.vocab_size
    loss_fn = make_elbo_criterion(vocab_size, freebits_param, mu_force_beta_param)

    prior = torch.distributions.Normal(
        torch.zeros(model.latent_size),
        torch.ones(model.latent_size)
    )

    for epoch in range(nr_epochs):
        print (epoch)

        for idx, train_batch in enumerate(train_loader):
            loss, preds = train_batch_vae(
                model,
                optimizer,
                loss_fn,
                train_batch,
                prior,
                device,
                mu_force_beta_param,
                results_writer
            )
            loss, kl_loss, nlll, mu_loss = loss
            results_writer.add_scalar('train-vae/elbo-loss', loss, epoch * len(train_loader) + idx)
            results_writer.add_scalar('train-vae/ppl', torch.log(torch.tensor(loss)), epoch * len(train_loader) + idx)
            results_writer.add_scalar('train-vae/kl-loss', kl_loss, epoch * len(train_loader) + idx)
            results_writer.add_scalar('train-vae/nll-loss', nlll, epoch * len(train_loader) + idx)
            results_writer.add_scalar('train-vae/mu-loss', mu_loss, epoch * len(train_loader) + idx)

            if idx % print_every == 0:
                print(f'iteration: {epoch * len(train_loader) + idx} || KL Loss: {kl_loss} || NLLL: {nlll} || MuLoss: {mu_loss} || Total: {loss}')

            # Every 100 iterations, predict a sentence and check the truth
            if idx % print_every == 0:
                decoded_first_pred = decoder(preds) # TODO: Cutt-off after sentence length?
                decoded_first_true = decoder(train_batch[:, 1:])
                results_writer.add_text(f'it {epoch * len(train_loader) + idx}: prediction', decoded_first_pred)
                results_writer.add_text(f'it {epoch * len(train_loader) + idx}: truth', decoded_first_true)

            if idx % validate_every == 0:
                print('Validating model')
                evaluate_VAE(model, 
                    valid_loader, 
                    epoch, 
                    device, 
                    loss_fn, 
                    mu_force_beta_param, 
                    prior, 
                    results_writer, 
                    iteration = epoch * len(train_loader) + idx
                )
                model.train()

    print('Done training the VAE')
