from config import Config
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from losses import make_elbo_criterion
from models.VAE import VAE
from models.RNNLM import RNNLM
import torch

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
    nr_epochs: int,
    device: str,
    results_writer: SummaryWriter,
):
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=0,
        reduction='sum'
    )

    for epoch in range(nr_epochs):
        print(f'Epoch: {epoch + 1} / {nr_epochs}')
        it = 0
        for index, train_batch in enumerate(train_loader):
            model.train()
            loss = train_batch_rnn(model, optimizer, loss_fn, train_batch, device)
            loss = loss / train_batch.shape[0]
            perplexity = torch.log(loss)

            # # TODO: Improve training results, log also on and across epoch
            # utils.store_training_results(
            #     results_writer,
            #     'training',
            #     loss,
            #     perplexity,
            #     total_iters
            # )

            # if it % print_every == 0:
            #     print(f'Iter: {it}, {round(it/no_iters*100)}/100% || Loss: {loss} || Perplexity {perplexity}')
            # iter += 1
            # total_iters += 1

            # if iter % validate_every == 0:
            #     print("Evaluating...")
            #     valid_loss, valid_perp = evaluate_model(rnn_lm, valid_loader, epoch)
            #     print(f'Validation -- Iter: {iter}, {round(iter/no_iters*100)}/100% || Loss: {loss} || Perplexity {perplexity}')
            #     utils.store_training_results(writer, 'validation' , valid_loss, valid_perp, total_iters)
        print('\n\n')
    print("Done with training!")


def train_batch_vae(model, optimizer, criterion, train_batch, prior, device, writer):
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

    # Backprop and gradient descent
    loss.backward()
    optimizer.step()

    return (loss.item(), kl_loss.item(), nlll.item()), preds

def train_vae(
    model: VAE,
    optimizer,
    train_loader: DataLoader,
    nr_epochs: int,
    device: str,
    results_writer: SummaryWriter,
    config: Config,
    decoder,
    freebits_param=-1,
):
    vocab_size = model.vocab_size
    loss_fn = make_elbo_criterion(vocab_size, freebits_param)

    prior = torch.distributions.Normal(
        torch.zeros(model.latent_size),
        torch.ones(model.latent_size)
    )

    for epoch in range(nr_epochs):
        print (epoch)
        i = 0
        for idx, train_batch in enumerate(train_loader):
            loss, preds = train_batch_vae(
                model,
                optimizer,
                loss_fn,
                train_batch,
                prior,
                device,
                results_writer
            )
            loss, kl_loss, nlll = loss
            results_writer.add_scalar('train-vae/total-loss', loss, epoch * len(train_loader) + idx)
            results_writer.add_scalar('train-vae/kl-loss', kl_loss, epoch * len(train_loader) + idx)
            results_writer.add_scalar('train-vae/nll-loss', nlll, epoch * len(train_loader) + idx)

            if i % 10 == 0:
                print(f'iteration: {i} || KL Loss: {kl_loss} || NLLL: {nlll} || Total: {loss}')

            # Every 100 iterations, predict a sentence and check the truth
            if i % 100 == 0:
                decoded_first_pred = decoder(preds) # TODO: Cutt-off after sentence length?
                decoded_first_true = decoder(train_batch[:, 1:])
                results_writer.add_text(f'it {epoch * len(train_loader) + idx}: prediction', decoded_first_pred)
                results_writer.add_text(f'it {epoch * len(train_loader) + idx}: truth', decoded_first_true)

            i += 1
    print('Done training the VAE')
