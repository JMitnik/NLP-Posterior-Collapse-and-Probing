from losses import make_elbo_criterion
from models.VAE import VAE
import torch

def train_batch_vae(model, optimizer, criterion, train_batch, prior):
    """
    Trains single batch of VAE
    """

    optimizer.zero_grad()

    # Current assumption: to not predict past final token, we dont include the EOS tag in the input
    inp = train_batch[:, 0:-1].to(config.device)

    # Creat both prediction of next word and the posterior of which we sample Z.
    preds, posterior = model(inp)

    # Define target as the next word to predict
    target = train_batch[:, 1:].to(config.device)

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

    return loss.item(), kl_loss.item(), nlll.item()

def train_vae(model: VAE, optimizer, nr_epochs, train_loader, freebits_param):
    vocab_size = model.vocab_size
    loss_fn = make_elbo_criterion(vocab_size, freebits_param=5)

    prior = torch.distributions.Normal(
        torch.zeros(model.latent_size),
        torch.ones(model.latent_size)
    )

    for epoch in range(nr_epochs):
        print (epoch)
        i = 0
        for train_batch in train_loader:
            loss = train_batch_vae(model, optimizer, loss_fn, train_batch, prior)
            loss, kl_loss, nlll = loss

            if i % 10 == 0:
                print(f'iteration: {i} || KL Loss: {kl_loss} || NLLL: {nlll} || Total: {loss}')
            i += 1
    print('Done training the VAE')
