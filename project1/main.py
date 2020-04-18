# %% [markdown]
# ## Imports

# %%
from dataclasses import asdict
import os
import importlib
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
from torch.utils.tensorboard import SummaryWriter

from customdata import CustomData

# %%
print(torch.version.cuda)
print(torch.cuda.is_available())


# %%
config = Config(
    batch_size=16,
    embedding_size=50,
    rnn_hidden_size=50,
    vae_encoder_hidden_size=128,
    vae_decoder_hidden_size=1281,
    param_wdropout_k=0.5,
    vae_latent_size=128,
    vocab_size=10000,
    nr_epochs=1,
    train_path = '/data/02-21.10way.clean',
    valid_path = '/data/22.auto.clean',
    test_path  = '/data/23.auto.clean',
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
)

print(f'Running Models on {config.device}.')

cd = CustomData(config)

train_loader = cd.get_data_loader(type="train", shuffle=True)
valid_loader = cd.get_data_loader(type='valid', shuffle=False)
test_loader = cd.get_data_loader(type='test', shuffle=False)


# %% [markdown]
# ## Defining the Model
# We import the model from our models folder.

# %%

def train_on_batch(model: RNNLM, optim: torch.optim.Optimizer, input_batch: torch.Tensor):
    optim.zero_grad()

    inp = input_batch[:, 0:-1].to(config.device)

    # Current assumption: to not predikct past final token, we dont include the EOS tag in the input
    output = model(inp)

    # One hot target and shift by one (so first word matches second token)
    # target = torch.nn.functional.one_hot(input_tensor, num_classes=vocab_size)
    target = input_batch[:, 1:].to(config.device)

    # Calc loss and perform backprop
    loss = criterion(output.reshape(-1, config.vocab_size), target.reshape(-1))
    loss.backward()

    optim.step()
    return loss

def evaluate_model(model, data_loader, epoch):
    model.eval()
    total_loss = 0

    for batch in data_loader:
        with torch.no_grad():
            input = batch[:, 0: -1].to(config.device)
            target = batch[:, 1:].to(config.device)
        
            output = model(input)

            loss = criterion(output.reshape(-1, config.vocab_size), target.reshape(-1))
            total_loss += loss / len(batch)
    
    total_loss = total_loss / len(data_loader)
    return total_loss, torch.log(total_loss)


# Define our model, optimizer and loss function
rnn_lm = RNNLM(config.vocab_size, config.embedding_size, config.rnn_hidden_size).to(config.device)
criterion = nn.CrossEntropyLoss(
    ignore_index=0,
    reduction='sum'
)
optim = torch.optim.Adam(rnn_lm.parameters())

# %%
# Start training
writer = SummaryWriter()
no_iters = len(train_loader)
print_every = round(no_iters / 50) # print 50 times
validate_every = round(no_iters/10) # validate 10 times
total_iters = 0


for epoch in range(config.nr_epochs):
    print(f'Epoch: {epoch + 1}')
    iter = 0
    for train_batch in train_loader:
        rnn_lm.train()
        loss = train_on_batch(rnn_lm, optim, train_batch)
        loss = loss / config.batch_size
        perplexity = torch.log(loss)

        # TODO: Improve training results, log also on and across epoch
        utils.store_training_results(writer, 'training' , loss, perplexity, total_iters)
        
        if iter % print_every == 0:
            print(f'Iter: {iter}, {round(iter/no_iters*100)}/100% || Loss: {loss} || Perplexity {perplexity}')
        iter += 1
        total_iters += 1

        if iter % validate_every == 0:
            print("Evaluating...")
            valid_loss, valid_perp = evaluate_model(rnn_lm, valid_loader, epoch)
            print(f'Validation -- Iter: {iter}, {round(iter/no_iters*100)}/100% || Loss: {loss} || Perplexity {perplexity}')
            utils.store_training_results(writer, 'validation' , valid_loss, valid_perp, total_iters)
    print('\n\n')
print("Done with training!")


# %%
import torch.nn.functional as F
import torch.distributions as D

temperature = 1.0

def impute_next_word(model, start="bank profits plummeted", max_length=20):
    print(f'Start of the sentence: {start} || Max Length {max_length} .')
    with torch.no_grad():
        encoded_start = cd.tokenizer.encode(start, add_special_tokens=True)[:-1]
        sentence = encoded_start
        print(sentence)

        for i in range(max_length):
            # Create input for the model
            model_inp = torch.tensor(sentence).to(config.device)
            model_inp = model_inp.unsqueeze(0) # Ensures we pass a 1(=batch-dimension) x sen-length vector
            output = model(model_inp).cpu().detach()
            # print(output)
            prediction_vector = F.softmax(output[0][-1] / temperature)
            sample_vector = D.Categorical(prediction_vector)
            sample = int(sample_vector.sample())
            if sample == 3: # cannot produces UNK token
                i = i-1
                continue
            sentence.append(sample)

            if sample == 2: # If we sampled EOS
                break

        print(sentence)
        print(f'Sentence Length: {len(sentence)}')

        return sentence
            
generated_sentence = impute_next_word(rnn_lm)
print(cd.tokenizer.decode(generated_sentence))

# %%
##-------------------- START VAE --------------------------------------------##
#%%
def make_elbo_criterion():
    likelihood_criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    print("Made a new loss-func")

    def elbo_criterion(
        prediction: torch.Tensor,
        original: torch.Tensor,
        prior_dist: torch.distributions.Distribution,
        posterior_dist: torch.distributions.Distribution
    ):
        batch_size = prediction.shape[0]

        kl_loss = torch.distributions.kl_divergence(prior_dist, posterior_dist).sum(1).to(config.device)

        negative_log_likelihood = likelihood_criterion(
            prediction.view([-1, config.vocab_size]),
            original.view(-1)
        ).to(config.device)

        # Mean of all words for each batch item
        negative_log_likelihood = negative_log_likelihood.view(prediction.shape[0], -1).sum(1)

        return negative_log_likelihood + kl_loss, kl_loss, negative_log_likelihood

    return elbo_criterion

def batch_train_vae(
    model,
    optimizer,
    criterion,
    train_batch,
    prior,
):
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


 
# Playing around with VAEs now
import models.VAE
importlib.reload(models.VAE)
from models.VAE import VAE
vae = VAE(
    encoder_hidden_size=config.vae_encoder_hidden_size,
    decoder_hidden_size=config.vae_decoder_hidden_size,
    latent_size=config.vae_latent_size,
    vocab_size=config.vocab_size,
    param_wdropout_k=config.param_wdropout_k,
    embedding_size=config.embedding_size
).to(config.device)

elbo_criterion = make_elbo_criterion()
prior = torch.distributions.Normal(
    torch.zeros(config.vae_latent_size),
    torch.ones(config.vae_latent_size)
)

for epoch in range(config.nr_epochs):
    print (epoch)
    i = 0
    for train_batch in train_loader:
        loss = batch_train_vae(
            vae,
            optim,
            elbo_criterion,
            train_batch,
            prior
        )
        loss, kl_loss, nlll = loss
        if i % 10 == 0:
            print(f'iteration: {i} || KL Loss: {kl_loss} || NLLL: {nlll} || Total: {loss}')
        i += 1
print('Done training the VAE')


# %%
print('Done')

# %%
