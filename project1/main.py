# %% [markdown]
# ## Imports

# %%
import os
import importlib
from nltk import Tree
from nltk.treeprettyprinter import TreePrettyPrinter

# Model-related imports
import torch
import torch.nn as nn

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
    hidden_size=50,
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
# We import the model from our models folder. For encoding, this will be our RNNLM, defined in RNNLM.py

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
rnn_lm = RNNLM(config.vocab_size, config.embedding_size, config.hidden_size).to(config.device)
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

# %%
import torch.nn.functional as F
import torch.distributions as D

temperature = 1.01

def impute_next_word(model, start="", max_length=10):
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

/# %%


# %%


# %%
