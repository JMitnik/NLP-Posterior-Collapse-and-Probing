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

train_loader = cd.get_data_loader(type="train", shuffle=False)#DataLoader(cd.train_set, batch_size=config.batch_size, shuffle=False, collate_fn=padded_collate)

# Small test for a data loader
# for di, d in enumerate(train_loader):
#     print(d)
#     print(d.tolist())
#     print(f'{"-"*20}')
#     if di == 1:
#         break

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

# Define our model, optimizer and loss function
rnn_lm = RNNLM(config.vocab_size, config.embedding_size, config.hidden_size).to(config.device)
criterion = nn.CrossEntropyLoss(
    ignore_index=0,
    reduction='sum'
)
optim = torch.optim.Adam(rnn_lm.parameters())

# Start training
training_writer = SummaryWriter()
no_iters = len(train_loader)
print(no_iters)
print_every = round(no_iters / 50) # print 50 times
print(print_every)


for epoch in range(config.nr_epochs):
    print(f'Epoch: {epoch}')
    iter = 0 + epoch * len(train_loader)
    for train_batch in train_loader:
        
        loss = train_on_batch(rnn_lm, optim, train_batch)
        loss = loss / config.batch_size
        perplexity = torch.log(loss)

        # TODO: Improve training results, log also on and across epoch
        utils.store_training_results(training_writer, loss, perplexity, iter)
        
        if iter % print_every == 0:
            print(f'Iter: {iter}, {round(iter/no_iters*100)}/100% || Loss: {loss} || Perplexity {perplexity}')
        iter += 1

print("Done with training!")

# %%

def impute_next_word(model, start="tomorrow", max_length=20):
    print(f'Start of the sentence: {start} || Max Length {max_length} .')
    with torch.no_grad():
        encoded_start = cd.tokenizer.encode(start, add_special_tokens=True)[:-1]
        print(encoded_start)
        print(cd.tokenizer.decode(encoded_start, skip_special_tokens=False))
        model_inp = torch.tensor(encoded_start).to(config.device)
        model_inp = model_inp.unsqueeze(0) # Ensures we pass a 1(=batch-dimension) x sen-length vector
        print(model_inp)
        
        # output_name = start_letter
        # What about our hidden layer? Musnt we set it to zero when we create a new sentence??
        # That way, our history is "cleared" and we can create a new sentence
        # hidden = model.init_hidden(model_inp)
        # print(hidden.size())
        sentence = [encoded_start]
        print(sentence)
        # output, hidden = model(model_inp, hidden)
        for i in range(max_length):
            print("get a new word")
            output = model(model_inp).cpu().detach()
            topi = output[:, -1, :].topk(1)
            print(topi)
            print(topi[0])
            print(topi.indices.item())
            word = [topi.indices[0][0].item()]
            # word = cd.tokenizer.decode(word)
            print(word)            
            # word = cd.tokenizer.decode(topi.indices[0][0])
            # print(word)
            if word == 2:
                break
            print(cd.tokenizer.decode([34, 6, 89, 3, 5, 6666, 9999]))
            break
        #     topv, topi = output.topk(1)
        #     topi = topi[0][0]
        #     if topi == n_letters - 1:
        #         break
        #     else:
        #         letter = all_letters[topi]
        #         output_name += letter
        #     input = inputTensor(letter)


    # pred = model(inp).cpu().detach()

    # # Get prediction for last token
    # last_pred = [pred[:, -1, :].argmax().item()]

    # # Decode prediction
    # output = cd.tokenizer.decode(last_pred)
    # print(output)
    # return output

# TODO: Shitty results, hmm
impute_next_word(rnn_lm)


# %%
