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

# %% [markdown]
# ## Initialization
# %% [markdown]
# # Data

# %%

config = Config(
    batch_size=16,
    embedding_size=50,
    hidden_size=50,
    vocab_size=10000,
    nr_epochs=50,
    train_path = '/data/02-21.10way.clean',
    valid_path = '/data/22.auto.clean',
    test_path  = '/data/23.auto.clean',
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
)

# %% [markdown]
# Our data files consist of lines of the Penn Tree Bank data set. Each line is a sentence in a tree shape. Let's pretty print the first line from the training set to see what we're dealing with!

# %%

def filereader(path:str):
    """
        Opens a PTS datafile yields one line at a time.
    """
    with open(os.getcwd() + path, mode='r') as f:
        for line in f:
            yield line


# %%
def convert_to_sentence(line: str):
    """
        Takes in a line from a PTS datafile and returns it as a lower-case string.
    """
    tree = Tree.fromstring(line)
    sentence = ' '.join(tree.leaves()).lower()
    return sentence

# %% [markdown]
# #### Let's see how our data looks:

# %%
line = next(filereader(config.train_path))
print(f'Original: {line}')
print(f'Prased: {convert_to_sentence(line)}')

# %% [markdown]
# #### Creating our datasets
#
# We have the data, now we want to create dataloaders so we can shuffle and batch our data. For this we will use pytorch's built in classes.

# %%
# We have a training, validation and test data set. For each, we need a list of sentences.
train_sents = [convert_to_sentence(l) for l in filereader(config.train_path)]
valid_sents = [convert_to_sentence(l) for l in filereader(config.valid_path)]
test_sents = [convert_to_sentence(l) for l in filereader(config.test_path)]

# %% [markdown]
# For our models, we tensors that are easily interpretable for our machines. For this, we convert our sentences to tensors where each word is represented by a number, also we want to have special character tokens and limit our vocabulary so our parameter space is a bit more managable. To do all this, we use the *tokenizers.py* file with the WordTokenizer class, presented by the NLP2-Team.
#
# *Note*: We had special tokens to our sentences such as BOS, EOS and UNK. The BOS and EOS tokens are important, because it tells the model what constitutes as the beginning and the end of a sentence.
#
# We see that, in the code block below, that add_special_tokens is set to True, and appends a BOS and EOS token to the sentences, and are represented as number 1 and 2 in tensor-form. We decode the sentences taking in these special tokens into account.

# %%
from tokenizers import WordTokenizer

# Creating and train our tokenizer. We want a relatively small vocabulary of 10000 words. Credits to the NLP2 team for creating this tokenizer.
tokenizer = WordTokenizer(train_sents, max_vocab_size=config.vocab_size)

# We check if the tokenizer en- and decodes our sentences correctly. Just look at the top-5 sentences in our training set.
for sentence in train_sents[:5]:
    tokenized = tokenizer.encode(sentence, add_special_tokens=True)
    sentence_decoded = tokenizer.decode(tokenized, skip_special_tokens=False)

    print('original: ' + sentence)
    print(f'{"-"*10}')
    print('tokenized: ', tokenized)
    print(f'{"-"*10}')
    print('decoded: ' + sentence_decoded)
    print(f'{"-"*10}')
    print('\n\n')

# %% [markdown]
# #### Creating custom pytorch data sets
#
# To work with pytorch data loaders, we want custom pytorch dataset.

# %%
from torch.utils.data import Dataset

class PTBDataset(Dataset):
    """
        A custom PTB dataset.
    """
    def __init__(self, sentences: list, tokenizer: WordTokenizer):
        self.sentences = sentences
        self.tokenizer = tokenizer

    def __len__(self):
        """
            Return the length of the dataset.
        """
        return len(self.sentences)

    def __getitem__(self, idx: int):
        """
            Returns a tokenized item at position idx from the dataset.
        """
        item = self.sentences[idx]
        tokenized = self.tokenizer.encode(item, add_special_tokens=True)
        return tokenized


# %%
# We instantiate the datasets.
train_set = PTBDataset(train_sents, tokenizer)
valid_set = PTBDataset(valid_sents, tokenizer)
test_set = PTBDataset(test_sents, tokenizer)

# Lets print some information about our datasets
print(f'train/validation/test :: {len(train_set)}/{len(valid_set)}/{len(test_set)}')

# %% [markdown]
# Now let's create dataloaders that can load/shuffle and batch our data. W
# When pytorch batches our data, it stacks the tensors. However, since our sentences are not equal in size, their corresponding tensors will also have different sizes. To fix this problem, we pad each sentence in a batch to the size, taking our longest sentence as the target size. This way, stacking tensors won't be a problem.

# %%
from torch.utils.data import DataLoader
import torch

def padded_collate(batch: list):
    """
     Pad each sentence to the length of the longest sentence in the batch
    """
    sentence_lengths = [len(s) for s in batch]
    max_length = max(sentence_lengths)
    padded_batch = [s + [0] * (max_length - len(s)) for s in batch]
    return torch.LongTensor(padded_batch)

# %% [markdown]
# We want to test if our data loader works, so we create one of our test set with a tiny batch size of 2. From to batches, we print the output.

# %%
train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=False, collate_fn=padded_collate)

# Small test for a data loader
for di, d in enumerate(train_loader):
    print(d)
    print(d.tolist())
    print(f'{"-"*20}')
    if di == 1:
        break

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

losses = []
perplexities = []

for epoch in range(config.nr_epochs):
    for train_batch in train_loader:
        loss = train_on_batch(rnn_lm, optim, train_batch)
        loss = loss / config.batch_size

        losses.append(loss)
        perplexity = torch.log(loss)

        losses.append(loss)

        # TODO: Improve training results, log also on and across epoch
        utils.store_training_results(training_writer, loss, perplexity, epoch)

# %%
def impute_next_word(model, sentence):
    inp = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).to(config.device)
    inp = inp.unsqueeze(0) # Ensures we pass a 1(=batch-dimension) x sen-length vector

    pred = model(inp).cpu().detach()

    # Get prediction for last token
    last_pred = [pred[:, -1, :].argmax().item()]

    # Decode prediction
    output = tokenizer.decode(last_pred)
    return output

# TODO: Shitty results, hmm
impute_next_word(rnn_lm, 'Thank the ')


