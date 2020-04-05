# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Initialization
# %% [markdown]
# # Data

# %%
train_path = '/data/02-21.10way.clean'
valid_path = '/data/22.auto.clean'
test_path  = '/data/23.auto.clean'

# %% [markdown]
# Our data files consist of lines of the Penn Tree Bank data set. Each line is a sentence in a tree shape. Let's pretty print the first line from the training set to see what we're dealing with!

# %%
import os
from nltk import Tree
from nltk.treeprettyprinter import TreePrettyPrinter

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
line = next(filereader(train_path))
print(f'Original: {line}')
print(f'Prased: {convert_to_sentence(line)}')

# %% [markdown]
# #### Creating our datasets
# 
# We have the data, now we want to create dataloaders so we can shuffle and batch our data. For this we will use pytorch's built in classes.

# %%
# We have a training, validation and test data set. For each, we need a list of sentences.
train_sents = [convert_to_sentence(l) for l in filereader(train_path)]
valid_sents = [convert_to_sentence(l) for l in filereader(valid_path)]
test_sents = [convert_to_sentence(l) for l in filereader(test_path)]

# %% [markdown]
# For our models, we tensors that are easily interpretable for our machines. For this, we convert our sentences to tensors where each word is represented by a number, also we want to have special character tokens and limit our vocabulary so our parameter space is a bit more managable. To do all this, we use the *tokenizers.py* file with the WordTokenizer class, presented by the NLP2-Team.
# 
# *Note*: We had special tokens to our sentences such as BOS, EOS and UNK. The BOS and EOS tokens are important, because it tells the model what constitutes as the beginning and the end of a sentence.
# 
# We see that, in the code block below, that add_special_tokens is set to True, and appends a BOS and EOS token to the sentences, and are represented as number 1 and 2 in tensor-form. We decode the sentences taking in these special tokens into account.

# %%
from tokenizers import WordTokenizer

# How big we want our vocabulary to be
vocab_size = 10000

# Creating and train our tokenizer. We want a relatively small vocabulary of 10000 words. Credits to the NLP2 team for creating this tokenizer.
tokenizer = WordTokenizer(train_sents, max_vocab_size=vocab_size)

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
train_loader = DataLoader(train_set, batch_size=2, shuffle=False, collate_fn=padded_collate)

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
import models.RNNLM
import importlib
importlib.reload(models.RNNLM)
from models.RNNLM import RNNLM
embedding_size = 50
hidden_size = 50
# vocab size is our vocab size, embedding is 500, hidden is 100. These number are arbitrary for now. Still trying to make the model work
rnnlm = RNNLM(vocab_size, embedding_size, hidden_size)


# %%
import torch
import torch.nn as nn

# Define loss function
criterion = nn.NLLLoss(ignore_index=0)
optim = torch.optim.Adam(rnnlm.parameters())


# %%
def train_model_on_batch(model: RNNLM, optim: torch.optim.Optimizer, input_tensor: torch.Tensor):
    optim.zero_grad()
    # inp is to be shaped as Sentences(=batch) x Words
    hidden = model.init_hidden(input_tensor)
    batch_loss = 0

    output = model(input_tensor)
    
    # Calc loss and perform backprop
    loss = criterion(torch.log(output), torch.tensor(next_words))
    loss.backward()

    optim.step()
    return batch_loss


# %%
all_losses = []

for batch in train_loader:
 
    print(batch)
    print(f'{"-"*20}')
    train_model_on_batch(rnnlm, optim, batch)
    break


# %%



