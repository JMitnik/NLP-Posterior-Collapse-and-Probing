import os
import importlib
from nltk import Tree
from nltk.treeprettyprinter import TreePrettyPrinter
from config import Config

from tokenizers import WordTokenizer

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class CustomData:
    """
        Put Information in here
    """
    def __init__(self, config: Config):
        print('Init Custom Dataset')
        self.train_sents = [self.convert_to_sentence(l) for l in filereader(config.train_path)]
        # self.valid_sents = [self.convert_to_sentence(l) for l in filereader(config.valid_path)]
        # self.test_sents = [self.convert_to_sentence(l) for l in filereader(config.test_path)]
        self.config = config
        self.tokenizer = WordTokenizer(self.train_sents, config.vocab_size)

        # self.train_set = PTBDataset(self.train_sents, self.tokenizer)
        # self.valid_set = PTBDataset(self.valid_sents, self.tokenizer)
        # self.test_set = PTBDataset(self.test_sents, self.tokenizer)
        print('Data loaders are ready')

    def get_data_loader(self, type:str, shuffle:bool):
        data_set = PTBDataset([], self.tokenizer)
        if type == "train":
             data_set = PTBDataset(self.train_sents, self.tokenizer)
        elif type == "valid":
            data_set = PTBDataset(self.valid_sents, self.tokenizer)
        elif type == "test":
            data_set = PTBDataset(self.test_sents, self.tokenizer)
        else:
            raise ValueError('type argument was invalid; provide either: "train", "valid" or "test".')
        print(f"Created a {type} data loader. Shuffle = {shuffle}, Batch Size = {self.config.batch_size}")
        return DataLoader(data_set, batch_size=self.config.batch_size, shuffle=shuffle, collate_fn=self.padded_collate)

    def convert_to_sentence(self, line: str):
        """
            Takes in a line from a PTS datafile and returns it as a lower-case string.
        """
        tree = Tree.fromstring(line)
        sentence = ' '.join(tree.leaves()).lower()
        return sentence

    def padded_collate(self, batch: list):
        """
        Pad each sentence to the length of the longest sentence in the batch
        """
        sentence_lengths = [len(s) for s in batch]
        max_length = max(sentence_lengths)
        padded_batch = [s + [0] * (max_length - len(s)) for s in batch]
        return torch.LongTensor(padded_batch)


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

def filereader(path:str):
    """
        Opens a PTS datafile yields one line at a time.
    """
    with open(os.getcwd() + path, mode='r') as f:
        for line in f:
            yield line

######## OLD CODE FOR DATA, ITS STILL HERE IN CASE WE FORGOT ANYTHING #########


# exit()
# # %% [markdown]
# # Our data files consist of lines of the Penn Tree Bank data set. Each line is a sentence in a tree shape. Let's pretty print the first line from the training set to see what we're dealing with!

# # %%

# def filereader(path:str):
#     """
#         Opens a PTS datafile yields one line at a time.
#     """
#     with open(os.getcwd() + path, mode='r') as f:
#         for line in f:
#             yield line


# # %%
# def convert_to_sentence(line: str):
#     """
#         Takes in a line from a PTS datafile and returns it as a lower-case string.
#     """
#     tree = Tree.fromstring(line)
#     sentence = ' '.join(tree.leaves()).lower()
#     return sentence

# # %% [markdown]
# # #### Let's see how our data looks:

# # %%
# line = next(filereader(config.train_path))
# print(f'Original: {line}')
# print(f'Prased: {convert_to_sentence(line)}')

# # %% [markdown]
# # #### Creating our datasets
# #
# # We have the data, now we want to create dataloaders so we can shuffle and batch our data. For this we will use pytorch's built in classes.

# # %%
# # We have a training, validation and test data set. For each, we need a list of sentences.
# train_sents = [convert_to_sentence(l) for l in filereader(config.train_path)]
# valid_sents = [convert_to_sentence(l) for l in filereader(config.valid_path)]
# test_sents = [convert_to_sentence(l) for l in filereader(config.test_path)]

# # %% [markdown]
# # For our models, we tensors that are easily interpretable for our machines. For this, we convert our sentences to tensors where each word is represented by a number, also we want to have special character tokens and limit our vocabulary so our parameter space is a bit more managable. To do all this, we use the *tokenizers.py* file with the WordTokenizer class, presented by the NLP2-Team.
# #
# # *Note*: We had special tokens to our sentences such as BOS, EOS and UNK. The BOS and EOS tokens are important, because it tells the model what constitutes as the beginning and the end of a sentence.
# #
# # We see that, in the code block below, that add_special_tokens is set to True, and appends a BOS and EOS token to the sentences, and are represented as number 1 and 2 in tensor-form. We decode the sentences taking in these special tokens into account.

# # %%
# from tokenizers import WordTokenizer

# # Creating and train our tokenizer. We want a relatively small vocabulary of 10000 words. Credits to the NLP2 team for creating this tokenizer.
# tokenizer = WordTokenizer(train_sents, max_vocab_size=config.vocab_size)

# # We check if the tokenizer en- and decodes our sentences correctly. Just look at the top-5 sentences in our training set.
# for sentence in train_sents[:5]:
#     tokenized = tokenizer.encode(sentence, add_special_tokens=True)
#     sentence_decoded = tokenizer.decode(tokenized, skip_special_tokens=False)

#     print('original: ' + sentence)
#     print(f'{"-"*10}')
#     print('tokenized: ', tokenized)
#     print(f'{"-"*10}')
#     print('decoded: ' + sentence_decoded)
#     print(f'{"-"*10}')
#     print('\n\n')

# # %% [markdown]
# # #### Creating custom pytorch data sets
# #
# # To work with pytorch data loaders, we want custom pytorch dataset.

# # %%
# from torch.utils.data import Dataset

# class PTBDataset(Dataset):
#     """
#         A custom PTB dataset.
#     """
#     def __init__(self, sentences: list, tokenizer: WordTokenizer):
#         self.sentences = sentences
#         self.tokenizer = tokenizer

#     def __len__(self):
#         """
#             Return the length of the dataset.
#         """
#         return len(self.sentences)

#     def __getitem__(self, idx: int):
#         """
#             Returns a tokenized item at position idx from the dataset.
#         """
#         item = self.sentences[idx]
#         tokenized = self.tokenizer.encode(item, add_special_tokens=True)
#         return tokenized


# # %%
# # We instantiate the datasets.
# train_set = PTBDataset(train_sents, tokenizer)
# valid_set = PTBDataset(valid_sents, tokenizer)
# test_set = PTBDataset(test_sents, tokenizer)

# # Lets print some information about our datasets
# print(f'train/validation/test :: {len(train_set)}/{len(valid_set)}/{len(test_set)}')

# # %% [markdown]
# # Now let's create dataloaders that can load/shuffle and batch our data. W
# # When pytorch batches our data, it stacks the tensors. However, since our sentences are not equal in size, their corresponding tensors will also have different sizes. To fix this problem, we pad each sentence in a batch to the size, taking our longest sentence as the target size. This way, stacking tensors won't be a problem.

# # %%
# from torch.utils.data import DataLoader
# import torch

# def padded_collate(batch: list):
#     """
#      Pad each sentence to the length of the longest sentence in the batch
#     """
#     sentence_lengths = [len(s) for s in batch]
#     max_length = max(sentence_lengths)
#     padded_batch = [s + [0] * (max_length - len(s)) for s in batch]
#     return torch.LongTensor(padded_batch)

# %% [markdown]
# We want to test if our data loader works, so we create one of our test set with a tiny batch size of 2. From to batches, we print the output.