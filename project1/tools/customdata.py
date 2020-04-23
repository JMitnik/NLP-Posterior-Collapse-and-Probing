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
    Custom dataset
    """
    def __init__(self, config: Config):
        print('Init Custom Dataset')

        self.train_sents = [self.convert_to_sentence(l) for l in filereader(config.train_path)]
        self.valid_sents = [self.convert_to_sentence(l) for l in filereader(config.valid_path)]
        self.test_sents = [self.convert_to_sentence(l) for l in filereader(config.test_path)]
        self.config = config
        self.tokenizer = WordTokenizer(self.train_sents, config.vocab_size)

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
        return torch.LongTensor(padded_batch), torch.LongTensor(sentence_lengths)


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

def filereader(path: str):
    """
    Opens a PTS datafile yields one line at a time.
    """
    with open(os.getcwd() + path, mode='r') as f:
        for line in f:
            yield line
