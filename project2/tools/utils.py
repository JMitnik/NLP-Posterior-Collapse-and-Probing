import os
import pickle
import torch

def ensure_path(path_to_file):
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)

def save_model(path_to_file, model):
    ensure_path(path_to_file)
    torch.save(model.state_dict(), path_to_file)

def load_model(path_to_file, model):
    model.load_state_dict(torch.load(path_to_file))

def save_vocab(path_to_file, vocab):
    ensure_path(path_to_file)
    vocab.default_factory = None
    pickle.dump(vocab, open(path_to_file, 'wb'))

def load_vocab(path_to_file):
    vocab = pickle.load(open(path_to_file, 'rb'))
    vocab.default_factory = lambda: 1

    return vocab
