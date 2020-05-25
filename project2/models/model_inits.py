import torch
import torch.nn as nn
from typing import Dict, Tuple
from models.lstm.model import RNNModel
from collections import defaultdict


def make_pretrained_lstm_and_tokenizer(
    path_to_pretrained_lstm='storage/pretrained_lstm_state_dict.pt',
    path_to_lstm_vocab='models/lstm/vocab.txt'
):
    """
    Loads pretrained LSTM and its vocab
    """
    # Load LSTM
    lstm = RNNModel('LSTM', 50001, 650, 650, 2)
    lstm.load_state_dict(torch.load(path_to_pretrained_lstm))

    # Load vocab
    with open('models/lstm/vocab.txt') as f:
        w2i = {w.strip(): i for i, w in enumerate(f)}

    vocab = defaultdict(lambda: w2i["<unk>"])
    vocab.update(w2i)

    return lstm, vocab
