import torch
from transformers import GPT2Model, GPT2Tokenizer, XLMRobertaModel, XLMRobertaTokenizer
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

def make_pretrained_transformer_and_tokenizer(
    transformer_name: str
):
    if 'distilgpt2' in transformer_name:
        print("DistilGPT2!")
        model = GPT2Model.from_pretrained('distilgpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    else:
        print(f"Loading {transformer_name}!")
        model = XLMRobertaModel.from_pretrained(transformer_name)
        tokenizer = XLMRobertaTokenizer.from_pretrained(transformer_name)

    return model, tokenizer
