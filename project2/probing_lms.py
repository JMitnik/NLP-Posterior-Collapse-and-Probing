# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# __Probing Language Models__
#
# This notebook serves as a start for your NLP2 assignment on probing Language Models. This notebook will become part of the contents that you will submit at the end, so make sure to keep your code (somewhat) clean :-)
#
# __note__: This is the first time _anyone_ is doing this assignment. That's exciting! But it might well be the case that certain aspects are too unclear. Do not hesitate at all to reach to me once you get stuck, I'd be grateful to help you out.
#
# __note 2__: This assignment is not dependent on big fancy GPUs. I run all this stuff on my own 3 year old CPU, without any Colab hassle. So it's up to you to decide how you want to run it.

# %% [markdown]
# # Models
#
# For the Transformer models you are advised to make use of the `transformers` library of Huggingface: https://github.com/huggingface/transformers
# Their library is well documented, and they provide great tools to easily load in pre-trained models.

# %%
# Custom Configuration|
from config import Config

config: Config = Config(
    will_train_simple_probe=True,
    struct_probe_train_epoch=100,
    struct_probe_lr=0.001
)

# %%
#
## Your code for initializing the transformer model(s)
#
# Note that most transformer models use their own `tokenizer`, that should be loaded in as well.
#
from transformers import GPT2Model, GPT2Tokenizer

# 🏁
model = GPT2Model.from_pretrained('distilgpt2')
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')


# Note that some models don't return the hidden states by default.
# This can be configured by passing `output_hidden_states=True` to the `from_pretrained` method.

# %%
#
## Your code for initializing the rnn model(s)
#
# The Gulordava LSTM model can be found here:
# https://drive.google.com/open?id=1w47WsZcZzPyBKDn83cMNd0Hb336e-_Sy
#
# N.B: I have altered the RNNModel code to only output the hidden states that you are interested in.
# If you want to do more experiments with this model you could have a look at the original code here:
# https://github.com/facebookresearch/colorlessgreenRNNs/blob/master/src/language_models/model.py
#
from collections import defaultdict
from lstm.model import RNNModel
import torch


model_location = config.path_to_pretrained_lstm  # <- point this to the location of the Gulordava .pt file
lstm = RNNModel('LSTM', 50001, 650, 650, 2)
lstm.load_state_dict(torch.load(model_location))


# This LSTM does not use a Tokenizer like the Transformers, but a Vocab dictionary that maps a token to an id.
with open('lstm/vocab.txt') as f:
    w2i = {w.strip(): i for i, w in enumerate(f)}

vocab = defaultdict(lambda: w2i["<unk>"])
vocab.update(w2i)

# %% [markdown]
# It is a good idea that before you move on, you try to feed some text to your LMs; and check if everything works accordingly.

# %% [markdown]
# # Data
#
# For this assignment you will train your probes on __treebank__ corpora. A treebank is a corpus that has been *parsed*, and stored in a representation that allows the parse tree to be recovered. Next to a parse tree, treebanks also often contain information about part-of-speech tags, which is exactly what we are after now.
#
# The treebank you will use for now is part of the Universal Dependencies project. I provide a sample of this treebank as well, so you can test your setup on that before moving on to larger amounts of data.
#
# Make sure you accustom yourself to the format that is created by the `conllu` library that parses the treebank files before moving on. For example, make sure you understand how you can access the pos tag of a token, or how to cope with the tree structure that is formed using the `to_tree()` functionality.

# %%
from typing import Callable, Dict, List, Optional, Union
from conllu import parse_incr, TokenList

def parse_corpus(filename: str) -> List[TokenList]:
    """
    Parses a file into a collection of TokenLists
    """
    data_file = open(filename, encoding="utf-8")

    ud_parses = list(parse_incr(data_file))

    return ud_parses


# %%
# 🏁
# Some utility data-elements for reference
sample_corpus: List[TokenList] = parse_corpus('data/sample/en_ewt-ud-train.conllu')
sample_sents: List[TokenList] = sample_corpus[:1]
sample_sent: TokenList = sample_corpus[0]

# %%
# 🏁
# Utility functions for dealing with the conllu dataset
from typing import Callable

get_pos_from_sent: Callable[[TokenList], List[str]] = lambda sent: [word['upostag'] for word in sent]
get_tokens_from_sent: Callable[[TokenList], List[str]] = lambda sent: [word['form'] for word in sent]
get_ids_from_sent: Callable[[TokenList], List[str]] = lambda sent: [word['id'] for word in sent]

# %% [markdown]
# # Generating Representations
#
# We now have our data all set, our models are running and we are good to go!
#
# The next step is now to create the model representations for the sentences in our corpora. Once we have generated these representations we can store them, and train additional diagnostic (/probing) classifiers on top of the representations.
#
# There are a few things you should keep in mind here. Read these carefully, as these tips will save you a lot of time in your implementation.
# - Transformer models make use of Byte-Pair Encodings (BPE), that chunk up a piece of next in subword pieces. For example, a word such as "largely" could be chunked up into "large" and "ly". We are interested in probing linguistic information on the __word__-level. Therefore, we will follow the suggestion of Hewitt et al. (2019a, footnote 4), and create the representation of a word by averaging over the representations of its subwords. So the representation of "largely" becomes the average of that of "large" and "ly".
#
# - Subword chunks never overlap multiple tokens. In other words, say we have a phrase like "None of the", then the tokenizer might chunk that into "No"+"ne"+" of"+" the", but __not__ into "No"+"ne o"+"f the", as those chunks overlap multiple tokens. This is great for our setup! Otherwise it would have been quite challenging to distribute the representation of a subword over the 2 tokens it belongs to.
#
# - If you closely examine the provided treebank, you will notice that some tokens are split up into multiple pieces, that each have their own POS-tag. For example, in the first sentence the word "Al-Zaman" is split into "Al", "-", and "Zaman". In such cases, the conllu `TokenList` format will add the following attribute: `('misc', OrderedDict([('SpaceAfter', 'No')]))` to these tokens. Your model's tokenizer does not need to adhere to the same tokenization. E.g., "Al-Zaman" could be split into "Al-"+"Za"+"man", making it hard to match the representations with their correct pos-tag. Therefore I recommend you to not tokenize your entire sentence at once, but to do this based on the chunking of the treebank. Make sure to still incoporate the spaces in a sentence though, as these are part of the BPE of the tokenizer. The tokenizer for GPT-2 adds spaces at the start of a token.
#
# - The LSTM LM does not have the issues related to subwords, but is far more restricted in its vocabulary. Make sure you keep the above points in mind though, when creating the LSTM representations. You might want to write separate functions for the LSTM, but that is up to you.
#
# I would like to stress that if you feel hindered in any way by the simple code structure that is presented here, you are free to modify it :-) Just make sure it is clear to an outsider what you're doing, some helpful comments never hurt.

# %%
# 🏁

from typing import List, DefaultDict
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor

# 🏁
### LSTM FETCH SEN REPR
def fetch_sen_repr_lstm(corpus: List[TokenList], lstm, tokenizer) -> List[Tensor]:
    # TODO: vocab returns the 0 token for the word "She", and now we pad with 0's. We may want to add 1 for every token so these don't overlap
    sentences_tokenized = []
    lstm.eval()
    for sent in corpus:
        tokenized  = torch.tensor([tokenizer[word] for word in get_tokens_from_sent(sent)])
        sentences_tokenized.append(tokenized)

    representations = []
    with torch.no_grad():
        hidden = lstm.init_hidden(1) # outputs an initial hidden state of zeros
        for sent in sentences_tokenized:
            output = lstm(sent.unsqueeze(0), hidden).squeeze() # Get state activation
            representations.append(output)

    return representations
### Transformer FETCH SEN REPR
def fetch_sen_repr_transformer(corpus: List[TokenList], model, tokenizer) -> List[Tensor]:
    model.eval() # set model in eval mode
    corpus_result = []

    for sent in corpus:
        add_space = False
        tokenized: List[Tensor] = []
        for i, w in enumerate(get_tokens_from_sent(sent)):
            if add_space == False:
                tokenized.append(torch.tensor(tokenizer.encode(w)))
            else:
                tokenized.append(torch.tensor(tokenizer.encode(" " + w)))

            if sent[i]['misc'] == None:
                add_space = True
            else:
                add_space = False

        # get the amount of tokens per word
        subwords_per_token: List[int] = [len(token_list) for (token_list) in tokenized]

        # Concatenate the subwords and feed to model
        with torch.no_grad():
            inp = torch.cat(tokenized)
            out = model(inp)

        h_states = out[0]

        # Form output per token by averaging over all subwords
        result = []
        current_idx = 0
        for nr_tokens in subwords_per_token:
            subword_range = (current_idx, current_idx + nr_tokens)
            current_idx += nr_tokens
            word_hidden_states = h_states[subword_range[0]:subword_range[1], :]
            token_tensor = word_hidden_states.mean(0)

            result.append(token_tensor)

        output_tokens = torch.stack(result)

        # Test shape output
        assert output_tokens.shape[0] is len(tokenized)

        corpus_result.append(output_tokens)

    return corpus_result


# %%
# FETCH SENTENCE REPRESENTATIONS
from torch import Tensor
import pickle

# Should return a tensor of shape (num_tokens_in_corpus, representation_size)
# Make sure you correctly average the subword representations that belong to 1 token!

def fetch_sen_reps(ud_parses: List[TokenList], model, tokenizer, concat=False) -> List[Tensor]:
    if 'RNN' in str(model):
        return fetch_sen_repr_lstm(ud_parses, model, tokenizer)

    return fetch_sen_repr_transformer(ud_parses, model, tokenizer)

# I provide the following sanity check, that compares your representations against a pickled version of mine.
# Note that I use the DistilGPT-2 LM here. For the LSTM I used 0-valued initial states.
def assert_sen_reps(transformer_model, transformer_tokenizer, lstm_model, lstm_tokenizer):
    with open('lstm_emb1.pickle', 'rb') as f:
        lstm_emb1: torch.Tensor = pickle.load(f)
    with open('distilgpt2_emb1.pickle', 'rb') as f:
        distilgpt2_emb1: torch.Tensor = pickle.load(f)

    corpus = parse_corpus('data/sample/en_ewt-ud-train.conllu')

    own_gpt2_emb1: torch.Tensor = fetch_sen_reps(corpus, transformer_model, transformer_tokenizer)[0]
    own_lstm_emb1: torch.Tensor = fetch_sen_reps(corpus, lstm_model, lstm_tokenizer)[0]

    assert distilgpt2_emb1.shape == own_gpt2_emb1.shape, "GPT2 Shapes don't match!"
    assert lstm_emb1.shape == own_lstm_emb1.shape, "LSTM Shapes don't match!"

    assert torch.allclose(distilgpt2_emb1, own_gpt2_emb1, atol=1e-04), "GPT2 Embeddings don't match!"
    assert torch.allclose(lstm_emb1, own_lstm_emb1, atol=1e-04), "LSTM Embeddings don't match!"

    print('Passed basic checks!')

assert_sen_reps(model, tokenizer, lstm, vocab)

# %%
# 🏁
from typing import Tuple

def fetch_pos_tags(ud_parses: List[TokenList], pos_vocab: Optional[DefaultDict[str, int]] = None) -> Tuple[List[Tensor], DefaultDict]:
    """
    Converts `ud_parses` into a tensor of POS tags.
    """
    # If `pos_vocab` is not known, make one based on all POS tokens in `ud_parses`
    if (pos_vocab is None):
        all_pos_tokens = set([pos for sent in ud_parses for pos in get_pos_from_sent(sent)])
        pos2i = {'<pad>': 0, '<unk>': 1, **{pos.strip(): i + 2 for i, pos in enumerate(all_pos_tokens)}}
        print(pos2i)
        pos_vocab = defaultdict(lambda: pos2i["<unk>"])
        pos_vocab.update(pos2i)
        print(pos_vocab)
    pos_tokens_result: List[Tensor] = []

    sent: TokenList
    for sent in ud_parses:
        pos_tokens = torch.tensor([pos_vocab[pos] for pos in get_pos_from_sent(sent)])
        pos_tokens_result.append(pos_tokens)

    return pos_tokens_result, pos_vocab


# %%
import os

# Function that combines the previous functions, and creates 2 tensors for a .conllu file:
# 1 containing the token representations, and 1 containing the (tokenized) pos_tags.

def create_data(filename: str, lm, w2i: Dict[str, int], pos_vocab=None):
    ud_parses = parse_corpus(filename)
    sen_reps = fetch_sen_reps(ud_parses, lm, w2i)
    pos_tags, pos_vocab = fetch_pos_tags(ud_parses, pos_vocab=pos_vocab)

    return sen_reps, pos_tags, pos_vocab


lm = model  # or `lstm`
w2i = tokenizer  # or `vocab`
use_sample = True

train_X, train_y, train_vocab = create_data(
    os.path.join('data', 'sample' if use_sample else '', 'en_ewt-ud-train.conllu'),
    lm,
    w2i
)
# print('\n\n')
# print('Train Vocab')
# print(len(train_vocab))
# print(train_vocab)

dev_x, dev_y, _ = create_data(
    os.path.join('data', 'sample' if use_sample else '', 'en_ewt-ud-dev.conllu'),
    lm,
    w2i,
    pos_vocab=train_vocab
)

# test_x, test_y, _ = create_data(
#     os.path.join('data', 'sample' if use_sample else '', 'en_ewt-ud-test.conllu'),
#     lm,
#     w2i,
#     pos_vocab=train_vocab
# )

# %%
# Utility functions for transforming X and y to appropriate formats
def transform_XY_to_concat_tensors(X: List[Tensor], y: List[Tensor]) -> Tuple[Tensor, Tensor]:
    X_concat = torch.cat(X, dim=0)
    y_concat = torch.cat(y, dim=0)

    return X_concat, y_concat

def transform_XY_to_padded_tensors(X: List[Tensor], y: List[Tensor]) -> Tuple[Tensor, Tensor]:
    X_padded = pad_sequence(X)
    y_padded = pad_sequence(y)

    return X_padded, y_padded

# %% [markdown]
# # Diagnostic Classification
#
# We now have our models, our data, _and_ our representations all set! Hurray, well done. We can finally move onto the cool stuff, i.e. training the diagnostic classifiers (DCs).
#
# DCs are simple in their complexity on purpose. To read more about why this is the case you could already have a look at the "Designing and Interpreting Probes with Control Tasks" by Hewitt and Liang (esp. Sec. 3.2).
#
# A simple linear classifier will suffice for now, don't bother with adding fancy non-linearities to it.
#
# I am personally a fan of the `skorch` library, that provides `sklearn`-like functionalities for training `torch` models, but you are free to train your dc using whatever method you prefer.
#
# As this is an Artificial Intelligence master and you have all done ML1 + DL, I expect you to use your train/dev/test splits correctly ;-)

# %%
import torch.nn as nn

# 🏁
class SimpleProbe(nn.Module):
    def __init__(
        self,
        hidden_dim,
        out_dim
    ):
        super().__init__()
        self.h2out = nn.Linear(hidden_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        logits = self.h2out(X)

        return self.softmax(logits)

# %%
# DIAGNOSTIC CLASSIFIER
# 🏁
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, Checkpoint, TrainEndCheckpoint, LoadInitState
from skorch.dataset import Dataset
from skorch.helper import predefined_split

# print('\n\n')
# print('Train Vocab')
# print(len(train_vocab))
# print(train_vocab)
# Fixed hidden_size
hidden_size = 768
vocab_size = len(train_vocab)#17

# Probe
probe: nn.Module = SimpleProbe(
    hidden_size,
    vocab_size
)

load_model = False

train_acc = EpochScoring(scoring='accuracy', on_train=True,
                         name='train_acc', lower_is_better=False)

cp = Checkpoint(dirname=config.path_to_POS_Probe(1))
train_end_cp = TrainEndCheckpoint(dirname=config.path_to_POS_Probe(1))

load_state = LoadInitState(cp)

callbacks = [train_acc, cp, train_end_cp]

if load_model == True:
    callbacks.append(load_state)

# Concatenate all the tensors
concat_X, concat_Y = transform_XY_to_concat_tensors(train_X, train_y)
valid_X, valid_y = transform_XY_to_concat_tensors(dev_x, dev_y)
valid_ds = Dataset(valid_X, valid_y)

# Have a trainer
# TODO: Add a train/validation split
net: NeuralNetClassifier = NeuralNetClassifier(
    probe,
    callbacks=callbacks,
    max_epochs=10,
    batch_size=8,
    lr=0.0001,
    train_split=predefined_split(valid_ds),
    iterator_train__shuffle=True,
    optimizer= torch.optim.Adam,
)



if config.will_train_simple_probe:
    print('Traing POS probe')
    net.fit(concat_X, concat_Y)
    print('Done Training Probe')

# %%
# A quick test to check the accuracy of the probe
#concat_test_X, concat_test_Y = transform_XY_to_concat_tensors(test_x, test_y)
import numpy as np

predictions = net.predict(concat_X)

golden_tags = concat_Y


for i in range(10):
    print(f'Pred {predictions[i]} | {concat_Y[i]} Gold')
    # print(train_y_probs[i])
    # print(train_y_probs[i].sum())
    # print(np.argmax(train_y_probs[i]))
    # print(all_predictions[i])
# print(golden_tags.shape)
# print(type(train_y_probs))

# print(train_y_probs.shape)
# print(train_y_probs[0].shape)
# print(train_y_probs[0])
# print(train_y_probs[0].sum())
# print(train_y_probs[0].max())
# print(np.argmax(train_y_probs[0]))
# print(train_vocab)
# print(8 in train_vocab)


# %% [markdown]
# # Trees
#
# For our gold labels, we need to recover the node distances from our parse tree. For this we will use the functionality provided by `ete3`, that allows us to compute that directly. I have provided code that transforms a `TokenTree` to a `Tree` in `ete3` format.

# %%
# In case you want to transform your conllu tree to an nltk.Tree, for better visualisation

def rec_tokentree_to_nltk(tokentree):
    token = tokentree.token["form"]
    tree_str = f"({token} {' '.join(rec_tokentree_to_nltk(t) for t in tokentree.children)})"

    return tree_str


def tokentree_to_nltk(tokentree):
    from nltk import Tree as NLTKTree

    tree_str = rec_tokentree_to_nltk(tokentree)

    return NLTKTree.fromstring(tree_str)


# %%
# # !pip install ete3
from ete3 import Tree as EteTree


class FancyTree(EteTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, format=1, **kwargs)

    def __str__(self):
        return self.get_ascii(show_internal=True)

    def __repr__(self):
        return str(self)


def rec_tokentree_to_ete(tokentree):
    idx = str(tokentree.token["id"])
    children = tokentree.children
    if children:
        return f"({','.join(rec_tokentree_to_ete(t) for t in children)}){idx}"
    else:
        return idx

def tokentree_to_ete(tokentree):
    newick_str = rec_tokentree_to_ete(tokentree)

    return FancyTree(f"{newick_str};")


# %%
# Let's check if it works!
# We can read in a corpus using the code that was already provided, and convert it to an ete3 Tree.
corpus = parse_corpus('data/sample/en_ewt-ud-train.conllu')
sent = corpus[0]
tokentree = sent.to_tree()

ete3_tree = tokentree_to_ete(tokentree)
print(ete3_tree)


# %% [markdown]
# As you can see we label a token by its token id (converted to a string). Based on these id's we are going to retrieve the node distances.
#
# To create the true distances of a parse tree in our treebank, we are going to use the `.get_distance` method that is provided by `ete3`: http://etetoolkit.org/docs/latest/tutorial/tutorial_trees.html#working-with-branch-distances
#
# We will store all these distances in a `torch.Tensor`.
#
# Please fill in the gap in the following method. I recommend you to have a good look at Hewitt's blog post  about these node distances.

# %%
def create_gold_distances(corpus) -> List[Tensor]:
    all_distances: List[Tensor] = []

    for sent in (corpus):
        tokentree = sent.to_tree()
        ete_tree = tokentree_to_ete(tokentree)

        sen_len = len(ete_tree.search_nodes())
        distances = torch.zeros((sen_len, sen_len))

        # 🏁
        token_ids = get_ids_from_sent(sent)

        # Go over all the token ids, as row and as columns
        for i, token_id_A in enumerate(token_ids):
            for j, token_id_B in enumerate(token_ids):

                # Set distance to 0 if they are equal
                if token_id_A == token_id_B:
                    distances[i, j] = 0
                else:
                    # Else use `.get_distance` to calculate the distance for row A and column B
                    A = str(token_id_A)
                    B = str(token_id_B)
                    distances[i, j] = ete_tree.get_distance(A, B)

        all_distances.append(distances)

    return all_distances

# TODO: Check if we did these distances properly
distances = create_gold_distances(sample_corpus)

# %%
sample_feature = train_X[0]
sample_feature = sample_feature.unsqueeze(0)
seqlen = sample_feature.shape[1]
transformed = sample_feature.unsqueeze(2)
transformed = transformed.expand(-1, -1, seqlen, -1)
transformed.shape

# %% [markdown]
# The next step is now to do the previous step the other way around. After all, we are mainly interested in predicting the node distances of a sentence, in order to recreate the corresponding parse tree.
#
# Hewitt et al. reconstruct a parse tree based on a _minimum spanning tree_ (MST, https://en.wikipedia.org/wiki/Minimum_spanning_tree). Fortunately for us, we can simply import a method from `scipy` that retrieves this MST.

# %%
from scipy.sparse.csgraph import minimum_spanning_tree
import torch

def create_mst(distances):
    distances = torch.triu(distances).detach().numpy()

    mst = minimum_spanning_tree(distances).toarray()
    mst[mst>0] = 1.

    return mst


# %% [markdown]
# Let's have a look at what this looks like, by looking at a relatively short sentence in the sample corpus.
#
# If your addition to the `create_gold_distances` method has been correct, you should be able to run the following snippet. This then shows you the original parse tree, the distances between the nodes, and the MST that is retrieved from these distances. Can you spot the edges in the MST matrix that correspond to the edges in the parse tree?

# %%
item = corpus[5]
tokentree = item.to_tree()
ete3_tree = tokentree_to_ete(tokentree)
print(ete3_tree, '\n')

gold_distance = create_gold_distances(corpus[5:6])[0]
print(gold_distance, '\n')

mst = create_mst(gold_distance)
print(mst)

# %%
# Utility cell to play around with the values
all_edges = list(zip(mst.nonzero()[0], mst.nonzero()[1]))
all_edges = set(tuple(frozenset(sub)) for sub in set(all_edges))
all_edges

# %% [markdown]
# Now that we are able to map edge distances back to parse trees, we can create code for our quantitative evaluation. For this we will use the Undirected Unlabeled Attachment Score (UUAS), which is expressed as:
#
# $$\frac{\text{number of predicted edges that are an edge in the gold parse tree}}{\text{number of edges in the gold parse tree}}$$
#
# To do this, we will need to obtain all the edges from our MST matrix. Note that, since we are using undirected trees, that an edge can be expressed in 2 ways: an edge between node $i$ and node $j$ is denoted by both `mst[i,j] = 1`, or `mst[j,i] = 1`.
#
# You will write code that computes the UUAS score for a matrix of predicted distances, and the corresponding gold distances. I recommend you to split this up into 2 methods: 1 that retrieves the edges that are present in an MST matrix, and one general method that computes the UUAS score.

# %%
# 🏁
# Utility function: check if edge is in set of edges
check_edge_in_edgeset = lambda edge, edgeset: (edge[0], edge[1]) in edgeset or (edge[1], edge[0]) in edgeset

def edges(mst):
    """
    Get all edges from a minimum spanning tree `mst`
    """

    all_edges = list(zip(mst.nonzero()[0], mst.nonzero()[1]))

    # Ensure (A, B) and (B, A) only return one combination
    all_edges = set(tuple(frozenset(sub)) for sub in set(all_edges))

    return all_edges

def calc_uuas(pred_distances, gold_distances):
    uuas = None

    # Get MSTs from distances
    pred_mst = create_mst(pred_distances)
    gold_mst = create_mst(gold_distances)

    # Convert MSTs to edges
    pred_edges = edges(pred_mst)
    gold_edges = edges(gold_mst)

    # Calculate UUAS
    nr_correct_edges =len(
        [pred_edge for pred_edge in pred_edges
         if check_edge_in_edgeset(pred_edge, gold_edges)]
    )

    uuas = nr_correct_edges / len(gold_edges)

    return uuas

calc_uuas(gold_distance, gold_distance)

# %% [markdown]
# # Structural Probes
#
# We now have everything in place to start doing the actual exciting stuff: training our structural probe!
#
# To make life easier for you, we will simply take the `torch` code for this probe from John Hewitt's repository. This allows you to focus on the training regime from now on.

# %%
import torch.nn as nn
import torch


class StructuralProbe(nn.Module):
    """ Computes squared L2 distance after projection by a matrix.
    For a batch of sentences, computes all n^2 pairs of distances
    for each sentence in the batch.
    """
    def __init__(self, model_dim, rank, device="cpu"):
        super().__init__()
        self.probe_rank = rank
        self.model_dim = model_dim

        self.proj = nn.Parameter(
            data = torch.zeros(self.model_dim, self.probe_rank),
            requires_grad=True
        )

        nn.init.uniform_(self.proj, -0.05, 0.05)
        self.to(device)

    def forward(self, batch):
        """ Computes all n^2 pairs of distances after projection
        for each sentence in a batch.
        Note that due to padding, some distances will be non-zero for pads.
        Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j
        Args:
          batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
          A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
        """
        transformed = torch.matmul(batch, self.proj)

        batchlen, seqlen, rank = transformed.size()

        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1,2)

        diffs = transformed - transposed

        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)

        return squared_distances


class L1DistanceLoss(nn.Module):
    """Custom L1 loss for distance matrices."""
    def __init__(self):
        super().__init__()

    def forward(self, predictions, label_batch, length_batch):
        """ Computes L1 loss on distance matrices.
        Ignores all entries where label_batch=-1
        Normalizes first within sentences (by dividing by the square of the sentence length)
        and then across the batch.
        Args:
          predictions: A pytorch batch of predicted distances
          label_batch: A pytorch batch of true distances
          length_batch: A pytorch batch of sentence lengths
        Returns:
          A tuple of:
            batch_loss: average loss in the batch
            total_sents: number of sentences in the batch
        """
        labels_1s = (label_batch != -1).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_batch * labels_1s
        total_sents = torch.sum((length_batch != 0)).float()
        squared_lengths = length_batch.pow(2).float()

        if total_sents > 0:
            loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=(1,2))
            normalized_loss_per_sent = loss_per_sent / squared_lengths
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents

        else:
            batch_loss = torch.tensor(0.0)

        return batch_loss, total_sents



# %% [markdown]
# I have provided a rough outline for the training regime that you can use. Note that the hyper parameters that I provide here only serve as an indication, but should be (briefly) explored by yourself.
#
# As can be seen in Hewitt's code above, there exists functionality in the probe to deal with batched input. It is up to you to use that: a (less efficient) method can still incorporate batches by doing multiple forward passes for a batch and computing the backward pass only once for the summed losses of all these forward passes. (_I know, this is not the way to go, but in the interest of time that is allowed ;-), the purpose of the assignment is writing a good paper after all_).

# %%
# 🏁
from torch.utils.data import DataLoader, Dataset

class ProbingDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.features[idx], self.targets[idx])

def custom_collate_fn(items: List[Tensor]) -> List[Tensor]:
    return items

# %%
# Utility functions for dependency parsing

get_parent_children_combo_from_tree = lambda tree: [
    (i.up.name, (i.name)) 
    for i in tokentree_to_ete(tree).search_nodes() 
    if i.up is not None
]

map_pc_combo_to_parent_edges = lambda id_idx_map, tuple_list: [(id_idx_map[int(i[0])], id_idx_map[int(i[1])]) for i in tuple_list]

def create_gold_parent_distance_edges(corpus: List[TokenList]):
    """
    Assigns for each sentence, a 1 for child (row) to parent (column).
    """
    all_edges: List[Tensor] = []
        
    for sent in corpus:
        sent_tree = sent.to_tree()
        sent_id2idx = {i['id']: idx for idx, i in enumerate(sent)}
        
        # Calculate tuples of (parent, child), and then map their ID to their respective indices         
        parent_child_tuples = get_parent_children_from_tree(sent_tree)
        parent_child_tuples = map_pc_combo_to_parent_edges(sent_id2idx, parent_child_tuples)
        
        # Initialize our matrix         
        sen_len = len(sent)
        distances = torch.zeros((sen_len, sen_len))
        
        for parent_edge in parent_child_tuples:
            parent_idx, child_idx = parent_edge
            distances[child_idx, parent_idx] = 1
        
        all_edges.append(distances)
    
    return all_edges

def create_gold_parent_distance_idxs_only(corpus: List[TokenList]) -> List[Tuple[Tensor, int]]:
    """
    Assigns for each sentence the index of the parent node.
    If node is -1, then that node has no parent (should be ignored).
    
    Input:
        - corpus: list of TokenLists
    Output:
        List of tuple(gold indices of parent, index of root)
    """
    all_edges: List[Tuple[Tensor, int]] = []
        
    print(len(corpus))
    
    for sent in corpus:
        sent_tree = sent.to_tree()
        sent_id2idx = {i['id']: idx for idx, i in enumerate(sent)}
        
        # Calculate tuples of (parent, child), and then map their ID to their respective indices         
        parent_child_tuples = get_parent_children_from_tree(sent_tree)
        parent_child_tuples = map_pc_combo_to_parent_edges(sent_id2idx, parent_child_tuples)
        
        # Initialize our matrix         
        sen_len = len(sent)
        edges = torch.zeros((sen_len))
        
        root_idx = sent_id2idx[sent_tree.token['id']]
        
        # For each edge, assign in the corresponding index the parent index, and -1 for root         
        for parent_edge in parent_child_tuples:
            parent_idx, child_idx = parent_edge
            edges[child_idx] = parent_idx
            edges[root_idx] = -1
        
        all_edges.append((edges, root_idx))
    
    return all_edges


# %%
def init_corpus(path, model=model, concat=False, cutoff=None, use_dependencies=False):
    """ Initialises the data of a corpus.

    Parameters
    ----------
    path : str
        Path to corpus location
    concat : bool, optional
        Optional toggle to concatenate all the tensors
        returned by `fetch_sen_reps`.
    cutoff : int, optional
        Optional integer to "cutoff" the data in the corpus.
        This allows only a subset to be used, alleviating
        memory usage.
    """
    corpus: List[TokenList] = parse_corpus(path)[:cutoff]

    embs: List[Tensor] = fetch_sen_reps(corpus, model, tokenizer, concat=concat)
#     max_length_size = embs.shape[0]

#     embs = embs.reshape(corpus_size, max_length_size, -1)
    if not use_dependencies:
        gold_distances = create_gold_distances(corpus)
    else:
        gold_distances = create_gold_parent_distance_idxs_only(corpus)

    return gold_distances, embs

def init_dataloader_sequential(path: str, batch_size: int, cutoff=None) -> DataLoader:
    y, X = init_corpus(path)
    dataset = ProbingDataset(X, y)
    data_loader = DataLoader(
        dataset=dataset,
        collate_fn=custom_collate_fn,
        batch_size=batch_size
    )

    return data_loader

# %%
# Prep data-loaders
train_dataloader = init_dataloader_sequential(config.path_to_data_train, config.struct_probe_train_batch_size)
valid_dataloader = init_dataloader_sequential('data/sample/en_ewt-ud-dev.conllu', 1)

# %%
from torch import optim
def evaluate_probe(probe, data_loader):
    loss_scores: List[Tensor] = []
    uuas_scores: List[Tensor] = []
    loss_score = 0
    uuas_score = 0

    probe.eval()
    loss_function = L1DistanceLoss()

    with torch.no_grad():
        for idx, batch_item in enumerate(data_loader):
            for item in batch_item:
                X, y = item

                if len(X.shape) == 2:
                    X = X.unsqueeze(0)

                # Sentences with strange tokens, we ignore for the moment
                if len(y) < 2:
                    print(f"Encountered: null sentence at idx {idx}")
                    continue

                pred_distances = probe(X)
                item_loss, _ = loss_function(pred_distances, y, torch.tensor(len(y)))

                if len(pred_distances.shape) > 2:
                    pred_distances = pred_distances.squeeze(0)

                uuas = calc_uuas(pred_distances, y)

                loss_score += item_loss.item()
                loss_scores.append(torch.tensor(item_loss.item()))
                uuas_scores.append(torch.tensor(uuas, dtype=torch.float))

    loss_score = torch.mean(torch.stack(loss_scores))
    uuas_score = torch.mean(torch.stack(uuas_scores))
    print(f"Average evaluation loss score is {loss_score}")
    print(f"Average evaluation uuas score is {uuas_score}")

    return loss_score, uuas_score

def train(
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    config: Config
):
    emb_dim: int = config.struct_probe_emb_dim
    rank: int = config.struct_probe_rank
    lr: float = config.struct_probe_lr
    epochs: int = config.struct_probe_train_epoch

    probe: nn.Module = StructuralProbe(emb_dim, rank)

    # Training tools
    optimizer = optim.Adam(probe.parameters(), lr=lr)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,patience=1)
    loss_function = L1DistanceLoss()

    for epoch in range(epochs):
        print(f"\n---- EPOCH {epoch + 1} ---- \n")
        for train_batch in train_dataloader:
            # Setup
            probe.train()
            optimizer.zero_grad()

            batch_loss = torch.tensor([0.0])

            for train_item in train_batch:
                train_X, train_y = train_item
                train_X, train_y = train_item

                if len(train_X.shape) == 2:
                    train_X = train_X.unsqueeze(0)

                pred_distances = probe(train_X)
                item_loss, _ = loss_function(pred_distances, train_y, torch.tensor(len(train_y)))
                batch_loss += item_loss

            batch_loss.backward()
            optimizer.step()

        # Calculate validation scores
        # TODO: Double-check that the UUAS works
        valid_loss, valid_uuas = evaluate_probe(probe, valid_dataloader)

        # TODO: Optional Param-tune scheduler (?)
#         scheduler.step(valid_loss)

    return probe

train(train_dataloader, valid_dataloader, config)


# %% [markdown]
# # Structural Probing Control Task

# %%

class TwoWordBilinearLabelProbe(nn.Module):
  """ Computes a bilinear function of pairs of vectors.
  For a batch of sentences, computes all n^2 pairs of scores
  for each sentence in the batch.
  """
  def __init__(
      self,
      max_rank: int,
      feature_dim: int,
      dropout_p: float
  ):
    super().__init__()
    self.maximum_rank = max_rank
    self.feature_dim = feature_dim
    self.proj_L = nn.Parameter(data = torch.zeros(self.feature_dim, self.maximum_rank), requires_grad=True)
    self.proj_R = nn.Parameter(data = torch.zeros(self.maximum_rank, self.feature_dim), requires_grad=True)
    self.bias = nn.Parameter(data=torch.zeros(1), requires_grad=True)

    nn.init.uniform_(self.proj_L, -0.05, 0.05)
    nn.init.uniform_(self.proj_R, -0.05, 0.05)
    nn.init.uniform_(self.bias, -0.05, 0.05)

    self.dropout = nn.Dropout(p=dropout_p)
    self.softmax = nn.Softmax(2)

  def forward(self, batch):
    """ Computes all n^2 pairs of attachment scores
    for each sentence in a batch.
    Computes h_i^TAh_j for all i,j
    where A = LR, L in R^{model_dim x maximum_rank}; R in R^{maximum_rank x model_rank}
    hence A is rank-constrained to maximum_rank.
    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of scores of shape (batch_size, max_seq_len, max_seq_len)
    """
    batchlen, seqlen, rank = batch.size()
    batch = self.dropout(batch)
    # A/Proj = L * R
    proj = torch.mm(self.proj_L, self.proj_R)

    # Expand matrix to allow another instance of all cols/rows
    batch_square = batch.unsqueeze(2).expand(batchlen, seqlen, seqlen, rank)

    # Add another instance of seqlen (do it on position 1 for some reason)
    # => change view so that it becomes rows of 'rank/hidden-dim'
    batch_transposed = batch.unsqueeze(1).expand(batchlen, seqlen, seqlen, rank).contiguous().view(batchlen*seqlen*seqlen,rank,1)

    # Multiply the `batch_square` matrix with the projection
    psd_transformed = torch.matmul(batch_square.contiguous(), proj).view(batchlen*seqlen*seqlen,1, rank)

    # Multiple resulting matrix `psd_transform` with each j
    logits = (torch.bmm(psd_transformed, batch_transposed) + self.bias).view(batchlen, seqlen, seqlen)

    probs = self.softmax(logits)
    return probs


# %%
probe = TwoWordBilinearLabelProbe(
    max_rank=64,
    feature_dim=768,
    dropout_p=0.5
)

# %%
pred = probe(sample_feature)

# %%
sample_tree = sample_sent.to_tree()
sample_ete_tree = tokentree_to_ete(sample_tree)


def get_children_from_parent(token_tree, results=[]):
    token_id = token_tree.token['id']
    children_ids = [child.token['id'] for child in sample_tree.children]
    
    if len(len(sample_tree.token.children) > 0):
        local_results = get_children_from_parent(local_results)
    
    return (token_id, children_ids)


# %%
from torch.nn import NLLLoss

def evaluate_dep_probe(probe, data_loader):
    loss_scores: List[Tensor] = []
    loss_score = 0

    probe.eval()
    loss_function = NLLLoss()

    with torch.no_grad():
        for idx, batch_item in enumerate(data_loader):
            for item in batch_item:
                valid_X, valid_y_tup = item
                
                valid_X = valid_X.unsqueeze(0)
                valid_y, valid_y_root_idx = valid_y_tup
                
                pred = probe(valid_X).squeeze()
                           # Sentences with strange tokens, we ignore for the moment
                if len(valid_y) < 2:
                    print(f"Encountered: null sentence at idx {idx}")
                    continue
                    
                masked_idx = valid_y != -1
                masked_pred = pred[masked_idx]
                masked_y = valid_y[masked_idx].long()
                
                item_loss = loss_function(masked_pred, masked_y)

 

                loss_scores.append(torch.tensor(item_loss.item()))

    loss_score = torch.mean(torch.stack(loss_scores))
    print(f"Average evaluation loss score is {loss_score}")

    return loss_score


def train_dep_parsing(
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    config: Config
):
    emb_dim: int = config.struct_probe_emb_dim
    rank: int = config.struct_probe_rank
    lr: float = config.struct_probe_lr
    epochs: int = config.struct_probe_train_epoch

    probe: nn.Module = TwoWordBilinearLabelProbe(
        max_rank=rank,
        feature_dim=emb_dim,
        dropout_p=0.5
    )

    # Training tools
    optimizer = optim.Adam(probe.parameters(), lr=lr)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,patience=1)
    loss_function = NLLLoss()

    for epoch in range(epochs):
        print(f"\n---- EPOCH {epoch + 1} ---- \n")
        
        for train_batch in train_dataloader:
            # Setup
            probe.train()
            optimizer.zero_grad()

            batch_loss = torch.tensor([0.0])

            for train_item in train_batch:
                train_X, train_y_tup = train_item
                
                train_X = train_X.unsqueeze(0)
                train_y, train_y_root_idx = train_y_tup
                
                pred = probe(train_X).squeeze()

                
                masked_idx = train_y != -1
                masked_pred = pred[masked_idx]
                masked_y = train_y[masked_idx].long()
                
                item_loss = loss_function(masked_pred, masked_y)
                batch_loss += item_loss

            batch_loss.backward()
            optimizer.step()

        # Calculate validation scores
        # TODO: Double-check that the UUAS works
        valid_loss = evaluate_dep_probe(probe, valid_dataloader)

        # TODO: Optional Param-tune scheduler (?)
#         scheduler.step(valid_loss)

    return probe


# %%
train_dep_parsing(
    train_dataloader,
    valid_dataloader,
    config
)

# %%
train_data_raw = init_corpus(config.path_to_data_train, use_dependencies=True)
train_dataset = ProbingDataset(train_data_raw[1], train_data_raw[0])
train_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=custom_collate_fn)

valid_data_raw = init_corpus(config.path_to_data_valid, use_dependencies=True)
valid_dataset = ProbingDataset(valid_data_raw[1], valid_data_raw[0])
valid_dataloader = DataLoader(valid_dataset, batch_size=8, collate_fn=custom_collate_fn)

# %%
sample_probe = TwoWordBilinearLabelProbe(64, 768, 0.5)
sample_X = train_data_raw[1][0].unsqueeze(0)
sample_Y, sample_Y_root_idx = train_data_raw[0][0]
sample_pred = sample_probe(sample_X).squeeze()

# Mask idx of the root
masked_idx = sample_Y != -1
masked_pred = sample_pred[masked_idx]
masked_y = sample_Y[masked_idx].long()



# %%
masked_y
