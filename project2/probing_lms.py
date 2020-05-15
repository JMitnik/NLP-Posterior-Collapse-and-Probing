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
#     display_name: xpython
#     language: python
#     name: xpython
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
# Custom Configuration
from config import Config
config = Config(
    will_train_simple_probe=False
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
from typing import List
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
# Utility functions
get_pos_from_sent: List[str] = lambda sent: [word['upostag'] for word in sent]
get_tokens_from_sent: List[str] = lambda sent: [word['form'] for word in sent]
get_ids_from_sent: List[str] = lambda sent: [word['id'] for word in sent]

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

from typing import List
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor

def fetch_sen_repr_lstm(corpus: List[TokenList], lstm, tokenizer) -> Tensor:
    # TODO: Needs to be fixed similarily to transformer
    result = []

    for sent in corpus:
        tokenized = torch.tensor([tokenizer[word] for word in get_tokens_from_sent(corpus)])
        result = result.append(tokenized)

    result = pad_sequence(result)

    # Get state activation
    with torch.no_grad():
        hidden = lstm.init_hidden(1)
        output = lstm(sample_tensor.unsqueeze(0), hidden).squeeze()

    return output

def fetch_sen_repr_transformer(corpus: List[TokenList], model, tokenizer) -> Tensor:
    corpus_result = []

    for sent in corpus:
        # Running and debugging the transformer
        tokenized: List[Tensor] = [torch.tensor(tokenizer.encode(word)) for word in get_tokens_from_sent(sent)]
        subwords_per_token: List[int] = [len(token_list) for (token_list) in tokenized]

        # Concatenate the subwords and feed to model
        with torch.no_grad():
            inp = torch.cat(tokenized)
            model.eval()
            out = model(inp)

        # Study the first output (last_hidden_state = nr_words x representation)
        h_states = out[0]

        # Form output per token by averaging over all subwords
        starting_idx = 0
        result = []
        for nr_tokens in subwords_per_token:
            subwords_per_token_tensor = h_states[starting_idx:starting_idx+nr_tokens, :]
            token_tensor = subwords_per_token_tensor.mean(0)
            result.append(token_tensor)
            starting_idx += nr_tokens

        output_tokens = torch.stack(result)

        # Test shape output
        assert output_tokens.shape[0] is len(tokenized)

        corpus_result.append(output_tokens)

    # TODO: Do we pad sequence or something else?
#     return pad_sequence(corpus_result)
    return pad_sequence(corpus_result)


# %%
# FETCH SENTENCE REPRESENTATIONS
from torch import Tensor
import pickle


# Should return a tensor of shape (num_tokens_in_corpus, representation_size)
# Make sure you correctly average the subword representations that belong to 1 token!

def fetch_sen_reps(ud_parses: List[TokenList], model, tokenizer, concat=False) -> Tensor:
    if 'RNN' in str(model):
        return fetch_sen_repr_lstm(ud_parses, model, tokenizer)

    return fetch_sen_repr_transformer(ud_parses, model, tokenizer)

# I provide the following sanity check, that compares your representations against a pickled version of mine.
# Note that I use the DistilGPT-2 LM here. For the LSTM I used 0-valued initial states.
def assert_sen_reps(model, tokenizer, lstm, vocab):
    with open('distilgpt2_emb1.pickle', 'rb') as f:
        distilgpt2_emb1 = pickle.load(f)

    with open('lstm_emb1.pickle', 'rb') as f:
        lstm_emb1 = pickle.load(f)

    corpus = parse_corpus('data/sample/en_ewt-ud-train.conllu')[:1]

    own_distilgpt2_emb1 = fetch_sen_reps(corpus, model, tokenizer)
    own_lstm_emb1 = fetch_sen_reps(corpus, lstm, vocab)

    assert distilgpt2_emb1.shape == own_distilgpt2_emb1.shape
    assert lstm_emb1.shape == own_lstm_emb1.shape

    # DEBUGGIN TOOLS
#     print(torch.max(torch.abs(distilgpt2_emb1 - own_distilgpt2_emb1)))

# TODO: These numbers need to match, but from index 3 (': token'), they start to diverge.
#     print(distilgpt2_emb1[:, 0])
#     print(own_distilgpt2_emb1[:, 0])

    assert torch.allclose(distilgpt2_emb1, own_distilgpt2_emb1), "DistilGPT-2 embs don't match!"
    assert torch.allclose(lstm_emb1, own_lstm_emb1), "LSTM embs don't match!"


# %%
# 🏁

def fetch_pos_tags(ud_parses: List[TokenList], pos_vocab=None) -> Tensor:
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

    pos_tokens_result: List[Tensor] = []

    sent: TokenList
    for sent in ud_parses:
        pos_tokens = torch.tensor([pos_vocab[pos] for pos in get_pos_from_sent(sent)])
        pos_tokens_result.append(pos_tokens)

    return pad_sequence(pos_tokens_result), pos_vocab


# %%
import os

# Function that combines the previous functions, and creates 2 tensors for a .conllu file:
# 1 containing the token representations, and 1 containing the (tokenized) pos_tags.

def create_data(filename: str, lm, w2i, pos_vocab=None):
    ud_parses = parse_corpus(filename)
    sen_reps = fetch_sen_reps(ud_parses, lm, w2i)
    pos_tags, pos_vocab = fetch_pos_tags(ud_parses, pos_vocab=pos_vocab)

    print(f"Shape of corpuses for filename {filename} is {sen_reps.shape}")
    print(f"Number of corpuses for filename {filename} is {len(ud_parses)}")
    print(f"Shape of pos_tags for filename {filename} is {pos_tags.shape}")

    return sen_reps, pos_tags, pos_vocab


lm = model  # or `lstm`
w2i = tokenizer  # or `vocab`
use_sample = True

train_x, train_y, train_vocab = create_data(
    os.path.join('data', 'sample' if use_sample else '', 'en_ewt-ud-train.conllu'),
    lm,
    w2i
)

dev_x, dev_y, _ = create_data(
    os.path.join('data', 'sample' if use_sample else '', 'en_ewt-ud-dev.conllu'),
    lm,
    w2i,
    pos_vocab=train_vocab
)

test_x, test_y, _ = create_data(
    os.path.join('data', 'sample' if use_sample else '', 'en_ewt-ud-test.conllu'),
    lm,
    w2i,
    pos_vocab=train_vocab
)

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
from skorch.dataset import Dataset

nr_dataitems = train_x.shape[1]
total_seqlength = train_x.shape[0]
feature_size = train_x.shape[2]

# Shape the data such that each row is a single token
reshaped_train_X = train_x.reshape(nr_dataitems, total_seqlength, feature_size)
reshaped_train_X = reshaped_train_X.reshape(nr_dataitems * total_seqlength, feature_size)
reshaped_train_y = train_y.reshape(nr_dataitems, -1)
reshaped_train_y = reshaped_train_y.reshape(-1)

# Fixed hidden_size
hidden_size = 768
vocab_size = 17

# Probe
probe = SimpleProbe(
    hidden_size,
    vocab_size
)

# Have a trainer
# TODO: Add a train/validation split
net = NeuralNetClassifier(
    probe,
    max_epochs=20,
    batch_size=8,
    train_split=None,
)

if config.will_train_simple_probe:
    # Train the network using Skorch's fit
    net.fit(reshaped_train_X, reshaped_train_y)


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
def create_gold_distances(corpus):
    all_distances = []

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

        self.proj = nn.Parameter(data = torch.zeros(self.model_dim, self.probe_rank))

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



# %% [raw]
# I have provided a rough outline for the training regime that you can use. Note that the hyper parameters that I provide here only serve as an indication, but should be (briefly) explored by yourself.
#
# As can be seen in Hewitt's code above, there exists functionality in the probe to deal with batched input. It is up to you to use that: a (less efficient) method can still incorporate batches by doing multiple forward passes for a batch and computing the backward pass only once for the summed losses of all these forward passes. (_I know, this is not the way to go, but in the interest of time that is allowed ;-), the purpose of the assignment is writing a good paper after all_).

# %%
# 🏁
from torch.utils.data import Dataset, DataLoader

class ProbingDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.features[idx], self.targets[idx])


# %%
def init_corpus(path, concat=False, cutoff=None):
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
    corpus = parse_corpus(path)[:cutoff]
    corpus_size = len(corpus)

    embs = fetch_sen_reps(corpus, model, tokenizer, concat=concat)
#     max_length_size = embs.shape[0]

#     embs = embs.reshape(corpus_size, max_length_size, -1)
    gold_distances = create_gold_distances(corpus)

    return gold_distances, embs



# %%
distances, embs = init_corpus('data/sample/en_ewt-ud-train.conllu')
embs = embs.reshape(embs.shape[1], embs.shape[0], -1)
data = ProbingDataset(embs, distances)


# %%
def custom_collate_fn(items):
    result = []
    for batch_item in items:
        embs = batch_item[0]
        raw_label = batch_item[1]
        length = torch.tensor([raw_label.shape[0]])
        label_placeholder = torch.full((embs.shape[0], embs.shape[0]), -1)
        label_placeholder[0: raw_label.shape[0], 0: raw_label.shape[1]] = raw_label
        result.append((embs, label_placeholder, length))
    return result

train_data_loader = DataLoader(data, collate_fn=custom_collate_fn, batch_size=4)

# %%
from torch import optim

'''
Similar to the `create_data` method of the previous notebook, I recommend you to use a method
that initialises all the data of a corpus. Note that for your embeddings you can use the
`fetch_sen_reps` method again. However, for the POS probe you concatenated all these representations into
1 big tensor of shape (num_tokens_in_corpus, model_dim).

The StructuralProbe expects its input to contain all the representations of 1 sentence, so I recommend you
to update your `fetch_sen_reps` method in a way that it is easy to retrieve all the representations that
correspond to a single sentence.
'''

# I recommend you to write a method that can evaluate the UUAS & loss score for the dev (& test) corpus.
# Feel free to alter the signature of this method.
def evaluate_probe(probe, _data):
    # YOUR CODE HERE

    return loss_score, uuas_score

# Feel free to alter the signature of this method.
def train(
    train_data_loader,
    config=None
):
    emb_dim = 768
    rank = 64
    lr = 10e-4
    batch_size = 24
    epochs = 5

    probe = StructuralProbe(emb_dim, rank)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,patience=1)
    loss_function =  L1DistanceLoss()

    for epoch in range(epochs):
        for train_batch in train_data_loader:
            optimizer.zero_grad()

            batch_loss = torch.Tensor([0])
            for train_item in train_batch:
                train_X, train_y, length = train_item

                if len(train_X.shape) == 2:
                    train_X = train_X.unsqueeze(0)

                pred_distances = probe(train_X)
                item_loss, sents = loss_function(pred_distances, train_y, length)
                batch_loss += item_loss

            batch_loss.backward()
            optimizer.step()

        # - [] Pass in _dev_data
#         dev_loss, dev_uuas = evaluate_probe(probe, _dev_data)

        # Using a scheduler is up to you, and might require some hyper param fine-tuning
#         scheduler.step(dev_loss)

    return probe

train(train_data_loader)
