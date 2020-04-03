# %%
# Imports
import torch
import torch.nn as nn
from typing import List
import importlib
from torchtext.data import Field, get_tokenizer
import models.RNNLM
importlib.reload(models.RNNLM)
from models.RNNLM import RNNLM

# %%
# Temporary cell

# Define field: this is a preprocessing unit from torchtext
data_processor: Field = Field(
    init_token='<BOS>',
    eos_token='<EOS>',
    tokenize=get_tokenizer('basic_english'),
    batch_first=True
)

# Define some corpus to get
training_sentences: List[str] = [
    'As noted above, you can create an nn. Sequential from an OrderedDict with keys as module names and nn.'
]

# Sentences to turn into a padded batch
test_sentences: List[str] = [
    'As you know about keys',
    'As you know about dogs',
]

# Learn from training data
data_processor.build_vocab([data_processor.preprocess(sent) for sent in training_sentences])

# Turn into batch for RNN to walk through
tensor = data_processor.process([data_processor.preprocess(sent) for sent in test_sentences])

# Define model
vocab_size = len(data_processor.vocab)
embedding_size = 50
hidden_size = 50
model = RNNLM(
    vocab_size,
    embedding_size,
    hidden_size
)

# Define loss function
lossf = nn.NLLLoss()
optim = torch.optim.Adam(model.parameters())

loss = 0
# %%
def train_model_on_batch(
    model: RNNLM,
    optim: torch.optim.Optimizer,
    inp: torch.Tensor
):
    optim.zero_grad()
    # inp is to be shaped as Sentences(=batch) x Words
    hidden_state = model.init_hidden(inp)

    for idx in range(inp.shape[1] - 1):
        active_words = inp[:, idx]
        pred, hidden_state = model(active_words, hidden_state)
        next_words = inp[:, idx + 1]

        # TODO: Ensure this works
        local_loss = lossf(torch.log(pred), torch.tensor(next_words))

        loss += local_loss

    loss.backward()
    optim.step()

train_model_on_batch(model, tensor)


# %%
