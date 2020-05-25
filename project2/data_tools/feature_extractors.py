from typing import List, DefaultDict, Callable
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from conllu import TokenList

get_tokens_from_sent: Callable[[TokenList], List[str]] = lambda sent: [word['form'] for word in sent]

def fetch_sen_reps(ud_parses: List[TokenList], model, tokenizer, concat=False) -> List[Tensor]:
    """
    Fetches feature representations from a corpus `ud_parse` either from an LSTM or Tokenizer.
    """
    if 'RNN' in type(model).__name__:
        return fetch_sen_repr_lstm(ud_parses, model, tokenizer)

    return fetch_sen_repr_transformer(ud_parses, model, tokenizer)

def fetch_sen_repr_lstm(corpus: List[TokenList], lstm, tokenizer) -> List[Tensor]:
    """
    Fetches feature representations from a corpus, using an LSTM and its tokenizer.
    """
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
            if output.shape[0] == 650:
                output = output.view(1,650)
            representations.append(output)

    return representations

def fetch_sen_repr_transformer(corpus: List[TokenList], trans_model, tokenizer) -> List[Tensor]:
    """
    Fetches feature representations from a corpus, using a Transformer model and its tokenizer.
    """
    trans_model.eval() # set model in eval mode
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
            out = trans_model(inp)

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
