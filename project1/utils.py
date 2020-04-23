from torch.utils.tensorboard import SummaryWriter
import socket
import torch
import os
import torch.nn.functional as F
import itertools

def store_training_results(
    writer: SummaryWriter,
    name: str,
    loss: torch.Tensor,
    perplexity: torch.Tensor,
    epoch_nr: int
):
    """
    Stores results by doing the following
        - Writes results to tensorboard (store in /runs)

    Arguments:
        loss {torch.Tensor} -- Loss for a training batch
        perplexity {torch.Tensor} -- Perplexity for a training batch
    """
    writer.add_scalar(f'{name}/loss', loss, epoch_nr)
    writer.add_scalar(f'{name}/perplexity', perplexity, epoch_nr)

def save_model(path, model, optimizer, step):
    """
    Save model, optimizer and number of training steps to path.
    """
    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'step': step}
    torch.save(checkpoint, f'models/saved_models/{path}.pt')

def load_model(path, model, device, optimizer=None):
    """
    Load a model and optimizer state to device from path.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, checkpoint['step']


def dict_product(dicts):
    """
    Make all combinations from a dictionary's key and values
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

def make_param_grid(ordered_dict):
    """
    Call dict-product to return all possible combos
    """
    return list(dict_product(ordered_dict))

def make_sentence_decoder(tokenizer, temperature=1):
    """
    Creates a decoder which uses a tokenizer's vocab to an encoded sentence (logits or indexes) back into text
    """
    def sentence_decoder(encoded_sentences):
        # If its an embedding (predictions)
        if len(encoded_sentences.shape) == 3:
            sentence = encoded_sentences[0]
            output_idxs = []

            for word in sentence:
                predicted_word_vector = F.softmax(word / temperature, 0)
                vector_sampler = torch.distributions.Categorical(predicted_word_vector)
                output_idxs.append(int(vector_sampler.sample()))

            return tokenizer.decode(output_idxs)

        # Else, its just the indices (targets)
        return tokenizer.decode(encoded_sentences[0])

    return sentence_decoder

def generate_run_name():
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    generate_run_name = os.path.join('results/runs', current_time + '_' + socket.gethostname())
    return generate_run_name
