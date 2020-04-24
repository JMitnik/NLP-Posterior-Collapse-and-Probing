from config import Config
import utils
from models.RNNLM import RNNLM

def make_rnnlm(config, trained=False, model_path=None):
    rnn_lm = RNNLM(
        config.vocab_size,
        config.embedding_size,
        config.rnn_hidden_size
    ).to(config.device)

    if trained and model_path is not None:
        rnn_lm, _, _ = utils.load_model(model_path, rnn_lm, config.device)

    return rnn_lm
