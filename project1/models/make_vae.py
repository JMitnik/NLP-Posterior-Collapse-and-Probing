from config import Config
import utils
from models.VAE import VAE

def make_vae(config, trained=False, model_path=None):
    vae = VAE(
        encoder_hidden_size=config.vae_encoder_hidden_size,
        decoder_hidden_size=config.vae_decoder_hidden_size,
        latent_size=config.vae_latent_size,
        vocab_size=config.vocab_size,
        param_wdropout_k=config.param_wdropout_k,
        embedding_size=config.embedding_size
    ).to(config.device)

    if trained and model_path is not None:
        vae, _, _ = utils.load_model(model_path, vae, config.device)

    return vae
