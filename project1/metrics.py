import torch
import numpy as np
import torch.nn as nn

def make_elbo_criterion(vocab_size: int, latent_size, freebits_param=-1, mu_force_beta_param=1):
    likelihood_criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    prior_dist = torch.distributions.Normal(
        torch.zeros(latent_size),
        torch.ones(latent_size)
    )

    def elbo_criterion(
        prediction: torch.Tensor,
        original: torch.Tensor,
        posterior_dist: torch.distributions.Distribution
    ):
        kl_divergence = torch.distributions.kl_divergence(prior_dist, posterior_dist)

        # Free bit implementation
        if freebits_param >= 0:
            # Mean across mini-batch needs to be larger than freebits_param
            hinge_mask = kl_divergence.mean(0) > freebits_param
            kl_divergence.T[hinge_mask] = freebits_param

        kl_loss = kl_divergence.sum(1).to(prediction.device)

        negative_log_likelihood = likelihood_criterion(
            prediction.view([-1, vocab_size]),
            original.view(-1)
        ).to(prediction.device)

        # Mean of all words for each batch item
        negative_log_likelihood = negative_log_likelihood.view(prediction.shape[0], -1).sum(1)

        return negative_log_likelihood + kl_loss, kl_loss, negative_log_likelihood

    return elbo_criterion

def calc_batch_perplexity(loss: torch.Tensor, sentence_lengths: torch.Tensor):
    nr_words_batch: int = int(torch.sum(sentence_lengths).item())
    return np.exp(loss.item() / nr_words_batch)

def calc_mu_loss(posterior, batch_size):
    batch_mean_vectors = posterior.loc # mu(n) mu vector of nth sample
    avg_batch_mean_vector = batch_mean_vectors.mean(0) # mu(stripe) mean of vectors mu
    mu_force_loss_var = torch.tensordot(batch_mean_vectors - avg_batch_mean_vector, batch_mean_vectors - avg_batch_mean_vector, 2) / batch_size / 2
    mu_force_loss = torch.max(torch.tensor([0.0]), mu_force_beta_param - mu_force_loss_var)

    return mu_force_loss
