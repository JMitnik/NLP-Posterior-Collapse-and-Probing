import torch
import torch.nn as nn

def make_elbo_criterion(vocab_size: int, freebits_param=-1, mu_force_beta_param=1):
    likelihood_criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    def elbo_criterion(
        prediction: torch.Tensor,
        original: torch.Tensor,
        prior_dist: torch.distributions.Distribution,
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
