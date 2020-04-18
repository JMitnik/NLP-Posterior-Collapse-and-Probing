import torch

def make_elbo_criterion(vocab_size: int, freebits_param=-1):
    likelihood_criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    print("Made a new loss-func")

    def elbo_criterion(
        prediction: torch.Tensor,
        original: torch.Tensor,
        prior_dist: torch.distributions.Distribution,
        posterior_dist: torch.distributions.Distribution
    ):
        batch_size = prediction.shape[0]

        kl_loss = torch.distributions.kl_divergence(prior_dist, posterior_dist).sum(1).to(prediction.device)
        # Free bit implementation
        if freebits_param >= 0:
            freebits_tensor = torch.full_like(kl_loss, fill_value=freebits_param, dtype=torch.float)
            freebit_loss = torch.max(kl_loss, freebits_tensor)
            kl_loss = freebit_loss

        negative_log_likelihood = likelihood_criterion(
            prediction.view([-1, vocab_size]),
            original.view(-1)
        ).to(prediction.device)

        # Mean of all words for each batch item
        negative_log_likelihood = negative_log_likelihood.view(prediction.shape[0], -1).sum(1)

        return negative_log_likelihood + kl_loss, kl_loss, negative_log_likelihood

    return elbo_criterion
