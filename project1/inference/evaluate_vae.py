import torch
from metrics import calc_batch_perplexity

def evaluate_vae(
    model,
    data_loader,
    epoch: int,
    device: str,
    criterion,
    mu_force_beta_param,
    eval_type: str = 'valid',
    iteration: int = 0
):
    model.eval()
    total_loss: float = 0
    total_kl_loss: float = 0
    total_nll: float = 0
    total_perp: float = 0
    total_mu_loss: float = 0

    for batch, sent_lengths in data_loader:
        with torch.no_grad():
            inp = batch[:, 0:-1].to(device)

            # Creat both prediction of next word and the posterior of which we sample Z.
            # Nr to sample
            # nr_MC_sample = 10 if eval_type == 'test' else 1 # Did not work out unfortunately
            nr_MC_sample = 1 if eval_type == 'test' else 1
            preds, posterior = model(inp, nr_MC_sample)

            # If we have multi-log sample, average over the likelihoods on the 0th dimension
            is_using_multi_samples = nr_MC_sample > 1

            if is_using_multi_samples:
                preds = preds.reshape(nr_MC_sample, batch.shape[0], -1).mean(0)

            # Define target as the next word to predict
            target = batch[:, 1:].to(device)

            # Calc loss by using the ELBO-criterion
            loss, kl_loss, nll = criterion(
                preds,
                target,
                posterior
            )

            # Perplexity
            perp = calc_batch_perplexity(nll.detach(), sent_lengths)

            # Calc perplexity
            # Take mean of mini-batch loss
            loss = loss.mean()
            kl_loss = kl_loss.mean()
            nll = nll.mean()

            # Now add to the loss mu force loss
            batch_mean_vectors = posterior.loc
            avg_batch_mean_vector = batch_mean_vectors.mean(0)
            mu_force_loss_var = torch.tensordot(batch_mean_vectors - avg_batch_mean_vector, batch_mean_vectors - avg_batch_mean_vector, 2) / batch.shape[0] / 2
            mu_force_loss = torch.max(torch.tensor([0.0]), mu_force_beta_param - mu_force_loss_var).to(device)

            loss = loss + mu_force_loss


            total_loss += loss.item()
            total_kl_loss += kl_loss.item()
            total_nll += nll.item()
            total_perp += perp
            total_mu_loss += mu_force_loss_var.item()

    total_loss = total_loss / len(data_loader)
    total_kl_loss = total_kl_loss / len(data_loader)
    total_nll = total_nll / len(data_loader)
    total_perp = total_perp / len(data_loader)
    total_mu_loss = total_mu_loss / len(data_loader)

    return (total_loss, total_kl_loss, total_nll, total_mu_loss), total_perp
