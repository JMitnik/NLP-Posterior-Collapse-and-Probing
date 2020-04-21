import torch
import torch.distributions as D
import torch.nn.functional as F
from models import VAE
import numpy as np

def generate_next_words(
    model,
    custom_data,
    start_sent="as they parked out front and owen stepped out of the car , he could see",
    device='cpu',
    max_length=10,
    temperature=1
):
    print(f'Start of the sentence: {start_sent} || Max Length {max_length} .')
    with torch.no_grad():
        encoded_start = custom_data.tokenizer.encode(start_sent, add_special_tokens=True)[:-1]
        sentence = encoded_start
        print(sentence)

        for i in range(max_length):
            # Create input for the model
            model_inp = torch.tensor(sentence).to(device)
            model_inp = model_inp.unsqueeze(0) # Ensures we pass a 1(=batch-dimension) x sen-length vector
            output = model(model_inp).cpu().detach()
            # print(output)
            prediction_vector = F.softmax(output[0][-1] / temperature)
            sample_vector = D.Categorical(prediction_vector)
            sample = int(sample_vector.sample())
            if sample == 3: # cannot produces UNK token
                i = i-1
                continue
            sentence.append(sample)

            if sample == 2: # If we sampled EOS
                break

        print(sentence)
        print(f'Sentence Length: {len(sentence)}')

        return sentence


def evaluate_rnn(
    model,
    data_loader,
    epoch,
    device,
    criterion,
    writer,
    eval_type: str = 'valid',
    it: int = 0
):
    model.eval()
    total_loss: float = 0
    total_perp: float = 0

    for batch in data_loader:
        with torch.no_grad():
            input = batch[:, 0: -1].to(device)
            target = batch[:, 1:].to(device)

            output = model(input)

            loss = criterion(output.reshape(-1, model.vocab_size), target.reshape(-1))
            sentence_length = batch[0].size()[0]
            perp = np.exp((loss.item() / batch.shape[0]) / sentence_length)
            total_loss += loss / len(batch)
            total_perp += perp

    total_loss = total_loss / len(data_loader)
    total_perp = total_perp / len(data_loader)
    
    writer.add_scalar(f'{eval_type}-rnn/loss' , total_loss, it)
    writer.add_scalar(f'{eval_type}-rnn/ppl' , total_perp, it)

    return total_loss, total_perp


# Better to make it specific for validation?
def evaluate_VAE(
    model,
    data_loader,
    epoch: int,
    device: str,
    criterion,
    mu_force_beta_param,
    prior,
    results_writer,
    eval_type: str = 'valid',
    iteration: int = 0
):
    model.eval()
    total_loss: float = 0
    total_kl_loss: float = 0
    total_nlll: float = 0
    total_perp: float = 0
    total_mu_loss: float = 0

    for batch in data_loader:
        with torch.no_grad():
            inp = batch[:, 0:-1].to(device)

            # Creat both prediction of next word and the posterior of which we sample Z.
            # Nr to sample
            nr_MC_sample = 10 if eval_type == 'test' else 1
            preds, posterior = model(inp, nr_MC_sample)

            # # Define target as the next word to predict
            target = batch[:, 1:].to(device)

            # # Calc loss by using the ELBO-criterion
            loss, kl_loss, nlll = criterion(
                preds,
                target,
                prior,
                posterior
            )

            # Take mean of mini-batch loss
            loss = loss.mean()
            kl_loss = kl_loss.mean()
            nlll = nlll.mean()
            
            sentence_length = batch[0].size()[0]
            perp = np.exp(loss.cpu().item() / sentence_length)



            # Now add to the loss mu force loss
            batch_mean_vectors = posterior.loc
            avg_batch_mean_vector = batch_mean_vectors.mean(0)
            mu_force_loss_var = torch.tensordot(batch_mean_vectors - avg_batch_mean_vector, batch_mean_vectors - avg_batch_mean_vector, 2) / batch.shape[0] / 2
            mu_force_loss = torch.max(torch.tensor([0.0]), mu_force_beta_param - mu_force_loss_var).to(device)

            loss = loss + mu_force_loss

            total_loss += loss.item()
            total_kl_loss += kl_loss.item()
            total_nlll += nlll.item()
            total_perp += perp
            total_mu_loss += mu_force_loss_var.item()

    total_loss = total_loss / len(data_loader)
    total_kl_loss = total_kl_loss / len(data_loader)
    total_nlll = total_nlll / len(data_loader)
    total_perp = total_perp / len(data_loader)
    total_mu_loss = total_mu_loss / len(data_loader)

    results_writer.add_scalar(f'{eval_type}-vae/elbo-loss', total_loss, iteration)
    results_writer.add_scalar(f'{eval_type}-vae/ppl', torch.log(torch.tensor(total_loss)), iteration)
    results_writer.add_scalar(f'{eval_type}-vae/kl-loss', total_kl_loss, iteration)
    results_writer.add_scalar(f'{eval_type}-vae/nll-loss', total_nlll, iteration)
    results_writer.add_scalar(f'{eval_type}-vae/mu-loss', total_mu_loss, iteration)

    return total_loss, total_kl_loss, total_nlll, total_perp, total_mu_loss
