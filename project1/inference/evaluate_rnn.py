from tools.results_writer import ResultsWriter
import torch
from metrics import calc_batch_perplexity

def evaluate_rnn(
    model,
    data_loader,
    epoch,
    device,
    criterion,
    eval_type: str = 'valid',
    it: int = 0
):
    model.eval()

    total_loss: float = 0
    total_perp: float = 0

    for batch, sent_length in data_loader:
        with torch.no_grad():
            input = batch[:, 0: -1].to(device)
            target = batch[:, 1:].to(device)

            output = model(input)

            loss = criterion(output.reshape(-1, model.vocab_size), target.reshape(-1))
            perp = calc_batch_perplexity(loss, sent_length)

            loss = loss / batch.shape[0]
            total_loss += loss
            total_perp += perp.item()

    total_perp = total_perp / len(data_loader)
    total_loss = total_loss / len(data_loader)
    return total_loss, total_perp
