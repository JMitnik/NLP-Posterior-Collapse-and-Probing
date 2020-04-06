from torch.utils.tensorboard import SummaryWriter
import torch

def store_training_results(
    writer: SummaryWriter,
    loss: torch.Tensor,
    perplexity: torch.Tensor, epoch_nr: int
):
    """
    Stores results by doing the following
        - Writes results to tensorboard (store in /runs)

    Arguments:
        loss {torch.Tensor} -- Loss for a training batch
        perplexity {torch.Tensor} -- Perplexity for a training batch
    """
    writer.add_scalar('Train/loss', loss, epoch_nr)
    writer.add_scalar('Train/perplexity', perplexity, epoch_nr)
