from torch.utils.tensorboard import SummaryWriter
import torch

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
    torch.save(checkpoint, f'models/saved_models/{path}')

def load_model(path, model, optimizer, device):
    """
    Load a model and optimizer state to device from path.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['step']