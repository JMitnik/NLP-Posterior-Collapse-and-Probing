from typing import Tuple, List
import torch
import torch.nn as nn
from torch.nn import NLLLoss
from sklearn.metrics import accuracy_score
import torch.optim as optim
from torch import Tensor
from config import Config
from torch.utils.data import DataLoader

def evaluate_dep_parsing(probe, data_loader):
    loss_scores: List[Tensor] = []
    acc_scores: List[Tensor] = []
    loss_score = 0

    probe.eval()
    loss_function = NLLLoss()

    with torch.no_grad():
        for idx, batch_item in enumerate(data_loader):
            for item in batch_item:
                valid_X, valid_y = item
                valid_X = valid_X.unsqueeze(0)

                pred = probe(valid_X).squeeze()
                           # Sentences with strange tokens, we ignore for the moment
                if len(valid_y) < 2:

                # In case we will deal with the control task, decrease the parent by 1
                sen_len = len(valid_y)
                if sen_len in valid_y:
                    valid_y[valid_y == sen_len] = sen_len - 1

                masked_idx = valid_y != -1
                masked_pred = pred[masked_idx]
                masked_y = valid_y[masked_idx].long()

                item_loss = loss_function(masked_pred, masked_y)
                acc = accuracy_score(masked_y, masked_pred.argmax(1))
                loss_scores.append(torch.tensor(item_loss.item()))
                acc_scores.append(torch.tensor(acc))

    loss_score = torch.mean(torch.stack(loss_scores))
    acc_score = torch.mean(torch.stack(acc_scores))

    print(f"Average evaluation loss score is {loss_score}")
    print(f"Average evaluation accuracy score is {acc_score}")

    return loss_score, acc_score
