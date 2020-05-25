from models.losses import L1DistanceLoss
from runners.metrics import calc_uuas

import torch
from torch import Tensor
from typing import List

def evaluate_struct_probe(probe, data_loader):
    loss_scores: List[Tensor] = []
    uuas_scores: List[Tensor] = []
    loss_score = 0
    uuas_score = 0

    probe.eval()
    loss_function = L1DistanceLoss()

    with torch.no_grad():
        for idx, batch_item in enumerate(data_loader):
            for item in batch_item:
                X, y = item

                if len(X.shape) == 2:
                    X = X.unsqueeze(0)

                # Sentences with strange tokens, we ignore for the moment
                if len(y) < 2:
                    print(f"Encountered: null sentence at idx {idx}")
                    continue

                pred_distances = probe(X)
                item_loss, _ = loss_function(pred_distances, y, torch.tensor(len(y)))

                if len(pred_distances.shape) > 2:
                    pred_distances = pred_distances.squeeze(0)

                uuas = calc_uuas(pred_distances, y)

                loss_score += item_loss.item()
                loss_scores.append(torch.tensor(item_loss.item()))
                uuas_scores.append(torch.tensor(uuas, dtype=torch.float))

    loss_score = torch.mean(torch.stack(loss_scores))
    uuas_score = torch.mean(torch.stack(uuas_scores))
    print(f"Average evaluation loss score is {loss_score}")
    print(f"Average evaluation uuas score is {uuas_score}")

    return loss_score, uuas_score

