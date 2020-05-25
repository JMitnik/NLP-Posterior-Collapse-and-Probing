import torch
import numpy as np
from typing import Tuple, List
import torch.nn as nn
from torch.nn import NLLLoss
import torch.optim as optim
from config import Config
from torch.utils.data import DataLoader
from models.probes import TwoWordBilinearLabelProbe
from runners.evaluators import evaluate_dep_parsing

def train_dep_parsing(
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    feature_dim: int,
    probe_rank: int,
    lr: float,
    nr_epochs: int = 30,
    patience: int = 4
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Trains dependency parsing probe
    """
    # Probing model
    probe: nn.Module = TwoWordBilinearLabelProbe(
        max_rank=probe_rank,
        feature_dim=feature_dim,
        dropout_p=0.2
    )

    # Training tools
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    loss_function = NLLLoss(reduction='none')

    # Scores
    valid_losses = []
    acc_scores = []
    lowest_loss = np.inf

    # Counter
    it = 0
    patience_counter = 0

    for epoch in range(nr_epochs):
        print(f"\n---- EPOCH {epoch + 1} ---- \n")

        for idx, train_batch in enumerate(train_dataloader):
            it += 1
            probe.train()
            optimizer.zero_grad()

            batch_loss = torch.tensor([0.0])

            for train_item in train_batch:
                train_X, train_y = train_item
                train_X = train_X.unsqueeze(0)

                pred = probe(train_X).squeeze()

                if len(train_y) < 2:
                    continue

                # In case we will deal with the control task, decrease the parent by 1
                sen_len = len(train_y)
                if sen_len in train_y:
                    train_y[train_y == sen_len] = sen_len - 1

                masked_idx = train_y != -1
                masked_pred = pred[masked_idx]
                masked_y = train_y[masked_idx].long()

                item_loss = loss_function(masked_pred, masked_y)
                item_loss = item_loss.mean()
                batch_loss += item_loss

            try:
                batch_loss.backward()
                optimizer.step()
            except:
                continue

        valid_loss, acc_score = evaluate_dep_parsing(probe, valid_dataloader)
        acc_scores.append(acc_score.item())
        valid_losses.append(valid_loss.item())

        if valid_loss.item() < lowest_loss:
            print(f"New loss has been reached of {valid_loss.item()} being lower than lowest_loss {lowest_loss}")
            patience_counter = 0
            lowest_loss = float(valid_loss.item())
        elif patience_counter >= patience:
            print("Started to overfit, patience has been reached")
            break
        else:
            print(f"...are we overfitting? Setting patience_counter to {patience_counter + 1}")
            patience_counter += 1


    return probe, valid_losses, acc_scores
