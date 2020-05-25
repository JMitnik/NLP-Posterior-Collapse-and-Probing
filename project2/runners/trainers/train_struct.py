import torch
from typing import Tuple, List
import torch.nn as nn
from torch.nn import NLLLoss
import torch.optim as optim
from torch.utils.data import DataLoader
from models.probes import StructuralProbe
from models.losses import L1DistanceLoss
from runners.evaluators import evaluate_struct_probe

def train_struct(
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    nr_epochs: int,
    struct_emb_dim: int,
    struct_lr: float,
    struct_rank: int,
):
    valid_losses = []
    valid_uuas_scores = []

    probe: nn.Module = StructuralProbe(struct_emb_dim, struct_rank)

    # Training tools
    optimizer = optim.Adam(probe.parameters(), lr=struct_lr)
    loss_function = L1DistanceLoss()

    for epoch in range(nr_epochs):
        print(f"\n---- EPOCH {epoch + 1} ---- \n")
        for train_batch in train_dataloader:
            probe.train()
            optimizer.zero_grad()

            batch_loss = torch.tensor([0.0])

            for train_item in train_batch:
                train_X, train_y = train_item

                if len(train_X.shape) == 2:
                    train_X = train_X.unsqueeze(0)

                pred_distances = probe(train_X)
                item_loss, _ = loss_function(pred_distances, train_y, torch.tensor(len(train_y)))
                batch_loss += item_loss

            batch_loss.backward()
            optimizer.step()

        # Calculate validation scores
        valid_loss, valid_uuas = evaluate_struct_probe(probe, valid_dataloader)
        valid_losses.append(valid_loss.item())
        valid_uuas_scores.append(valid_uuas.item())

    return probe, valid_losses, valid_uuas_scores
