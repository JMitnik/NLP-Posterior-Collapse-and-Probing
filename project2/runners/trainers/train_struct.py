import torch
from typing import Tuple, List
import torch.nn as nn
from torch.nn import NLLLoss
import torch.optim as optim
from config import Config
from torch.utils.data import DataLoader
from models.probes import StructuralProbe
from models.losses import L1DistanceLoss
from runners.evaluators import evaluate_struct_probe

def train_struct(
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    config: Config
):
    emb_dim: int = config.feature_model_dimensionality
    rank: int = config.struct_probe_rank
    lr: float = config.struct_probe_lr
    epochs: int = config.struct_probe_train_epoch

    valid_losses = []
    valid_uuas_scores = []

    probe: nn.Module = StructuralProbe(emb_dim, rank)

    # Training tools
    optimizer = optim.Adam(probe.parameters(), lr=lr)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,patience=1)
    loss_function = L1DistanceLoss()

    for epoch in range(epochs):
        print(f"\n---- EPOCH {epoch + 1} ---- \n")
        for train_batch in train_dataloader:
            # Setup
            probe.train()
            optimizer.zero_grad()

            batch_loss = torch.tensor([0.0])

            for train_item in train_batch:
                train_X, train_y = train_item
                train_X, train_y = train_item

                if len(train_X.shape) == 2:
                    train_X = train_X.unsqueeze(0)

                pred_distances = probe(train_X)
                item_loss, _ = loss_function(pred_distances, train_y, torch.tensor(len(train_y)))
                batch_loss += item_loss

            batch_loss.backward()
            optimizer.step()

        # Calculate validation scores
        # TODO: Double-check that the UUAS works
        valid_loss, valid_uuas = evaluate_struct_probe(probe, valid_dataloader)
        valid_losses.append(valid_loss.item())
        valid_uuas_scores.append(valid_uuas.item())

        # TODO: Optional Param-tune scheduler (?)
#         scheduler.step(valid_loss)

    return probe, valid_losses, valid_uuas_scores
