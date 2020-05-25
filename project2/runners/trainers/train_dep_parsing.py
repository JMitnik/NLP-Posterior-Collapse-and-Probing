import torch
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
    config: Config
)-> Tuple[nn.Module, List[float], List[float]]:
    emb_dim: int = config.feature_model_dimensionality
    rank: int = config.struct_probe_rank
    lr: float = config.struct_probe_lr
    epochs: int = config.dep_probe_train_epoch

    probe: nn.Module = TwoWordBilinearLabelProbe(
        max_rank=rank,
        feature_dim=emb_dim,
        dropout_p=0.2
    )

    # Training tools
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    loss_function = NLLLoss(reduction='none')

    valid_losses = []
    acc_scores = []
    it = 0

    for epoch in range(epochs):
        print(f"\n---- EPOCH {epoch + 1} ---- \n")

        for idx, train_batch in enumerate(train_dataloader):
            it += 1
            # Setup
            probe.train()
            optimizer.zero_grad()

            batch_loss = torch.tensor([0.0])

            for train_item in train_batch:
                train_X, train_y_tup = train_item

                train_X = train_X.unsqueeze(0)
                train_y, _ = train_y_tup

                pred = probe(train_X).squeeze()

                if len(train_y) < 2:
                    print(f"Encountered: null sentence")
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

    return probe, valid_losses, acc_scores
