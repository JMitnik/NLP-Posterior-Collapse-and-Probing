from torch import Tensor
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple

def transform_XY_to_concat_tensors(X: List[Tensor], y: List[Tensor]) -> Tuple[Tensor, Tensor]:
    # X_concat = torch.cat([x.reshape(1) for x in X], dim=0)
    X_concat = torch.cat(X, dim=0)
    y_concat = torch.cat(y, dim=0)

    return X_concat, y_concat

def transform_XY_to_padded_tensors(X: List[Tensor], y: List[Tensor]) -> Tuple[Tensor, Tensor]:
    X_padded = pad_sequence(X)
    y_padded = pad_sequence(y)

    return X_padded, y_padded
