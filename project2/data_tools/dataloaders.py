from torch import Tensor
from typing import List

def custom_collate_fn(items: List[Tensor]) -> List[Tensor]:
    return items
