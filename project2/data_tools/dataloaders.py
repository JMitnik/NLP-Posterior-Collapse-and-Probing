from data_tools.transforms import transform_XY_to_concat_tensors
from data_tools.datasets import ProbingDataset
from data_tools.data_inits import init_tree_corpus
from torch import Tensor
from torch.utils.data import DataLoader
from typing import List, Tuple

def custom_collate_fn(items: List[Tensor]) -> List[Tensor]:
    return items

def make_struct_dataloaders(
    path_to_train,
    path_to_valid,
    feature_model,
    feature_model_tokenizer,
    train_batch_size = 8,
    valid_batch_size = 1,
    use_dependencies: bool = False,
    use_corrupted: bool = False,
    use_shuffled_dataset: bool = False,
    corruped_vocab = None
) -> Tuple[DataLoader, DataLoader]:
    # Make training data-loader
    train_X, train_y = init_tree_corpus(
        path_to_train,
        feature_model,
        feature_model_tokenizer,
        use_dependencies=use_dependencies,
        use_corrupted=use_corrupted,
        dep_vocab=corruped_vocab
    )
    train_dataset = ProbingDataset(train_X, train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=use_shuffled_dataset, collate_fn=custom_collate_fn)

    # Make validation data-loader
    valid_X, valid_y = init_tree_corpus(
        path_to_valid,
        feature_model,
        feature_model_tokenizer,
        use_dependencies=use_dependencies,
        use_corrupted=use_corrupted,
        dep_vocab=corruped_vocab
    )
    valid_dataset = ProbingDataset(valid_X, valid_y)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=use_shuffled_dataset, collate_fn=custom_collate_fn)

    return train_dataloader, valid_dataloader
