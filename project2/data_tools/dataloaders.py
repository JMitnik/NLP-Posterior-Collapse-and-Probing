from data_tools.transforms import transform_XY_to_concat_tensors
from data_tools.datasets import ProbingDataset
from data_tools.data_inits import init_tree_corpus
from torch import Tensor
from torch.utils.data import DataLoader
from typing import List, Tuple

def custom_collate_fn(items: List[Tensor]) -> List[Tensor]:
    return items

def make_struct_dataloader(
    path_to_eval,
    feature_model,
    feature_model_tokenizer,
    use_dependencies: bool = False,
    use_corrupted: bool = False,
    use_shuffled_dataset: bool = False,
    corrupted_vocab = None,
    verbose: bool = True
):
    eval_X, eval_y = init_tree_corpus(
        path_to_eval,
        feature_model,
        feature_tokenizer=feature_model_tokenizer,
        use_dependencies=use_dependencies,
        use_corrupted=use_corrupted,
        dep_vocab=corrupted_vocab
    )
    eval_dataset = ProbingDataset(eval_X, eval_y)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    return eval_dataloader

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
    corrupted_vocab = None,
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader]:
    # Make training data-loader
    train_X, train_y = init_tree_corpus(
        path_to_train,
        feature_model,
        feature_model_tokenizer,
        use_dependencies=use_dependencies,
        use_corrupted=use_corrupted,
        dep_vocab=corrupted_vocab
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
        dep_vocab=corrupted_vocab
    )
    valid_dataset = ProbingDataset(valid_X, valid_y)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=use_shuffled_dataset, collate_fn=custom_collate_fn)

    if verbose:
        print(f"Loaded in train dataloader with {len(train_dataloader)} items and valid dataloader with {len(valid_dataloader)}")

    return train_dataloader, valid_dataloader
