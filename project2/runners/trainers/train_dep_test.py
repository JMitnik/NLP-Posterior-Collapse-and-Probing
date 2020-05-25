from runners.trainers.train_dep_parsing import train_dep_parsing
from data_tools.data_inits import parse_all_corpora
from runners.trainers.train_struct import train_struct
import pytest
from models.model_inits import make_pretrained_lstm_and_tokenizer, make_pretrained_transformer_and_tokenizer
from data_tools.dataloaders import make_struct_dataloaders
from data_tools.target_extractors import create_corrupted_dep_vocab

def test_dep_dataloader_returns_parent():
    path_to_train = 'data/sample/en_ewt-ud-train.conllu'
    path_to_valid = 'data/sample/en_ewt-ud-dev.conllu'
    nr_epochs = 3

    transformer, transformer_tokenizer = make_pretrained_transformer_and_tokenizer('distilgpt2')

    train_dataloader, _ = make_struct_dataloaders(
        path_to_train,
        path_to_valid,
        feature_model=transformer,
        feature_model_tokenizer=transformer_tokenizer,
        use_dependencies=True
    )

    train_sample = next(iter(train_dataloader))

    for train_item in train_sample:
        _, parent_edges = train_item

        root_nodes = [i for i in parent_edges if i == -1]
        assert len(root_nodes) == 1, f"Encountered root_nodes of size {len(root_nodes)}"

def test_dep_dataloader_returns_corrupted_idxs():
    path_to_train = 'data/sample/en_ewt-ud-train.conllu'
    path_to_valid = 'data/sample/en_ewt-ud-dev.conllu'
    nr_epochs = 3

    # Read all corpora and extract into
    all_corpora = parse_all_corpora(True)
    corrupted_dep_vocab = create_corrupted_dep_vocab(all_corpora)

    transformer, transformer_tokenizer = make_pretrained_transformer_and_tokenizer('distilgpt2')

    train_dataloader, _ = make_struct_dataloaders(
        path_to_train,
        path_to_valid,
        feature_model=transformer,
        feature_model_tokenizer=transformer_tokenizer,
        use_dependencies=True,
        use_corrupted=True,
        corrupted_vocab=corrupted_dep_vocab
    )

    # Sample of training
    train_sample = next(iter(train_dataloader))
    for train_item in train_sample:
        _, parent_edges = train_item

        for idx, parent_tensor in enumerate(parent_edges):
            parent = parent_tensor.item()
            assert parent == idx or parent==-1 or parent == 0 or parent == len(parent_edges), "Not corrupted labels"

def test_dep_regular_training():
    path_to_train = 'data/sample/en_ewt-ud-train.conllu'
    path_to_valid = 'data/sample/en_ewt-ud-dev.conllu'
    nr_epochs = 3

    transformer, transformer_tokenizer = make_pretrained_transformer_and_tokenizer('distilgpt2')

    train_dataloader, valid_dataloader = make_struct_dataloaders(
        path_to_train,
        path_to_valid,
        feature_model=transformer,
        feature_model_tokenizer=transformer_tokenizer,
        use_dependencies=True
    )

    probe, losses, acc, = train_dep_parsing(
        train_dataloader,
        valid_dataloader,
        768,
        64,
        10e-4,
    )

    assert losses[0] > losses[-1], "Loss did not decrease over the training"
    assert acc[0] < acc[-1], "Accuracy did not increase over the training"

def test_dep_control_task_training():
    path_to_train = 'data/sample/en_ewt-ud-train.conllu'
    path_to_valid = 'data/sample/en_ewt-ud-dev.conllu'
    nr_epochs = 3

    transformer, transformer_tokenizer = make_pretrained_transformer_and_tokenizer('distilgpt2')

    # Read all corpora and extract into
    all_corpora = parse_all_corpora(True)
    corrupted_dep_vocab = create_corrupted_dep_vocab(all_corpora)

    train_dataloader, valid_dataloader = make_struct_dataloaders(
        path_to_train,
        path_to_valid,
        feature_model=transformer,
        feature_model_tokenizer=transformer_tokenizer,
        use_dependencies=True,
        use_corrupted=True,
        corrupted_vocab=corrupted_dep_vocab
    )

    probe, losses, acc, = train_dep_parsing(
        train_dataloader,
        valid_dataloader,
        768,
        64,
        10e-4,
    )

    assert losses[0] > losses[-1], "Loss did not decrease over the training"
    assert acc[0] < acc[-1], "Accuracy did not increase over the training"
