from runners.trainers.train_struct import train_struct
import pytest
from models.model_inits import make_pretrained_lstm_and_tokenizer, make_pretrained_transformer_and_tokenizer
from data_tools.dataloaders import make_struct_dataloaders

def test_train_struct_for_LSTM():
    path_to_train = 'data/sample/en_ewt-ud-train.conllu'
    path_to_valid = 'data/sample/en_ewt-ud-dev.conllu'
    nr_epochs = 3

    lstm, lstm_tokenizer = make_pretrained_lstm_and_tokenizer()

    train_dataloader, valid_dataloader = make_struct_dataloaders(
        path_to_train,
        path_to_valid,
        feature_model=lstm,
        feature_model_tokenizer=lstm_tokenizer,
    )

    probe, losses, acc = train_struct(
        train_dataloader,
        valid_dataloader,
        nr_epochs=nr_epochs,
        struct_emb_dim=lstm.nhid,
        struct_lr=10e-4,
        struct_rank=64,
    )

    # Let's test that the first loss is higher than the last loss
    assert losses[0] > losses[-1], "Loss does not seem to improve"

    # Let's test that the accuracy is lower
    assert acc[0] < acc[-1], "Accuracy does not seem to improve"

@pytest.mark.parametrize('model', ['distilgpt2', 'xlm-roberta-base'])
def test_train_struct_for_transformer(model):
    path_to_train = 'data/sample/en_ewt-ud-train.conllu'
    path_to_valid = 'data/sample/en_ewt-ud-dev.conllu'
    nr_epochs = 3

    print(f"Loading in {model}")
    transformer, transformer_tokenizer = make_pretrained_transformer_and_tokenizer(model)

    train_dataloader, valid_dataloader = make_struct_dataloaders(
        path_to_train,
        path_to_valid,
        feature_model=transformer,
        feature_model_tokenizer=transformer_tokenizer,
    )

    probe, losses, acc = train_struct(
        train_dataloader,
        valid_dataloader,
        nr_epochs=nr_epochs,
        struct_emb_dim=768,
        struct_lr=10e-4,
        struct_rank=64,
    )

    # Let's test that the first loss is higher than the last loss
    assert losses[0] > losses[-1], "Loss does not seem to improve"

    # Let's test that the accuracy is lower
    assert acc[0] < acc[-1], "Accuracy does not seem to improve"
