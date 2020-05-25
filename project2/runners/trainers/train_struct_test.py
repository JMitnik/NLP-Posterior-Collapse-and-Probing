from runners.trainers.train_struct import train_struct
from models.model_inits import make_pretrained_lstm_and_tokenizer
from data_tools.dataloaders import make_struct_dataloaders


def test_train_struct_for_LSTM():
    path_to_train = 'data/sample/en_ewt-ud-train.conllu'
    path_to_valid = 'data/sample/en_ewt-ud-dev.conllu'
    nr_epochs = 1

    lstm, lstm_tokenizer = make_pretrained_lstm_and_tokenizer()

    train_dataloader, valid_dataloader = make_struct_dataloaders(
        path_to_train,
        path_to_valid,
        feature_model=lstm,
        feature_model_tokenizer=lstm_tokenizer,
    )

    train_struct(
        train_dataloader,
        valid_dataloader,
        nr_epochs=nr_epochs,
        struct_emb_dim=lstm.nhid,
        struct_lr=10e-4,
        struct_rank=64,
    )
