from dataclasses import dataclass

@dataclass
class Config:
    # Paths
    path_to_pretrained_lstm: str = 'storage/pretrained_lstm_state_dict.pt'
    path_to_data_train: str = 'data/sample/en_ewt-ud-train.conllu'
    path_to_data_valid: str = 'data/sample/en_ewt-ud-dev.conllu'
    path_to_data_test: str = 'data/sample/en_ewt-ud-test.conllu'

    # Saved Model Paths
    path_to_POS_Probe: str = lambda version: 'saved_models/POS_Probe_' + str(version)

    # Booleans
    will_train_simple_probe: bool = True
    will_train_structural_probe: bool = True
    will_train_dependency_probe: bool = True

    # Structural probe parameters
    struct_probe_emb_dim: int = 768
    struct_probe_rank: int = 64
    struct_probe_lr: float = 10e-4
    struct_probe_train_batch_size: int = 24
    struct_probe_train_epoch: int = 5
    struct_probe_train_factor: float = 0.5
    struct_probe_train_patience: int = 1
