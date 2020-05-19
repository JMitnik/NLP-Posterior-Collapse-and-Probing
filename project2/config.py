from dataclasses import dataclass, asdict

@dataclass
class Config:
    # Paths
    run_label: str = ''
    path_to_pretrained_lstm: str = 'storage/pretrained_lstm_state_dict.pt'
    path_to_data_train: str = 'data/sample/en_ewt-ud-train.conllu'
    path_to_data_valid: str = 'data/sample/en_ewt-ud-dev.conllu'
    path_to_data_test: str = 'data/sample/en_ewt-ud-test.conllu'

    # Saved Model Paths
    path_to_POS_Probe: str = lambda version: 'saved_models/POS_Probe_' + str(version)

    # Feature Model types
    feature_model_type: str = 'LSTM'
    default_trans_model_type: str = 'distilgpt2'

    # Booleans
    will_train_simple_probe: bool = True
    will_control_task_simple_prob: bool = True
    will_train_structural_probe: bool = True
    will_train_dependency_probe: bool = True
    will_control_task_dependency_probe: bool = True

    # POS Probe Model Params
    pos_probe_linear = True
    pos_probe_train_batch_size: int = 32
    pos_probe_train_epoch: int = 1000
    pos_probe_train_patience: int = 4
    pos_probe_train_lr: float = 0.0001

    # Field names for the POS probe results

    # Structural probe parameters
    struct_probe_emb_dim: int = 768
    struct_probe_rank: int = 64
    struct_probe_lr: float = 10e-4
    struct_probe_train_batch_size: int = 24
    struct_probe_train_epoch: int = 5
    struct_probe_train_factor: float = 0.5
    struct_probe_train_patience: int = 1

    def to_dict(self):
        config_dict = asdict(self)

