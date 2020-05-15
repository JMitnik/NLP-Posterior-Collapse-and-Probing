from dataclasses import dataclass

@dataclass
class Config:
    # Paths
    path_to_pretrained_lstm = 'storage/pretrained_lstm_state_dict.pt'

    # Booleans
    will_train_simple_probe: bool = True
    will_train_structural_probe: bool = True
