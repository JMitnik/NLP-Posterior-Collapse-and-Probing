from dataclasses import dataclass

@dataclass
class Config:
    # Sizes
    batch_size: int
    embedding_size: int
    hidden_size: int
    vocab_size: int
    nr_epochs: int

    # Paths
    train_path: str
    valid_path: str
    test_path: str
    device: str
