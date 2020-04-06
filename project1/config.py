from dataclasses import dataclass

@dataclass
class Config:
    # Sizes
    embedding_size: int
    hidden_size: int
    vocab_size: int
    nr_epochs: int

    # Paths
    train_path: str
    valid_path: str
    test_path: str
