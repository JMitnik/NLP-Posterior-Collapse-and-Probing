from torch.utils.data import DataLoader, Dataset

class ProbingDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.features[idx], self.targets[idx])
