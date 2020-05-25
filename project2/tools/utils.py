import os
import torch

def ensure_path(path_to_file):
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)

def save_model(path_to_file, model):
    ensure_path(path_to_file)
    torch.save(model.state_dict(), path_to_file)

def load_model(path_to_file, model):
    model.load_state_dict(torch.load(path_to_file))
