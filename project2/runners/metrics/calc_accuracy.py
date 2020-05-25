accuracy = lambda preds, targets: sum([1 if pred == target else 0 for pred, target in zip(preds, targets)]) / len(preds)
