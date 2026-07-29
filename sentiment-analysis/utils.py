import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_best_model(model, val_loss, best_val_loss, model_name):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_path = f"best_{model_name}.pth"
        torch.save(model.state_dict(), model_path)
    return best_val_loss



def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return acc, f1