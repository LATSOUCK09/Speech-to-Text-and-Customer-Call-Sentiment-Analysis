"""Utilitaires pour l'entraînement de sentiment.

Ce module fournit des outils de reproducibilité, de sauvegarde du meilleur
modèle et de calcul de métriques de performance.
"""

import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed=42):
    """Fixe les graines aléatoires pour la reproductibilité (Python, NumPy, PyTorch, CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_best_model(model, val_loss, best_val_loss, model_name):
    """Sauvegarde le modèle si la loss de validation s'améliore.

    Args:
        model: Modèle PyTorch à sauvegarder.
        val_loss: Loss de validation de l'époque courante.
        best_val_loss: Meilleure loss de validation précédente.
        model_name: Préfixe du fichier (ex. ``bert_sentiment`` → ``best_bert_sentiment.pth``).

    Returns:
        La nouvelle meilleure loss de validation.
    """
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_path = f"best_{model_name}.pth"
        torch.save(model.state_dict(), model_path)
    return best_val_loss



def compute_metrics(preds, labels):
    """Calcule l'accuracy et le F1 score pondéré.

    Args:
        preds: Liste ou array des prédictions (indices de classe).
        labels: Liste ou array des labels réels.

    Returns:
        Tuple ``(accuracy, f1_weighted)``.
    """
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return acc, f1