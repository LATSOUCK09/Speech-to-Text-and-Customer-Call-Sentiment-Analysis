"""Chargement et tokenisation du jeu de données de sentiment.

Convertit les CSV (colonnes ``text``, ``label``) en jeux HuggingFace
tokenisés, prêts pour l'entraînement PyTorch.

Fichiers attendus dans ``data/`` :
    - train.csv, val.csv, test.csv
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from config import *


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    """Tokenise un batch de textes et convertit les labels texte en indices.

    Args:
        batch: Dictionnaire avec clés ``text`` (list[str]) et ``label`` (list[str]).

    Returns:
        Encodage tokenizer + clé ``labels`` (list[int]).
    """
    encoded = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
    encoded["labels"] = [LABEL2ID[label] for label in batch["label"]]
    return encoded


def load_data():
    """Charge et tokenise les jeux train/validation/test depuis des CSV.

    Returns:
        ``DatasetDict`` HuggingFace formaté en tenseurs PyTorch.
    """
    dataset = load_dataset(
        "csv",
        data_files={
            "train": "data/train.csv",
            "validation": "data/val.csv",
            "test": "data/test.csv"
        }
    )

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text", "label"]
    )
    tokenized_dataset.set_format("torch")
    return tokenized_dataset


if __name__ == "__main__":
    dataset = load_data()
    print(dataset)