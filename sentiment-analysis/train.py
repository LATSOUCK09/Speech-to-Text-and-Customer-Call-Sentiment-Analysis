"""Entraîne et évalue le modèle de classification de sentiment.

Ce module contient les boucles d'entraînement et de validation, ainsi que la
fonction `main` pour lancer le processus complet.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset import load_data
from model import BertForSentimentClassification
from utils import set_seed, save_best_model, compute_metrics
import config

N_CLASSES = len(config.ID2LABEL)


def train_epoch(
        model,
        optimizer,
        criterion,
        train_loader,
        device="cpu"):
    """Effectue une passe d'entraînement complète sur le train loader.

    Returns:
        Tuple ``(train_loss, train_accuracy)``.
    """

    train_loss = 0.0
    all_preds = []
    all_labels = []
    model.train()

    num_batches = len(train_loader)
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device).long()
        attention_mask = batch["attention_mask"].to(device)
        output = model(inputs, attention_mask=attention_mask)  
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_loss = train_loss / num_batches
    train_acc, _ = compute_metrics(all_preds, all_labels)
    return train_loss, train_acc


def val_epoch(model,
              val_loader,
              criterion,
              device="cpu"):
    """Évalue le modèle sur un jeu de validation ou test.

    Returns:
        Tuple ``(valid_loss, accuracy, f1_weighted)``.
    """

    valid_loss = 0.0
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", unit="batch")
        for batch in pbar:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device).long()
            attention_mask = batch["attention_mask"].to(device)
            output = model(inputs, attention_mask=attention_mask)
            loss = criterion(output, labels)
            valid_loss += loss.item()

            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    valid_loss = valid_loss / len(val_loader)
    acc, f1 = compute_metrics(all_preds, all_labels)
    return valid_loss, acc, f1


def main():
    """Lance l'entraînement complet : boucle d'époques, sauvegarde du meilleur modèle, évaluation test.

    Produit ``best_bert_sentiment.pth`` dans le répertoire courant.
    Copier vers ``../model/best_model.pth`` pour l'inférence en production.
    """

    set_seed(config.RANDOM_STATE)

    MODEL_NAME = config.MODEL_NAME
    BATCH_SIZE = 16
    EPOCHS = 5
    LR = 2e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # dataset.py charge et tokenise déjà train/validation/test à partir des 3 CSV
    # et renomme la colonne label -> labels
    tokenized_dataset = load_data()

    train_loader = DataLoader(
        tokenized_dataset["train"],
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=BATCH_SIZE
    )
    test_loader = DataLoader(
        tokenized_dataset["test"],
        batch_size=BATCH_SIZE
    )

    model = BertForSentimentClassification(MODEL_NAME, n_class=N_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    best_val_loss = float("inf")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_f1_score": [],
        "val_accuracy": []
    }

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            optimizer,
            criterion,
            train_loader,
            device=device
        )
        valid_loss, valid_accuracy, valid_f1 = val_epoch(
            model,
            val_loader,
            criterion,
            device=device
        )

        best_val_loss = save_best_model(
            model,
            valid_loss,
            best_val_loss,
            "bert_sentiment"
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(valid_loss)
        history["val_f1_score"].append(valid_f1)
        history["val_accuracy"].append(valid_accuracy)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_accuracy={train_accuracy:.4f} | "
            f"val_loss={valid_loss:.4f} val_accuracy={valid_accuracy:.4f} | "
            f"val_f1_score={valid_f1:.4f}"
        )

    # évaluation finale sur le test set (jamais vu pendant l'entraînement ni pour le choix du meilleur modèle)
    test_loss, test_accuracy, test_f1 = val_epoch(
        model,
        test_loader,
        criterion,
        device=device
    )
    print(
        f"Test | test_loss={test_loss:.4f} "
        f"test_accuracy={test_accuracy:.4f} test_f1_score={test_f1:.4f}"
    )


if __name__ == "__main__":
    main()