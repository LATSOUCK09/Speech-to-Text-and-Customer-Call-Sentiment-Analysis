"""Définit l'architecture PyTorch du classifieur de sentiment.

Cette classe encapsule CamemBERT pour la classification multi-classes de
sentiment sur des transcriptions de texte.
"""

import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from config import *


class BertForSentimentClassification(nn.Module):
    """Classifieur CamemBERT à 3 classes (mecontent, neutre, satisfait).

    Utilisé uniquement pendant l'entraînement. L'inférence en production
    passe par le module ``sentiment.py`` à la racine du projet.
    """

    def __init__(self, model_name=MODEL_NAME, n_class=NUM_CLASSES):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=n_class,
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )

    def forward(self, input_ids, attention_mask):
        """Passe avant : retourne les logits bruts."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits


SentimentClassifier = BertForSentimentClassification