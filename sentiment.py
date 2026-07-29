"""
Analyse de sentiment sur le texte transcrit d'un appel.
Tout est regroupé ici : config, architecture du modèle, chargement, inférence.
"""

from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RACINE_PROJET = Path(__file__).parent
CHECKPOINT_PATH = RACINE_PROJET / "model" / "best_model.pth"

MODEL_NAME = "camembert-base"
ID2LABEL = {0: "mecontent", 1: "neutre", 2: "satisfait"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
MAX_LENGTH = 256


class BertForSentimentClassification(nn.Module):
    """Votre architecture, inchangée."""

    def __init__(self, model_name=MODEL_NAME, n_class=len(ID2LABEL)):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=n_class, id2label=ID2LABEL, label2id=LABEL2ID
        )

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


class AnalyseurSentiment:
    """Charge le modèle une seule fois, puis l'appelle sur chaque transcription."""

    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Chargement du tokenizer '{MODEL_NAME}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        print(f"Chargement des poids depuis '{checkpoint_path}'...")
        self.model = BertForSentimentClassification()
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("Modèle de sentiment chargé.")

    @torch.no_grad()
    def analyser(self, texte: str) -> dict:
        """Retourne {"label": ..., "scores": {...}}"""
        if not texte or not texte.strip():
            return {"label": "indetermine", "scores": {}}

        inputs = self.tokenizer(
            texte, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True
        )
        logits = self.model(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
        )
        probabilites = torch.softmax(logits, dim=-1)[0]
        id_predit = int(torch.argmax(probabilites).item())

        return {
            "label": ID2LABEL[id_predit],
            "scores": {ID2LABEL[i]: round(p.item(), 4) for i, p in enumerate(probabilites)},
        }