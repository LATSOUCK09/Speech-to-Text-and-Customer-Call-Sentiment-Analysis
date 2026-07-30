"""
Chargement du modèle wav2vec2 et transcription d'un segment audio.
"""

import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

try:
    from .config import Config
except ImportError:  
    from config import Config


class ModeleASR:
    """Modèle Wav2Vec2 pour la reconnaissance vocale (CTC).

    Charge ``Wav2Vec2Processor`` et ``Wav2Vec2ForCTC`` depuis Hugging Face
    et les réutilise pour transcrire tous les segments d'un lot.

    Args:
        config: Configuration contenant ``model_name`` et ``device``.
    """

    def __init__(self, config: Config):
        self.config = config
        print(f"Chargement du modèle '{config.model_name}' sur '{config.device}'...")
        self.processor = Wav2Vec2Processor.from_pretrained(config.model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(config.model_name)
        self.model.to(config.device)
        self.model.eval()
        print("Modèle chargé.")

    @torch.no_grad()
    def transcrire_segment(self, audio: np.ndarray) -> str:
        """Transcrit un segment audio prétraité (mono, 16 kHz).

        Args:
            audio: Signal numpy 1D.

        Returns:
            Texte transcrit (chaîne vide possible si silence/bruit).
        """
        inputs = self.processor(
            audio,
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.config.device)

        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        texte = self.processor.batch_decode(predicted_ids)[0]

        return texte.strip()
