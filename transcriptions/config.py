"""Configuration centralisée du pipeline de transcription ASR.

Toutes les valeurs modifiables (modèle, sample rate, découpage, silence)
sont regroupées dans la dataclass ``Config``.
"""

from dataclasses import dataclass
import torch


@dataclass
class Config:
    """Paramètres du pipeline Wav2Vec2.

    Attributes:
        model_name: Identifiant Hugging Face du modèle ASR.
        sample_rate: Fréquence d'échantillonnage cible (Hz).
        device: Device PyTorch (``cuda`` ou ``cpu``).
        chunk_duration_sec: Durée de chaque segment audio (s).
        overlap_sec: Chevauchement entre segments consécutifs (s).
        min_duration_sec: Durée minimale pour tenter une transcription.
        silence_amplitude_threshold: Seuil d'amplitude max pour ignorer un segment.
        extensions_audio: Extensions de fichiers acceptées.
    """
    # Modèle
    model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-french"
    sample_rate: int = 16000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Découpage des appels longs
    chunk_duration_sec: float = 20.0   # durée de chaque segment
    overlap_sec: float = 2.0           # chevauchement entre segments

    # Filtrage des segments silencieux
    min_duration_sec: float = 0.3
    silence_amplitude_threshold: float = 1e-4

    # Formats acceptés
    extensions_audio: tuple = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
