"""
Prétraitement audio : chargement d'un fichier, conversion mono, rééchantillonnage.

(La normalisation z-score de l'amplitude est gérée automatiquement par le
Wav2Vec2Processor au moment de la transcription — voir model.py — donc on ne
la refait pas ici, pour éviter de la faire deux fois.)
"""

import numpy as np
import librosa

try:
    from .config import Config
except ImportError:  
    from config import Config


def charger_et_pretraiter(chemin_fichier: str, config: Config) -> np.ndarray:
    """Charge un fichier audio et le convertit en mono / 16 kHz."""
    signal, _ = librosa.load(chemin_fichier, sr=config.sample_rate, mono=True)

    if len(signal) == 0:
        raise ValueError("Fichier audio vide.")

    return signal


def est_silencieux(audio: np.ndarray, config: Config) -> bool:
    """Détecte un segment quasi-silencieux (peu utile de le transcrire)."""
    return np.max(np.abs(audio)) < config.silence_amplitude_threshold
