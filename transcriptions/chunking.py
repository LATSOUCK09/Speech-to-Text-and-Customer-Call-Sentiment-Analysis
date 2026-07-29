"""
Découpage d'un long signal audio en segments courts, avec chevauchement,
et recomposition du texte final à partir des segments transcrits.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

try:
    from .config import Config
except ImportError: 
    from config import Config


@dataclass
class Segment:
    """Un segment audio découpé, avec son timing d'origine."""
    audio: np.ndarray
    start_sec: float
    end_sec: float
    texte: str = ""


def decouper_en_segments(audio: np.ndarray, config: Config) -> List[Segment]:
    """
    Découpe un signal en segments de taille fixe avec chevauchement.

    Le chevauchement évite qu'un mot prononcé pile à la frontière d'un
    découpage soit coupé en deux et mal reconnu.
    """
    sr = config.sample_rate
    chunk_len = int(config.chunk_duration_sec * sr)
    overlap_len = int(config.overlap_sec * sr)
    pas = chunk_len - overlap_len

    duree_totale = len(audio) / sr

    if duree_totale <= config.chunk_duration_sec:
        return [Segment(audio=audio, start_sec=0.0, end_sec=duree_totale)]

    segments = []
    debut = 0

    while debut < len(audio):
        fin = min(debut + chunk_len, len(audio))
        segments.append(Segment(
            audio=audio[debut:fin],
            start_sec=debut / sr,
            end_sec=fin / sr,
        ))
        if fin == len(audio):
            break
        debut += pas

    return segments


def fusionner_transcriptions(segments: List[Segment]) -> str:
    """
    Recompose le texte complet à partir des segments transcrits.

    Version simple : concaténation avec espace. À cause du chevauchement
    audio, de légères répétitions de mots peuvent apparaître aux frontières.
    """
    return " ".join(s.texte for s in segments if s.texte)
