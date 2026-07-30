"""
Structure de données représentant le résultat de transcription d'un appel.
Placée dans son propre fichier pour éviter les imports circulaires entre
pipeline.py et export.py (les deux en ont besoin).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ResultatTranscription:
    """Résultat de la transcription d'un appel audio.

    Attributes:
        fichier: Nom du fichier source.
        duree_sec: Durée totale de l'audio en secondes.
        transcription_complete: Texte transcrit reconstitué.
        segments: Liste de dicts ``{debut_sec, fin_sec, texte}`` par segment.
        erreur: Message d'erreur si le traitement a échoué, sinon ``None``.
    """
    fichier: str
    duree_sec: float
    transcription_complete: str
    segments: List[Dict] = field(default_factory=list)
    erreur: Optional[str] = None
