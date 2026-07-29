"""
Structure de données représentant le résultat de transcription d'un appel.
Placée dans son propre fichier pour éviter les imports circulaires entre
pipeline.py et export.py (les deux en ont besoin).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ResultatTranscription:
    fichier: str
    duree_sec: float
    transcription_complete: str
    segments: List[Dict] = field(default_factory=list)
    erreur: Optional[str] = None
