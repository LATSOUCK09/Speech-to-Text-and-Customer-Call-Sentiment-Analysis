"""
Export des résultats de transcription en CSV (vue synthétique)
et JSON (détail complet avec segments et timestamps).
"""

import os
import json
from typing import List

import pandas as pd

try:
    from .result import ResultatTranscription
except ImportError:  
    from result import ResultatTranscription


def sauvegarder_resultats(resultats: List[ResultatTranscription], dossier_sortie: str):
    """Sauvegarde les résultats de transcription en CSV et JSON.

    Produit :
        - ``transcriptions.csv`` : une ligne par appel (fichier, durée, texte)
        - ``transcriptions_detail.json`` : détail complet avec segments et timestamps

    Args:
        resultats: Liste de ``ResultatTranscription``.
        dossier_sortie: Répertoire de destination (créé si absent).
    """
    os.makedirs(dossier_sortie, exist_ok=True)

    # --- CSV : une ligne par appel ---
    lignes_csv = [
        {
            "fichier": r.fichier,
            "duree_sec": r.duree_sec,
            "transcription": r.transcription_complete,
            "erreur": r.erreur or "",
        }
        for r in resultats
    ]
    chemin_csv = os.path.join(dossier_sortie, "transcriptions.csv")
    pd.DataFrame(lignes_csv).to_csv(chemin_csv, index=False, encoding="utf-8-sig")
    print(f"CSV sauvegardé : {chemin_csv}")

    # --- JSON : détail complet avec segments ---
    lignes_json = [
        {
            "fichier": r.fichier,
            "duree_sec": r.duree_sec,
            "transcription_complete": r.transcription_complete,
            "segments": r.segments,
            "erreur": r.erreur,
        }
        for r in resultats
    ]
    chemin_json = os.path.join(dossier_sortie, "transcriptions_detail.json")
    with open(chemin_json, "w", encoding="utf-8") as f:
        json.dump(lignes_json, f, ensure_ascii=False, indent=2)
    print(f"JSON sauvegardé : {chemin_json}")
