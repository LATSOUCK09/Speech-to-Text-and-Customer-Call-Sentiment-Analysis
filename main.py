"""Point d'entrée CLI — pipeline complet transcription + sentiment.

Orchestre le traitement batch d'appels clients :
  1. Transcription ASR (Wav2Vec2) via ``transcriptions.pipeline``
  2. Analyse de sentiment (CamemBERT) via ``sentiment.AnalyseurSentiment``
  3. Export des résultats combinés en CSV et JSON

Utilisation :
  python main.py --input appel1.wav --output resultats/
  python main.py --input dossier_appels/ --output resultats/
"""


import sys
import os
import json
import argparse
from pathlib import Path

import pandas as pd

RACINE_PROJET = Path(__file__).parent
sys.path.insert(0, str(RACINE_PROJET / "transcription"))

from transcriptions.config import Config as ConfigTranscription          
from transcriptions.pipeline import traiter_lot as transcrire_lot_appels  

from sentiment import AnalyseurSentiment                   


def analyser_sentiments(resultats_transcription, analyseur: AnalyseurSentiment):
    """Enrichit chaque transcription avec un label et des scores de sentiment.

    Args:
        resultats_transcription: Liste de ``ResultatTranscription`` produite par le pipeline ASR.
        analyseur: Instance ``AnalyseurSentiment`` déjà chargée.

    Returns:
        Liste de dictionnaires prêts à l'export, chacun contenant
        ``fichier``, ``duree_sec``, ``transcription``, ``sentiment``,
        ``sentiment_scores`` et ``erreur``.
    """
    resultats_combines = []

    for resultat in resultats_transcription:
        if resultat.erreur:
            resultats_combines.append({
                "fichier": resultat.fichier,
                "duree_sec": resultat.duree_sec,
                "transcription": resultat.transcription_complete,
                "sentiment": None,
                "sentiment_scores": None,
                "erreur": resultat.erreur,
            })
            continue

        analyse = analyseur.analyser(resultat.transcription_complete)

        resultats_combines.append({
            "fichier": resultat.fichier,
            "duree_sec": resultat.duree_sec,
            "transcription": resultat.transcription_complete,
            "sentiment": analyse["label"],
            "sentiment_scores": analyse["scores"],
            "erreur": None,
        })

    return resultats_combines


def sauvegarder_resultats_combines(resultats_combines, dossier_sortie: str):
    """Exporte le résultat final (transcription + sentiment) en CSV et JSON.

    Produit ``resultats_finaux.csv`` (vue tabulaire) et
    ``resultats_finaux.json`` (détail complet avec scores par classe).

    Args:
        resultats_combines: Liste retournée par ``analyser_sentiments()``.
        dossier_sortie: Répertoire de destination (créé si absent).
    """
    os.makedirs(dossier_sortie, exist_ok=True)

    lignes_csv = [
        {
            "fichier": r["fichier"],
            "duree_sec": r["duree_sec"],
            "transcription": r["transcription"],
            "sentiment": r["sentiment"] or "",
            "score_satisfait": (r["sentiment_scores"] or {}).get("satisfait", ""),
            "score_neutre": (r["sentiment_scores"] or {}).get("neutre", ""),
            "score_mecontent": (r["sentiment_scores"] or {}).get("mecontent", ""),
            "erreur": r["erreur"] or "",
        }
        for r in resultats_combines
    ]
    chemin_csv = os.path.join(dossier_sortie, "resultats_finaux.csv")
    pd.DataFrame(lignes_csv).to_csv(chemin_csv, index=False, encoding="utf-8-sig")
    print(f"CSV final sauvegardé : {chemin_csv}")

    chemin_json = os.path.join(dossier_sortie, "resultats_finaux.json")
    with open(chemin_json, "w", encoding="utf-8") as f:
        json.dump(resultats_combines, f, ensure_ascii=False, indent=2)
    print(f"JSON final sauvegardé : {chemin_json}")


def main():
    """Parse les arguments de la ligne de commande et lance le traitement batch."""
    parser = argparse.ArgumentParser(
        description="Transcription d'appels clientèle + analyse de sentiment"
    )
    parser.add_argument("--input", "-i", required=True,
                         help="Fichier audio unique OU dossier contenant plusieurs appels")
    parser.add_argument("--output", "-o", default="resultats",
                         help="Dossier de sortie pour les résultats finaux")
    parser.add_argument("--chunk-duration", type=float, default=20.0,
                         help="Durée (sec) de chaque segment audio pour le découpage")
    parser.add_argument("--overlap", type=float, default=2.0,
                         help="Chevauchement (sec) entre segments audio consécutifs")

    args = parser.parse_args()

    # --- Étape 1 : transcription de tous les appels ---
    config_transcription = ConfigTranscription(
        chunk_duration_sec=args.chunk_duration,
        overlap_sec=args.overlap,
    )
    # dossier_sortie_transcription : le pipeline ASR sauvegarde aussi ses propres
    # fichiers intermédiaires (utile pour debug), séparément du résultat final
    dossier_intermediaire = os.path.join(args.output, "_transcription_brute")
    resultats_transcription = transcrire_lot_appels(
        args.input, dossier_intermediaire, config_transcription
    )

    if not resultats_transcription:
        print("Aucun résultat de transcription, arrêt.")
        return

    # --- Étape 2 : analyse de sentiment sur chaque transcription ---
    analyseur_sentiment = AnalyseurSentiment()
    resultats_combines = analyser_sentiments(resultats_transcription, analyseur_sentiment)

    # --- Étape 3 : export du résultat final combiné ---
    sauvegarder_resultats_combines(resultats_combines, args.output)

    # --- Résumé ---
    n_ok = sum(1 for r in resultats_combines if r["erreur"] is None)
    n_erreur = len(resultats_combines) - n_ok
    print(f"\nTerminé : {n_ok} appel(s) traité(s) avec succès, {n_erreur} en erreur.")

    if n_ok:
        repartition = pd.Series([r["sentiment"] for r in resultats_combines if r["sentiment"]]).value_counts()
        print("\nRépartition des sentiments détectés :")
        print(repartition.to_string())


if __name__ == "__main__":
    main()