"""
Orchestration du pipeline complet.
Ce fichier ne fait qu'assembler les briques des autres modules :
préprocessing -> découpage -> transcription -> export.
"""

from pathlib import Path
from typing import List

from tqdm import tqdm

try:
    from .config import Config
    from .preprocessing import charger_et_pretraiter, est_silencieux
    from .chunking import decouper_en_segments, fusionner_transcriptions
    from .model import ModeleASR
    from .result import ResultatTranscription
    from .export import sauvegarder_resultats
except ImportError:  
    from config import Config
    from preprocessing import charger_et_pretraiter, est_silencieux
    from chunking import decouper_en_segments, fusionner_transcriptions
    from model import ModeleASR
    from result import ResultatTranscription
    from export import sauvegarder_resultats


def transcrire_appel(chemin_fichier: str, modele: ModeleASR, config: Config) -> ResultatTranscription:
    """Pipeline complet pour un seul fichier audio."""
    nom_fichier = Path(chemin_fichier).name

    try:
        # 1. Prétraitement (mono, 16 kHz)
        audio = charger_et_pretraiter(chemin_fichier, config)
        duree_sec = len(audio) / config.sample_rate

        # 2. Découpage en segments
        segments = decouper_en_segments(audio, config)
        print(f"'{nom_fichier}' : {duree_sec:.1f}s -> {len(segments)} segment(s)")

        # 3. Transcription segment par segment
        for segment in segments:
            duree_segment = segment.end_sec - segment.start_sec
            if duree_segment < config.min_duration_sec or est_silencieux(segment.audio, config):
                segment.texte = ""
                continue
            segment.texte = modele.transcrire_segment(segment.audio)

        # 4. Recomposition
        transcription_complete = fusionner_transcriptions(segments)

        return ResultatTranscription(
            fichier=nom_fichier,
            duree_sec=round(duree_sec, 2),
            transcription_complete=transcription_complete,
            segments=[
                {"debut_sec": round(s.start_sec, 2), "fin_sec": round(s.end_sec, 2), "texte": s.texte}
                for s in segments
            ],
        )

    except Exception as e:
        print(f"Erreur sur '{nom_fichier}' : {e}")
        return ResultatTranscription(
            fichier=nom_fichier, duree_sec=0.0, transcription_complete="", erreur=str(e)
        )


def lister_fichiers_audio(chemin: str, config: Config) -> List[str]:
    """Retourne la liste des fichiers audio à traiter (fichier unique ou dossier)."""
    chemin_obj = Path(chemin)

    if chemin_obj.is_file():
        return [str(chemin_obj)]

    if chemin_obj.is_dir():
        return [
            str(p) for p in sorted(chemin_obj.iterdir())
            if p.suffix.lower() in config.extensions_audio
        ]

    raise FileNotFoundError(f"Chemin introuvable : {chemin}")


def traiter_lot(chemin_entree: str, dossier_sortie: str, config: Config) -> List[ResultatTranscription]:
    """Traite un fichier unique ou un dossier entier d'appels, puis exporte les résultats."""
    fichiers = lister_fichiers_audio(chemin_entree, config)

    if not fichiers:
        print("Aucun fichier audio trouvé.")
        return []

    print(f"{len(fichiers)} fichier(s) audio à transcrire.")
    modele = ModeleASR(config)  # chargé une seule fois pour tout le lot

    resultats = [
        transcrire_appel(f, modele, config)
        for f in tqdm(fichiers, desc="Transcription des appels")
    ]

    sauvegarder_resultats(resultats, dossier_sortie)
    return resultats
