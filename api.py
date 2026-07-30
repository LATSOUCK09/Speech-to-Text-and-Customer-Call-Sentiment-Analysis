"""API HTTP pour la transcription d'appels et l'analyse de sentiment.

Ce module expose un service FastAPI capable de recevoir un fichier audio,
le transcrire puis analyser le sentiment du texte obtenu.
"""

import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI(
    title="Speech-to-Text and Sentiment API",
    description="Expose le pipeline de transcription + analyse de sentiment via une API HTTP.",
    version="1.0.0",
)


class PipelineService:
    """Service de pipeline pour initialiser et exécuter la transcription + analyse.

    Ce service charge paresseusement les composants du pipeline lorsque la
    première requête arrive, puis conserve les objets en mémoire pour les
    requêtes suivantes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._components: Optional[dict] = None

    def get_components(self):
        """Initialise et retourne les composants du pipeline.

        Utilise un verrou pour garantir que les composants sont construits une
        seule fois dans un contexte multi-thread.
        """
        if self._components is None:
            with self._lock:
                if self._components is None:
                    from transcriptions.config import Config as ConfigTranscription
                    from transcriptions.model import ModeleASR
                    from transcriptions.pipeline import transcrire_appel
                    from sentiment import AnalyseurSentiment

                    config = ConfigTranscription()
                    modele_asr = ModeleASR(config)
                    analyseur_sentiment = AnalyseurSentiment()
                    self._components = {
                        "config": config,
                        "modele_asr": modele_asr,
                        "analyseur_sentiment": analyseur_sentiment,
                        "transcrire_appel": transcrire_appel,
                    }
        return self._components

    def traiter_audio(self, chemin_audio: str, chunk_duration: float, overlap: float) -> dict:
        """Transcrit un fichier audio puis analyse le sentiment du texte.

        Args:
            chemin_audio: Chemin vers le fichier audio temporaire.
            chunk_duration: Durée des segments d'audio en secondes.
            overlap: Recouvrement entre segments en secondes.

        Returns:
            Un dictionnaire contenant le résultat de la transcription et de l'analyse.
        """
        components = self.get_components()
        config = components["config"]
        config.chunk_duration_sec = chunk_duration
        config.overlap_sec = overlap

        resultat = components["transcrire_appel"](
            chemin_audio,
            components["modele_asr"],
            config,
        )

        if resultat.erreur:
            return {
                "fichier": resultat.fichier,
                "duree_sec": resultat.duree_sec,
                "transcription": "",
                "sentiment": "erreur",
                "sentiment_scores": {},
                "erreur": resultat.erreur,
                "segments": [],
            }

        analyse = components["analyseur_sentiment"].analyser(resultat.transcription_complete)
        return {
            "fichier": resultat.fichier,
            "duree_sec": resultat.duree_sec,
            "transcription": resultat.transcription_complete,
            "sentiment": analyse.get("label", "indetermine"),
            "sentiment_scores": analyse.get("scores", {}),
            "erreur": None,
            "segments": resultat.segments,
        }


service = PipelineService()


@app.get("/")
def root() -> dict:
    """Point d'entrée principal de l'API.

    Renvoie un message de bienvenue et les routes disponibles.
    """
    return {
        "message": "API prête à traiter des fichiers audio.",
        "endpoints": {"health": "/health", "analyze": "/analyze"},
    }


@app.get("/health")
def health() -> dict:
    """Vérifie que le service est opérationnel.

    Retourne un statut simple qui peut être utilisé pour une vérification de santé.
    """
    return {"status": "ok", "message": "API prête à traiter des fichiers audio."}


@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    chunk_duration: float = 20.0,
    overlap: float = 2.0,
) -> dict:
    """Reçoit un fichier audio, le transcrit et analyse le sentiment.

    Args:
        file: Fichier audio uploadé via multipart/form-data.
        chunk_duration: Durée en secondes des segments de traitement audio.
        overlap: Recouvrement en secondes entre les segments audio.

    Returns:
        Un dictionnaire contenant la transcription, l'analyse du sentiment et
        les métadonnées du fichier.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Aucun fichier audio fourni.")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
        raise HTTPException(status_code=400, detail="Format audio non supporté.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        return service.traiter_audio(tmp_path, chunk_duration=chunk_duration, overlap=overlap)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api:app", host=host, port=port, reload=False)
   
