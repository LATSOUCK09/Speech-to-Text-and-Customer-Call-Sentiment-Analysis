

"""Interface Gradio pour la démonstration de transcription audio et analyse de sentiment.

Ce module contient la logique de création de l'interface, le pipeline de
traitement et l'export de l'application Gradio.
"""

import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
import spaces


@lru_cache(maxsize=1)
def _get_runtime_components():
    """Charge et met en cache les composants du pipeline (modèles + config).

    Returns:
        Tuple ``(transcrire_appel, modele_asr, config_transcription, analyseur_sentiment)``.
    """
    from sentiment import AnalyseurSentiment
    from transcriptions.config import Config as ConfigTranscription
    from transcriptions.model import ModeleASR
    from transcriptions.pipeline import transcrire_appel

    config_transcription = ConfigTranscription()
    modele_asr = ModeleASR(config_transcription)
    analyseur_sentiment = AnalyseurSentiment()
    return transcrire_appel, modele_asr, config_transcription, analyseur_sentiment

@spaces.GPU
def pipeline(audio_path: str):
    """Pipeline de traitement appelé par Gradio.

    Args:
        audio_path: Chemin local du fichier audio uploadé via Gradio.

    Returns:
        transcription, label de sentiment et probabilités.
    """
    if not audio_path:
        return "", "indetermine", [["mecontent", 0.0], ["neutre", 0.0], ["satisfait", 0.0]]

    try:
        transcrire_appel, modele_asr, config_transcription, analyseur_sentiment = _get_runtime_components()
        resultat = transcrire_appel(audio_path, modele_asr, config_transcription)
        transcription = resultat.transcription_complete or ""

        if resultat.erreur:
            return transcription, "erreur", [["erreur", 1.0]]

        analyse = analyseur_sentiment.analyser(transcription)
        scores = analyse.get("scores", {})
        probabilites = [[label, score] for label, score in scores.items()]

        return transcription, analyse.get("label", "indetermine"), probabilites
    except Exception as exc:
        return f"Erreur lors du traitement : {exc}", "erreur", [["erreur", 1.0]]


def create_demo():
    """Construit l'interface Gradio pour l'application.

    Returns:
        Une instance `gr.Interface` prête à être lancée.
    """
    return gr.Interface(
        fn=pipeline,
        inputs=gr.Audio(type="filepath", label="Déposer un appel client"),
        outputs=[
            gr.Textbox(label="Transcription"),
            gr.Label(label="Sentiment détecté"),
            gr.Dataframe(label="Probabilités", headers=["Sentiment", "Probabilité"]),
        ],
        title="🎙 Analyse de sentiment des appels clients",
        description="""
Déposez un fichier audio.

Le système :
• transcrit automatiquement l'appel avec Wav2Vec2
• analyse le sentiment avec CamemBERT
• affiche la transcription ainsi que le sentiment détecté.
""",
    )


demo = create_demo()


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, quiet=True)