

import gradio as gr
from transcriptions.pipeline import transcrire_appel
from transcriptions.config import Config as ConfigTranscription
from transcriptions.model import ModeleASR
from sentiment import AnalyseurSentiment

config_transcription = ConfigTranscription()
modele_asr = ModeleASR(config_transcription)
analyseur_sentiment = AnalyseurSentiment()


def pipeline(audio_path: str):
    if not audio_path:
        return "", "indetermine", [["mecontent", 0.0], ["neutre", 0.0], ["satisfait", 0.0]]

    resultat = transcrire_appel(audio_path, modele_asr, config_transcription)
    transcription = resultat.transcription_complete or ""

    if resultat.erreur:
        return transcription, "erreur", [["erreur", 1.0]]

    analyse = analyseur_sentiment.analyser(transcription)
    scores = analyse.get("scores", {})
    probabilites = [[label, score] for label, score in scores.items()]

    return transcription, analyse.get("label", "indetermine"), probabilites


demo = gr.Interface(
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

demo.launch()