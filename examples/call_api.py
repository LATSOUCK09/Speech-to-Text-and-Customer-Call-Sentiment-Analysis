"""Exemple de client simple pour tester l'API de transcription et d'analyse.

Ce script envoie un fichier audio à l'endpoint `/analyze` et affiche la
réponse JSON.
"""

import os
import requests

url = "http://127.0.0.1:8000/analyze"
audio_path = os.path.join("audio", "ElevenLabs_2026-07-28T17_47_35_Mélanie - Storyteller_pvc_sp115_s53_sb60_se4_b_m2.mp3")

if not os.path.exists(audio_path):
    raise FileNotFoundError(f"Fichier audio introuvable : {audio_path}")

with open(audio_path, "rb") as fh:
    files = {"file": (os.path.basename(audio_path), fh, "audio/mpeg")}
    response = requests.post(
        url,
        files=files,
        data={"chunk_duration": 20.0, "overlap": 2.0},
        timeout=600,
    )

print("Status:", response.status_code)
print(response.json())
