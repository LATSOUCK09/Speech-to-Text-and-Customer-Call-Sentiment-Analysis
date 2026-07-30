# Documentation des fichiers du projet

Ce document décrit le rôle, les responsabilités et les interactions de chaque fichier
du dépôt **Speech-to-Text and Customer Call Sentiment Analysis**.

---

## Vue d'ensemble

```
Racine
├── main.py                 ──► CLI pipeline complet (ASR + sentiment)
├── api.py                  ──► Service REST FastAPI
├── app.py / gradio_demo.py ──► Interface web Gradio
├── sentiment.py            ──► Inférence sentiment (CamemBERT)
├── transcriptions/         ──► Pipeline ASR (Wav2Vec2)
├── sentiment-analysis/     ──► Entraînement du classifieur
├── examples/               ──► Scripts d'exemple pour l'API
├── model/                  ──► Poids fine-tunés (Git LFS)
├── Dockerfile              ──► Image Docker pour l'API
└── docker-compose.yml      ──► Orchestration conteneur
```

---

## Fichiers racine

### `main.py`

**Rôle :** Point d'entrée CLI pour le pipeline complet (transcription + sentiment).

**Flux :**
1. Transcription batch via `transcriptions.pipeline.traiter_lot`
2. Analyse de sentiment via `sentiment.AnalyseurSentiment`
3. Export CSV/JSON combiné dans le dossier de sortie

**Fonctions principales :**

| Fonction | Description |
|----------|-------------|
| `analyser_sentiments()` | Enrichit chaque `ResultatTranscription` avec label et scores de sentiment |
| `sauvegarder_resultats_combines()` | Écrit `resultats_finaux.csv` et `resultats_finaux.json` |
| `main()` | Parse les arguments CLI et orchestre les 3 étapes |

**Arguments CLI :**

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--input`, `-i` | *(requis)* | Fichier audio ou dossier d'appels |
| `--output`, `-o` | `resultats` | Dossier de sortie |
| `--chunk-duration` | `20.0` | Durée des segments ASR (s) |
| `--overlap` | `2.0` | Chevauchement entre segments (s) |

**Sorties :**
- `resultats/resultats_finaux.csv`
- `resultats/resultats_finaux.json`
- `resultats/_transcription_brute/` (transcriptions intermédiaires)

---

### `api.py`

**Rôle :** Expose le pipeline via une API HTTP FastAPI.

**Classes :**

| Classe | Description |
|--------|-------------|
| `PipelineService` | Chargement paresseux et thread-safe des modèles ASR et sentiment |

**Endpoints :**

| Route | Méthode | Description |
|-------|---------|-------------|
| `/` | GET | Informations et liste des routes |
| `/health` | GET | Health check |
| `/analyze` | POST | Upload audio → transcription + sentiment |

**Paramètres POST `/analyze` :**

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `file` | UploadFile | *(requis)* | Fichier audio (.wav, .mp3, .flac, .ogg, .m4a) |
| `chunk_duration` | float | `20.0` | Durée des segments |
| `overlap` | float | `2.0` | Chevauchement |

**Variables d'environnement :** `HOST` (défaut `127.0.0.1`), `PORT` (défaut `8000`)

---

### `app.py`

**Rôle :** Point d'entrée minimal pour lancer l'interface Gradio.

Importe `create_demo()` depuis `gradio_demo.py` et expose l'objet `app`.
Lancement : `python app.py` → port `7860` (configurable via `PORT`).

---

### `gradio_demo.py`

**Rôle :** Interface web interactive pour tester le pipeline.

**Fonctions :**

| Fonction | Description |
|----------|-------------|
| `_get_runtime_components()` | Charge et met en cache les modèles (LRU cache) |
| `pipeline()` | Callback Gradio : audio → transcription + sentiment |
| `create_demo()` | Construit l'interface `gr.Interface` |

**Entrées :** fichier audio (filepath)
**Sorties :** transcription (texte), label sentiment, tableau de probabilités

> Utilise le décorateur `@spaces.GPU` pour le déploiement Hugging Face Spaces.

---

### `sentiment.py`

**Rôle :** Module d'inférence pour l'analyse de sentiment sur texte transcrit.

**Constantes :**

| Constante | Valeur | Description |
|-----------|--------|-------------|
| `MODEL_NAME` | `camembert-base` | Modèle de base Hugging Face |
| `CHECKPOINT_PATH` | `model/best_model.pth` | Poids fine-tunés |
| `ID2LABEL` | `{0: mecontent, 1: neutre, 2: satisfait}` | Mapping classe → label |
| `MAX_LENGTH` | `256` | Troncature tokenizer |

**Classes :**

| Classe | Description |
|--------|-------------|
| `BertForSentimentClassification` | Wrapper PyTorch autour de CamemBERT pour classification 3 classes |
| `AnalyseurSentiment` | Charge le modèle une fois, expose `analyser(texte) → {label, scores}` |

**Utilisé par :** `main.py`, `api.py`, `gradio_demo.py`

---

### `requirements.txt`

Liste des dépendances Python : PyTorch, Transformers, librosa, FastAPI, Gradio, etc.

---

### `Dockerfile`

Image Docker basée sur `python:3.10-slim` :
- Installe `ffmpeg` et `libsndfile1`
- Installe les dépendances Python
- Expose le port 8000
- CMD : `python api.py`

---

### `docker-compose.yml`

Service unique `speech-sentiment-api` :
- Build depuis le Dockerfile local
- Port `8000:8000`
- Variables `HOST=0.0.0.0`, `PORT=8000`

---

## Module `transcriptions/`

Pipeline ASR (Automatic Speech Recognition) basé sur Wav2Vec2.

### `transcriptions/config.py`

**Rôle :** Configuration centralisée du pipeline de transcription.

**Classe `Config` (dataclass) :**

| Attribut | Défaut | Description |
|----------|--------|-------------|
| `model_name` | `jonatasgrosman/wav2vec2-large-xlsr-53-french` | Modèle Hugging Face |
| `sample_rate` | `16000` | Fréquence d'échantillonnage (Hz) |
| `device` | `cuda` ou `cpu` | Device PyTorch |
| `chunk_duration_sec` | `20.0` | Durée d'un segment (s) |
| `overlap_sec` | `2.0` | Chevauchement entre segments (s) |
| `min_duration_sec` | `0.3` | Durée minimale d'un segment |
| `silence_amplitude_threshold` | `1e-4` | Seuil de détection du silence |
| `extensions_audio` | `.wav, .mp3, .flac, .ogg, .m4a` | Formats acceptés |

---

### `transcriptions/preprocessing.py`

**Rôle :** Chargement et prétraitement audio.

| Fonction | Description |
|----------|-------------|
| `charger_et_pretraiter()` | Charge un fichier via librosa → mono, 16 kHz |
| `est_silencieux()` | Retourne `True` si l'amplitude max < seuil configuré |

---

### `transcriptions/chunking.py`

**Rôle :** Découpage et fusion des segments audio.

| Élément | Description |
|---------|-------------|
| `Segment` (dataclass) | Segment audio avec timing (`start_sec`, `end_sec`) et texte transcrit |
| `decouper_en_segments()` | Découpe un signal long en segments avec chevauchement |
| `fusionner_transcriptions()` | Concatène les textes des segments (peut produire des répétitions aux frontières) |

---

### `transcriptions/model.py`

**Rôle :** Chargement et inférence du modèle Wav2Vec2.

| Classe / Méthode | Description |
|------------------|-------------|
| `ModeleASR` | Charge `Wav2Vec2Processor` + `Wav2Vec2ForCTC` depuis Hugging Face |
| `ModeleASR.transcrire_segment()` | Transcrit un segment numpy (mono, 16 kHz) → texte |

---

### `transcriptions/pipeline.py`

**Rôle :** Orchestration du pipeline ASR complet.

| Fonction | Description |
|----------|-------------|
| `transcrire_appel()` | Pipeline pour un fichier : prétraitement → découpage → transcription → fusion |
| `lister_fichiers_audio()` | Résout un chemin (fichier ou dossier) en liste de fichiers audio |
| `traiter_lot()` | Traite un lot de fichiers, exporte les résultats, retourne la liste |

**Flux `transcrire_appel()` :**
```
chemin_fichier
  → charger_et_pretraiter()
  → decouper_en_segments()
  → [pour chaque segment] modele.transcrire_segment() (si non silencieux)
  → fusionner_transcriptions()
  → ResultatTranscription
```

---

### `transcriptions/result.py`

**Rôle :** Structure de données pour les résultats de transcription.

**Classe `ResultatTranscription` (dataclass) :**

| Champ | Type | Description |
|-------|------|-------------|
| `fichier` | str | Nom du fichier source |
| `duree_sec` | float | Durée totale de l'audio |
| `transcription_complete` | str | Texte transcrit complet |
| `segments` | List[Dict] | Détail par segment (timestamps + texte) |
| `erreur` | Optional[str] | Message d'erreur si échec |

---

### `transcriptions/export.py`

**Rôle :** Export des résultats de transcription.

| Fonction | Description |
|----------|-------------|
| `sauvegarder_resultats()` | Écrit `transcriptions.csv` (synthèse) et `transcriptions_detail.json` (détail) |

---

### `transcriptions/main.py`

**Rôle :** CLI pour la transcription seule (sans sentiment).

Mêmes arguments que `main.py` racine (`--input`, `--output`, `--chunk-duration`, `--overlap`).
Sortie par défaut : `resultats_transcription/`.

---

## Module `sentiment-analysis/`

Scripts d'entraînement du classifieur CamemBERT. **Non utilisés en inférence** — l'inférence passe par `sentiment.py` à la racine.

### `sentiment-analysis/config.py`

Hyperparamètres d'entraînement :

| Constante | Valeur | Description |
|-----------|--------|-------------|
| `MODEL_NAME` | `camembert-base` | Modèle de base |
| `MAX_LENGTH` | `256` | Longueur max des tokens |
| `BATCH_SIZE` | `16` | Taille de batch |
| `LR` | `2e-5` | Learning rate |
| `EPOCHS` | `5` | Nombre d'époques |
| `LABEL2ID` / `ID2LABEL` | 3 classes | mecontent, neutre, satisfait |

---

### `sentiment-analysis/dataset.py`

**Rôle :** Chargement et tokenisation des données d'entraînement.

| Fonction | Description |
|----------|-------------|
| `tokenize()` | Tokenise un batch et convertit les labels texte → indices |
| `load_data()` | Charge `data/train.csv`, `data/val.csv`, `data/test.csv` via HuggingFace datasets |

**Format CSV attendu :**
```csv
text,label
"Le client est satisfait",satisfait
```

---

### `sentiment-analysis/model.py`

**Rôle :** Architecture du classifieur (identique à `sentiment.py`).

| Classe | Description |
|--------|-------------|
| `BertForSentimentClassification` | CamemBERT + tête de classification 3 classes |
| `SentimentClassifier` | Alias de `BertForSentimentClassification` |

---

### `sentiment-analysis/train.py`

**Rôle :** Script d'entraînement complet.

| Fonction | Description |
|----------|-------------|
| `train_epoch()` | Une passe d'entraînement (loss + accuracy) |
| `val_epoch()` | Évaluation validation/test (loss + accuracy + F1) |
| `main()` | Boucle d'entraînement, sauvegarde du meilleur modèle, évaluation test |

**Sortie :** `best_bert_sentiment.pth` (à copier vers `model/best_model.pth` pour l'inférence)

---

### `sentiment-analysis/utils.py`

**Rôle :** Utilitaires d'entraînement.

| Fonction | Description |
|----------|-------------|
| `set_seed()` | Fixe les graines aléatoires (reproductibilité) |
| `save_best_model()` | Sauvegarde le modèle si la validation loss s'améliore |
| `compute_metrics()` | Calcule accuracy et F1 weighted |

---

## Dossier `examples/`

### `examples/call_api.py`

Script Python minimal pour tester l'endpoint `POST /analyze`.
Envoie un fichier audio et affiche la réponse JSON.

### `examples/call_api.sh`

Équivalent curl du script Python ci-dessus.

---

## Dossier `model/`

### `model/best_model.pth`

Poids fine-tunés du classifieur CamemBERT. Suivi via **Git LFS** (voir `.gitattributes`).

Chargé par `sentiment.py` → `AnalyseurSentiment`.

---

## Fichiers de configuration et CI

### `.gitignore`

Exclut : environnements virtuels, caches Python, `model/`, `*.pth`, fichiers audio, dossiers de résultats.

### `.gitattributes`

Configure Git LFS pour `model/*.pth`.

### `.github/workflows/sync.yml`

Synchronise automatiquement le dépôt vers Hugging Face Spaces (`LATSOUCK/Speech-Text-Analysis`) à chaque push sur `main`.

---

## Graphe de dépendances

```
main.py
  ├── transcriptions.pipeline ──► config, preprocessing, chunking, model, result, export
  └── sentiment.AnalyseurSentiment

api.py
  ├── transcriptions.config, model, pipeline
  └── sentiment.AnalyseurSentiment

gradio_demo.py
  ├── transcriptions.config, model, pipeline
  └── sentiment.AnalyseurSentiment

sentiment-analysis/train.py
  ├── dataset ──► config
  ├── model ──► config
  └── utils
```

---

## Fichiers de sortie générés

| Fichier | Produit par | Contenu |
|---------|-------------|---------|
| `transcriptions.csv` | `transcriptions/export.py` | Synthèse transcription |
| `transcriptions_detail.json` | `transcriptions/export.py` | Détail avec segments |
| `resultats_finaux.csv` | `main.py` | Transcription + sentiment |
| `resultats_finaux.json` | `main.py` | Détail complet avec scores |
| `best_bert_sentiment.pth` | `sentiment-analysis/train.py` | Meilleur modèle entraîné |
