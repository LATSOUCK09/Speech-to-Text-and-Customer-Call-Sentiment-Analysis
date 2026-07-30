"""Configuration d'entraînement pour le module sentiment-analysis.

Ce fichier regroupe les hyperparamètres, les labels et les mappings
texte→indice utilisés dans la préparation des données, la construction du
modèle et l'entraînement.
"""

MODEL_NAME="camembert-base"
MAX_LENGTH=256
BATCH_SIZE=16
LR=2e-5
EPOCHS=5
WEIGHT_DECAY=0.01
SEED=42
RANDOM_STATE=42
NUM_CLASSES=3
LABEL2ID = {
    "mecontent":0,
    "neutre":1,
    "satisfait":2
}
ID2LABEL = {
    0:"mecontent",
    1:"neutre",
    2:"satisfait"
}