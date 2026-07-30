"""
Point d'entrée en ligne de commande pour la transcription d'appels.

Utilisation :
  python main.py --input appel1.wav --output resultats/
  python main.py --input dossier_appels/ --output resultats/
"""

import argparse

try:
    from .config import Config
    from .pipeline import traiter_lot
except ImportError:  
    from config import Config
    from pipeline import traiter_lot


def main():
    """Parse les arguments de la ligne de commande et exécute le pipeline."""
    parser = argparse.ArgumentParser(
        description="Pipeline de transcription d'appels clientèle (wav2vec2 français)"
    )
    parser.add_argument("--input", "-i", required=True,
                         help="Fichier audio unique OU dossier contenant plusieurs appels")
    parser.add_argument("--output", "-o", default="resultats_transcription",
                         help="Dossier de sortie (CSV + JSON)")
    parser.add_argument("--chunk-duration", type=float, default=20.0,
                         help="Durée (sec) de chaque segment pour le découpage")
    parser.add_argument("--overlap", type=float, default=2.0,
                         help="Chevauchement (sec) entre segments consécutifs")

    args = parser.parse_args()

    config = Config(chunk_duration_sec=args.chunk_duration, overlap_sec=args.overlap)
    resultats = traiter_lot(args.input, args.output, config)

    n_ok = sum(1 for r in resultats if r.erreur is None)
    n_erreur = len(resultats) - n_ok
    print(f"Terminé : {n_ok} appel(s) transcrit(s) avec succès, {n_erreur} en erreur.")


if __name__ == "__main__":
    main()
