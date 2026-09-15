"""Interfaz de línea de comandos de acp-classifier."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from . import __version__
from .fasta import ErrorFasta

CAMPOS = ["identificador", "secuencia", "longitud", "probabilidad_acp", "prediccion"]


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="acp-classifier",
        description="Clasificación de péptidos anticancerígenos (ACPs) a partir de "
        "secuencias en formato FASTA, mediante embeddings de ESM-2 y "
        "Regresión Logística.",
        epilog="Ejemplo:  acp-classifier -i peptidos.fasta -o resultados.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", required=True, metavar="FASTA",
                   help="archivo FASTA de entrada con una o varias secuencias")
    p.add_argument("-o", "--output", metavar="CSV",
                   help="archivo CSV de salida (por defecto se imprime en pantalla)")
    p.add_argument("-u", "--umbral", type=float, default=0.5, metavar="P",
                   help="probabilidad mínima para clasificar como ACP")
    p.add_argument("-b", "--batch-size", type=int, default=32, metavar="N",
                   help="secuencias procesadas por lote")
    p.add_argument("-m", "--modelo", metavar="RUTA",
                   help="artefacto .joblib alternativo (por defecto, el incluido)")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="suprime los mensajes de progreso")
    p.add_argument("--version", action="version", version=f"acp-classifier {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)

    if not 0.0 < args.umbral < 1.0:
        print("Error: el umbral debe estar entre 0 y 1 (exclusivos).", file=sys.stderr)
        return 2

    def log(mensaje: str) -> None:
        if not args.quiet:
            print(mensaje, file=sys.stderr)

    try:
        from .predictor import AcpClassifier

        log("Cargando modelo...")
        clf = AcpClassifier(args.modelo)

        log(f"Leyendo {args.input}...")
        predicciones = clf.predecir_fasta(args.input, args.umbral, args.batch_size)

    except ErrorFasta as e:
        print(f"Error de lectura: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"Archivo no encontrado: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 - la CLI no debe volcar trazas al usuario
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    filas = [p.como_dict() for p in predicciones]

    if args.output:
        destino = Path(args.output)
        destino.parent.mkdir(parents=True, exist_ok=True)
        with destino.open("w", newline="", encoding="utf-8") as fh:
            escritor = csv.DictWriter(fh, fieldnames=CAMPOS)
            escritor.writeheader()
            escritor.writerows(filas)
        log(f"Resultados guardados en {destino}")
    else:
        escritor = csv.DictWriter(sys.stdout, fieldnames=CAMPOS)
        escritor.writeheader()
        escritor.writerows(filas)

    n_acp = sum(f["prediccion"] == "ACP" for f in filas)
    log(f"\n{len(filas)} secuencias analizadas: {n_acp} ACP, {len(filas) - n_acp} no-ACP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
