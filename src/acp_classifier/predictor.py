"""Clasificador de péptidos anticancerígenos."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from importlib import resources
from pathlib import Path

import joblib
import numpy as np

from .embedder import Esm2Embedder
from .fasta import Peptido, leer_fasta

UMBRAL_POR_DEFECTO = 0.5


@dataclass(frozen=True)
class Prediccion:
    identificador: str
    secuencia: str
    longitud: int
    probabilidad_acp: float
    prediccion: str

    def como_dict(self) -> dict:
        return asdict(self)


class AcpClassifier:
    """Predice actividad anticancerígena a partir de secuencias peptídicas.

    Combina embeddings de ESM-2 8M con un clasificador de Regresión Logística,
    seleccionado por obtener el mayor AUC-ROC sobre el conjunto de prueba
    independiente de mACPpred 2.0 (Sangaraju et al., 2024).

    Ejemplo
    -------
    >>> from acp_classifier import AcpClassifier
    >>> clf = AcpClassifier()
    >>> clf.predecir_secuencias(["KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"])
    """

    def __init__(self, ruta_modelo: str | Path | None = None):
        artefacto = self._cargar_artefacto(ruta_modelo)
        self.modelo = artefacto["modelo"]
        self.scaler = artefacto["scaler"]
        self.metadatos = artefacto["metadatos"]
        self._embedder: Esm2Embedder | None = None

    @staticmethod
    def _cargar_artefacto(ruta_modelo):
        if ruta_modelo is not None:
            return joblib.load(Path(ruta_modelo))
        with resources.as_file(
            resources.files("acp_classifier.modelos") / "acp_esm2_8m_lr.joblib"
        ) as p:
            return joblib.load(p)

    @property
    def embedder(self) -> Esm2Embedder:
        """El modelo de lenguaje se carga de forma diferida, en el primer uso."""
        if self._embedder is None:
            self._embedder = Esm2Embedder(pooling=self.metadatos["pooling"])
        return self._embedder

    def probabilidades(self, secuencias, tamano_lote: int = 32) -> np.ndarray:
        X = self.embedder(secuencias, tamano_lote=tamano_lote)
        return self.modelo.predict_proba(self.scaler.transform(X))[:, 1]

    def predecir_secuencias(
        self,
        secuencias,
        umbral: float = UMBRAL_POR_DEFECTO,
        tamano_lote: int = 32,
    ) -> list[Prediccion]:
        peptidos = [
            p if isinstance(p, Peptido) else Peptido(f"secuencia_{i + 1}", p)
            for i, p in enumerate(secuencias)
        ]
        probas = self.probabilidades([p.secuencia for p in peptidos], tamano_lote)
        return [
            Prediccion(
                identificador=p.identificador,
                secuencia=p.secuencia,
                longitud=p.longitud,
                probabilidad_acp=round(float(pr), 4),
                prediccion="ACP" if pr >= umbral else "no-ACP",
            )
            for p, pr in zip(peptidos, probas)
        ]

    def predecir_fasta(
        self,
        ruta_fasta: str | Path,
        umbral: float = UMBRAL_POR_DEFECTO,
        tamano_lote: int = 32,
    ) -> list[Prediccion]:
        return self.predecir_secuencias(leer_fasta(ruta_fasta), umbral, tamano_lote)
