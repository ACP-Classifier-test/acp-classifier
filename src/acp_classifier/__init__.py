"""acp-classifier — clasificación de péptidos anticancerígenos.

Universidad Industrial de Santander
Escuela de Ingeniería de Sistemas e Informática
Trabajo de grado, 2026
"""

__version__ = "1.0.1"

from .fasta import ErrorFasta, Peptido, leer_fasta
from .predictor import AcpClassifier, Prediccion

__all__ = [
    "AcpClassifier",
    "Prediccion",
    "Peptido",
    "leer_fasta",
    "ErrorFasta",
    "__version__",
]
