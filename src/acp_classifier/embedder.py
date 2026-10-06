"""Generación de embeddings con ESM-2 8M (esm2_t6_8M_UR50D).

El modelo se descarga de HuggingFace la primera vez (~31 MB) y queda en caché
local. No requiere GPU: sobre CPU procesa varios cientos de secuencias por minuto.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np

MODELO = "facebook/esm2_t6_8M_UR50D"
DIMENSIONES = 320

Pooling = Literal["tesis", "estricto"]


def _mascara_tesis(attention_mask):
    """Enmascarado empleado para obtener los resultados reportados en el trabajo.

    Excluye el token <cls> y la ÚLTIMA POSICIÓN CON ATENCIÓN CONTADA TRAS ANULAR <cls>,
    que corresponde al último aminoácido y no al token <eos>. Se conserva tal cual
    porque es el procedimiento con el que se entrenó y evaluó el modelo distribuido
    (AUC-ROC de 0.8745 sobre el conjunto de prueba independiente de mACPpred 2.0).
    """
    mask = attention_mask.clone()
    mask[:, 0] = 0
    for j, longitud in enumerate(mask.sum(dim=1)):
        if longitud > 1:
            mask[j, longitud - 1] = 0
    return mask


def _mascara_estricta(attention_mask):
    """Excluye exactamente <cls> y <eos>, dejando solo los aminoácidos reales."""
    mask = attention_mask.clone()
    mask[:, 0] = 0
    longitudes = attention_mask.sum(dim=1)
    for j, longitud in enumerate(longitudes):
        if longitud > 1:
            mask[j, longitud - 1] = 0
    return mask


class Esm2Embedder:
    """Convierte secuencias peptídicas en vectores de 320 dimensiones."""

    def __init__(self, pooling: Pooling = "tesis", dispositivo: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from transformers import logging as transformers_logging

        if pooling not in ("tesis", "estricto"):
            raise ValueError(f"pooling debe ser 'tesis' o 'estricto', no {pooling!r}")

        self._torch = torch
        self.pooling = pooling
        self.dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(MODELO)

        # El punto de control publicado por HuggingFace es el del modelo de lenguaje
        # enmascarado, mientras que aquí solo se emplea el codificador. Al cargarlo,
        # transformers emite un informe que marca como UNEXPECTED los pesos de
        # `lm_head` —que este paquete no usa— y como MISSING los del `pooler`, que
        # inicializaría al azar. Ninguna de las dos cosas altera el resultado: el
        # promediado se hace a mano en __call__ y el `pooler` nunca se invoca. Pero
        # el informe aparece en rojo en cada ejecución y aparenta un problema.
        #
        # `add_pooling_layer=False` evita construir esa capa inútil, que es la causa
        # real de la mitad del aviso. El resto del informe se silencia solo durante
        # la carga, restaurando después el nivel previo para no ocultar advertencias
        # legítimas que transformers emita más adelante.
        nivel_previo = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        try:
            modelo = AutoModel.from_pretrained(MODELO, add_pooling_layer=False)
        except TypeError:
            # Algún backend de transformers podría no admitir ese argumento.
            # El aviso reaparecería, pero la predicción es la misma.
            modelo = AutoModel.from_pretrained(MODELO)
        finally:
            transformers_logging.set_verbosity(nivel_previo)

        self.model = modelo.to(self.dispositivo).eval()

    def __call__(self, secuencias: Iterable[str], tamano_lote: int = 32) -> np.ndarray:
        secuencias = list(secuencias)
        if not secuencias:
            return np.empty((0, DIMENSIONES), dtype=np.float32)

        torch = self._torch
        lotes: list[np.ndarray] = []

        for i in range(0, len(secuencias), tamano_lote):
            lote = secuencias[i : i + tamano_lote]
            entradas = self.tokenizer(
                lote, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            entradas = {k: v.to(self.dispositivo) for k, v in entradas.items()}

            with torch.no_grad():
                salidas = self.model(**entradas)

            oculto = salidas.last_hidden_state
            construir = _mascara_tesis if self.pooling == "tesis" else _mascara_estricta
            mask = construir(entradas["attention_mask"])

            expandida = mask.unsqueeze(-1).float()
            suma = (oculto * expandida).sum(dim=1)
            cuenta = expandida.sum(dim=1).clamp(min=1e-9)
            lotes.append((suma / cuenta).cpu().numpy())

        return np.vstack(lotes).astype(np.float32)
