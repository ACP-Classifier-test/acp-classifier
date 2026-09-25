"""Entrena y serializa el modelo final que distribuye el paquete acp-classifier.

Reproduce la configuración seleccionada en el trabajo de grado:
    embeddings de ESM-2 8M (esm2_t6_8M_UR50D) + Regresión Logística (C=0.01, saga)
entrenada sobre Training.fasta de mACPpred 2.0 y evaluada sobre Independent.fasta.

Compara además las dos variantes de agregación (mean pooling) descritas en
`embedder.py`, para documentar cuál se distribuye y por qué.

Uso:
    python entrenar_modelo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))

from acp_classifier.embedder import Esm2Embedder  # noqa: E402
from acp_classifier.fasta import limpiar  # noqa: E402

DATOS = RAIZ.parent / "dataset" / "mACPpred2" / "data"
TRAIN_FASTA = DATOS / "Training.fasta"
IND_FASTA = DATOS / "Independent.fasta"
DESTINO = RAIZ / "src" / "acp_classifier" / "modelos"
CACHE = RAIZ / ".cache_embeddings"

SEED = 42
HIPERPARAMETROS = {"C": 0.01, "solver": "saga", "max_iter": 1000, "random_state": SEED}


def leer_etiquetado(ruta: Path) -> tuple[list[str], np.ndarray]:
    """Lee un FASTA de mACPpred 2.0 conservando la etiqueta de la cabecera."""
    secuencias, etiquetas = [], []
    etiqueta = None
    partes: list[str] = []

    def cerrar():
        if etiqueta is None:
            return
        limpia = limpiar("".join(partes))
        if len(limpia) >= 2:
            secuencias.append(limpia)
            etiquetas.append(etiqueta)

    with ruta.open(encoding="utf-8") as fh:
        for linea in fh:
            linea = linea.strip()
            if not linea:
                continue
            if linea.startswith(">"):
                cerrar()
                etiqueta = 1 if "Positive" in linea else 0
                partes = []
            else:
                partes.append(linea)
    cerrar()
    return secuencias, np.array(etiquetas, dtype=int)


def embeddings(secuencias, pooling, nombre):
    CACHE.mkdir(exist_ok=True)
    ruta = CACHE / f"{nombre}_{pooling}.npy"
    if ruta.exists():
        X = np.load(ruta)
        if X.shape[0] == len(secuencias):
            print(f"    caché: {ruta.name} {X.shape}")
            return X
    t0 = time.perf_counter()
    X = Esm2Embedder(pooling=pooling)(secuencias)
    np.save(ruta, X)
    print(f"    generados {X.shape} en {time.perf_counter() - t0:.0f}s")
    return X


def evaluar(Xtr, ytr, Xte, yte):
    scaler = StandardScaler().fit(Xtr)
    modelo = LogisticRegression(**HIPERPARAMETROS).fit(scaler.transform(Xtr), ytr)
    prob = modelo.predict_proba(scaler.transform(Xte))[:, 1]
    pred = (prob >= 0.5).astype(int)
    return modelo, scaler, {
        "auc_roc": round(float(roc_auc_score(yte, prob)), 4),
        "f1": round(float(f1_score(yte, pred)), 4),
        "exactitud": round(float(accuracy_score(yte, pred)), 4),
        "mcc": round(float(matthews_corrcoef(yte, pred)), 4),
    }


def main() -> int:
    print("=" * 70)
    print("  ENTRENAMIENTO DEL MODELO FINAL — acp-classifier")
    print("=" * 70)

    tr_seq, y_tr = leer_etiquetado(TRAIN_FASTA)
    te_seq, y_te = leer_etiquetado(IND_FASTA)
    print(f"\n  Entrenamiento: {len(tr_seq)} ({y_tr.sum()} ACP / {(y_tr == 0).sum()} no-ACP)")
    print(f"  Prueba:        {len(te_seq)} ({y_te.sum()} ACP / {(y_te == 0).sum()} no-ACP)")

    resultados = {}
    artefactos = {}
    for pooling in ("tesis", "estricto"):
        print(f"\n  --- pooling '{pooling}' ---")
        Xtr = embeddings(tr_seq, pooling, "train")
        Xte = embeddings(te_seq, pooling, "test")
        modelo, scaler, met = evaluar(Xtr, y_tr, Xte, y_te)
        resultados[pooling] = met
        artefactos[pooling] = (modelo, scaler)
        print(f"    AUC-ROC {met['auc_roc']} | F1 {met['f1']} | "
              f"Exactitud {met['exactitud']} | MCC {met['mcc']}")

    print("\n" + "=" * 70)
    print("  COMPARACIÓN DE ESTRATEGIAS DE POOLING")
    print("=" * 70)
    print(f"  {'estrategia':<12}{'AUC-ROC':>10}{'F1':>10}{'MCC':>10}")
    for k, m in resultados.items():
        print(f"  {k:<12}{m['auc_roc']:>10}{m['f1']:>10}{m['mcc']:>10}")
    print("\n  Referencia reportada en el trabajo (pooling 'tesis'): AUC-ROC 0.8745")
    print("  Referencia mACPpred 2.0 sobre el mismo conjunto:      AUC-ROC 0.887")
    print("  Referencia MLACP 2.0 sobre el mismo conjunto:         AUC-ROC 0.893")

    # Se distribuye la variante reportada en el documento, para que el paquete
    # coincida exactamente con las métricas defendidas en el trabajo de grado.
    elegido = "tesis"
    modelo, scaler = artefactos[elegido]

    DESTINO.mkdir(parents=True, exist_ok=True)
    (DESTINO / "__init__.py").write_text(
        '"""Artefactos serializados del modelo entrenado."""\n', encoding="utf-8"
    )
    artefacto = {
        "modelo": modelo,
        "scaler": scaler,
        "metadatos": {
            "version": "1.0.0",
            "modelo_lenguaje": "facebook/esm2_t6_8M_UR50D",
            "dimensiones": int(modelo.coef_.shape[1]),
            "clasificador": "LogisticRegression",
            "hiperparametros": HIPERPARAMETROS,
            "pooling": elegido,
            "entrenado_con": "mACPpred 2.0 Training.fasta (1176 ACP / 1176 no-ACP)",
            "evaluado_con": "mACPpred 2.0 Independent.fasta (610 ACP / 2760 no-ACP)",
            "metricas": resultados[elegido],
            "metricas_pooling_estricto": resultados["estricto"],
        },
    }
    salida = DESTINO / "acp_esm2_8m_lr.joblib"
    joblib.dump(artefacto, salida, compress=3)
    print(f"\n  Modelo guardado: {salida}  ({salida.stat().st_size / 1024:.0f} KB)")
    print(json.dumps(artefacto["metadatos"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
