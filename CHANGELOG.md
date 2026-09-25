# Registro de cambios

El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/)
y el versionado sigue [SemVer](https://semver.org/lang/es/).

## [1.0.0] — 2026-09-24

Primera versión publicada en PyPI.

### Cambiado

- El artefacto del modelo guarda ahora los **pesos como arrays de NumPy** (coeficientes,
  intercepto, media y escala) en lugar de objetos de scikit-learn serializados con pickle.
  La predicción se calcula directamente y **ya no depende de la versión de scikit-learn**
  instalada, lo que elimina los avisos `InconsistentVersionWarning` y el riesgo de que una
  versión futura de la biblioteca impidiera cargar el modelo. Las predicciones son idénticas:
  AUC-ROC 0.8745, F1 0.6230, exactitud 0.8362 y MCC 0.5341 sobre el conjunto de prueba
  independiente, los mismos valores que reporta el trabajo de grado.
- **`scikit-learn` deja de ser una dependencia de ejecución.** Pasa a los extras `[dev]`,
  donde lo necesita `entrenar_modelo.py`. El paquete instalado requiere ahora únicamente
  `numpy`, `joblib`, `torch` y `transformers`.
- La licencia se declara mediante expresión SPDX (`license = "MIT"`), conforme a la PEP 639.

Los artefactos en el formato anterior siguen cargándose correctamente, siempre que
`scikit-learn` esté disponible en el entorno.

## [1.0.0] — 2026-09-07

Primera versión publicable. Corresponde al modelo final del trabajo de grado.

### Añadido

- Clasificador `AcpClassifier`: embeddings de ESM-2 8M (`esm2_t6_8M_UR50D`, 320 dimensiones)
  con *mean pooling*, `StandardScaler` y Regresión Logística (`C=0.01`, solver `saga`),
  entrenado sobre las 2.352 secuencias de `Training.fasta` de mACPpred 2.0.
- Interfaz de línea de comandos `acp-classifier`, con opciones de umbral (`-u`), tamaño de
  lote (`-b`), modelo alternativo (`-m`) y salida silenciosa (`-q`).
- API de Python: `predecir_fasta`, `predecir_secuencias` y `probabilidades`.
- Lectura de FASTA tolerante a secuencias repartidas en varias líneas, con descarte de los
  residuos no estándar (`X`, `B`, `Z`, `U`, `O`), igual que durante el entrenamiento.
- Modelo serializado incluido en la rueda (`modelos/acp_esm2_8m_lr.joblib`), de modo que la
  instalación no requiere descargar artefactos adicionales salvo los pesos de ESM-2.
- `entrenar_modelo.py`, que regenera el artefacto desde los FASTA del benchmark y reimprime
  las métricas sobre el conjunto de prueba independiente.
- 15 pruebas con `pytest` y flujo de integración continua en GitHub Actions.

### Desempeño

Sobre el conjunto de prueba independiente de mACPpred 2.0 (610 ACPs y 2.760 no-ACPs), no
visto durante el entrenamiento:

| Métrica | Valor |
|---|---|
| AUC-ROC | 0,8745 |
| F1-score | 0,6230 |
| Exactitud | 0,8362 |
| MCC | 0,5341 |

### Notas

- La validación es exclusivamente computacional (TRL 4): una predicción positiva es una
  hipótesis a verificar experimentalmente.
- El desempeño está caracterizado sobre un único benchmark.
- La cifra de referencia del estado del arte citada en el README está pendiente de
  verificación contra la tabla del artículo original.
