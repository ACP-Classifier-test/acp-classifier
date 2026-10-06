# Registro de cambios

El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/)
y el versionado sigue [SemVer](https://semver.org/lang/es/).

## [1.0.2] — 2026-10-06

Corrección del entorno de instalación. No hay cambios en el código, en el
modelo ni en los resultados: la predicción es idéntica a la de la 1.0.1.

### Corregido

- **La instalación en Linux ya no arrastra CUDA.** La rueda de `torch` publicada
  en PyPI declara como dependencias obligatorias `triton` y cuatro paquetes
  `nvidia-cu13` (cuDNN, NCCL, cuSPARSELt, nvSHMEM), unos 2 GB de runtime que el
  paquete no usa porque la inferencia corre en CPU. En una máquina sin el CUDA
  Toolkit eso terminaba en `fatal error: cuda.h: No such file or directory`. Se
  añade `requirements-cpu.txt`, que declara el índice de ruedas de CPU de PyTorch
  como índice principal: las ruedas `+cpu` no declaran ninguna de esas cinco
  dependencias y, por la PEP 440, `2.14.1+cpu` ordena por encima de `2.14.1`, así
  que `pip` las prefiere frente a las de PyPI.
- **`environment.yml` usa `pytorch-cpu` en lugar de `pytorch`.** El metapaquete
  `pytorch` de conda-forge resuelve en Linux hacia la variante de GPU. La sección
  de pip pasa a usar `--no-deps`, porque sin esa bandera reinstalaba `torch`
  desde PyPI —con CUDA— encima del `pytorch-cpu` recién puesto por conda.

### Cambiado

- **El rango de versiones de Python pasa de `>=3.9` a `>=3.10,<3.15`.** PyTorch
  dejó de publicar ruedas para 3.9 tras la serie 2.8, de modo que el rango
  anterior permitía crear entornos en los que `pip` intentaba compilar PyTorch
  desde el código fuente, con el mismo error de `cuda.h`. El límite superior
  evita el caso simétrico en versiones de Python todavía sin ruedas. Ahora `pip`
  rechaza el entorno incompatible antes de descargar nada.
- Se añade `.python-version` con `3.11`, la versión de referencia del proyecto.
- Los pisos de las dependencias suben a los primeros con ruedas en todo el rango
  admitido: `numpy>=1.24`, `joblib>=1.3`, `torch>=2.2` y `transformers>=4.40`.
- El README documenta la versión de Python exigida, la instalación por la vía de
  CPU y una sección de diagnóstico para los errores de `triton`, `cuda.h` y los
  paquetes `nvidia-*`.
- La integración continua amplía la matriz a Python 3.10–3.14 y añade una corrida
  en Windows y otra en macOS. Dos guardas nuevas: una falla si en el entorno
  aparecen `triton` o paquetes `nvidia-*`, o si en Linux la rueda de `torch` no es
  `+cpu`; la otra comprueba que la instalación en Python 3.9 se rechaza con un
  mensaje claro.

## [1.0.1] — 2026-09-29

Cambio de ubicación del repositorio. No hay cambios en el código, en el modelo
ni en los resultados: la predicción es idéntica a la de la 1.0.0.

### Cambiado

- El proyecto pasa a alojarse en una organización propia. Se actualizan las
  seis declaraciones de la URL del repositorio en `pyproject.toml` (Homepage,
  Repository, Issues), `README.md` y `CITATION.cff`.

## [1.0.0] — 2026-09-24

Primera versión publicada en PyPI. Corresponde al modelo final del trabajo de grado.

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
  instalación no requiere descargar artefactos adicionales salvo los pesos de ESM-2. El
  artefacto guarda los **pesos como arrays de NumPy** (coeficientes, intercepto, media y
  escala) en lugar de objetos de scikit-learn serializados con pickle: la predicción se
  calcula directamente y **no depende de la versión de scikit-learn** instalada, lo que
  evita los avisos `InconsistentVersionWarning` y el riesgo de que una versión futura de la
  biblioteca impidiera cargar el modelo.
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

### Notas de empaquetado

- **`scikit-learn` no es una dependencia de ejecución.** Vive en los extras `[dev]`, donde lo
  necesita `entrenar_modelo.py`. El paquete instalado requiere únicamente `numpy`, `joblib`,
  `torch` y `transformers`.
- La licencia se declara mediante expresión SPDX (`license = "MIT"`), conforme a la PEP 639.
- Los artefactos generados durante el desarrollo en el formato anterior —objetos de
  scikit-learn serializados— siguen cargándose correctamente, siempre que `scikit-learn`
  esté disponible en el entorno.

### Notas

- La validación es exclusivamente computacional (TRL 4): una predicción positiva es una
  hipótesis a verificar experimentalmente.
- El desempeño está caracterizado sobre un único benchmark.
- La cifra de referencia del estado del arte citada en el README está pendiente de
  verificación contra la tabla del artículo original.
