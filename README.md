# acp-classifier

Clasificación de péptidos anticancerígenos (ACPs) a partir de secuencias en formato FASTA,
mediante representaciones contextuales generadas por el modelo de lenguaje proteico
**ESM-2** y un clasificador de **Regresión Logística**.

Desarrollado como trabajo de grado en la Escuela de Ingeniería de Sistemas e Informática de
la **Universidad Industrial de Santander**.

> **Nota para quien venga del documento de grado.** El Apéndice A del documento indica
> «Python 3.9 o superior». Desde la versión **1.0.2** el rango admitido es **3.10 a 3.14**,
> por el motivo que se explica en [Versión de Python](#versión-de-python). Este README,
> versionado junto al código, es la referencia vigente sobre instalación y uso: el
> documento describe el estado del paquete en el momento de su entrega.

---

## Por qué existe

Las herramientas de referencia para predicción de ACPs —AntiCP 2.0, MLACP 2.0, mACPpred 2.0,
ACP-EPC— están disponibles únicamente como servidores web, o como repositorios sin empaquetar.
Ninguna se instala mediante gestores estándar, lo que dificulta su integración en flujos de
trabajo bioinformáticos automatizados.

`acp-classifier` se instala con `pip`, se ejecuta desde consola y no requiere GPU.

## Desempeño

Evaluado sobre el **conjunto de prueba independiente oficial** de mACPpred 2.0
(610 ACPs y 2.760 no-ACPs), sin haber visto ninguna de esas secuencias durante el entrenamiento:

| Métrica | Valor |
|---|---|
| AUC-ROC | **0.8745** |
| F1-score | 0.6230 |
| Exactitud | 0.8362 |
| MCC | 0.5341 |

Como referencia, sobre ese mismo conjunto mACPpred 2.0 reporta un AUC-ROC de 0.887 y
MLACP 2.0 uno de 0.893. Este paquete no supera a ninguna de las dos, pero sí a las otras
nueve herramientas con AUC-ROC publicado sobre el conjunto —entre ellas mACPpred (0.842),
iDACP (0.852) y AntiCP 2.0 en su variante alterna (0.835)—, y lo hace con un modelo de ocho
millones de parámetros y un clasificador lineal, sin GPU en inferencia.

> El conjunto de evaluación está desbalanceado (18,1 % de positivos). Por ese motivo se
> reporta AUC-ROC como métrica principal, junto con F1 y MCC, y no la exactitud: un
> clasificador que asignara siempre la clase mayoritaria obtendría un 81,9 % de exactitud
> sin haber aprendido nada.

## Instalación

### Versión de Python

| | |
|---|---|
| **Admitidas** | 3.10, 3.11, 3.12, 3.13 y 3.14 |
| **Recomendada** | **3.11** — es la que fija `.python-version`, la que usa la integración continua y la que tiene ruedas precompiladas de todas las dependencias en las tres plataformas |
| **No admitidas** | 3.9 o anterior (PyTorch dejó de publicar ruedas tras la serie 2.8) y 3.15 o posterior (sin cobertura verificada) |

El rango está declarado en los metadatos del paquete, así que `pip` rechaza una
versión incompatible con un mensaje claro en vez de intentar compilar PyTorch
desde el código fuente, que es lo que termina en el error de `cuda.h` descrito
más abajo.

#### Por qué cambió respecto del documento de grado

El Apéndice A del documento de grado indica «Python 3.9 o superior», que era el
rango declarado por las versiones 1.0.0 y 1.0.1. Resultó ser incorrecto en la
práctica: PyTorch dejó de publicar ruedas precompiladas para Python 3.9 tras la
serie 2.8, de modo que en un entorno 3.9 `pip` no encontraba rueda, intentaba
compilar PyTorch desde el código fuente y la instalación terminaba en

```
fatal error: cuda.h: No such file or directory
```

El rango se corrigió a `>=3.10,<3.15` en la versión 1.0.2. El límite inferior
excluye las versiones sin ruedas; el superior evita el mismo problema en
versiones de Python todavía no cubiertas. Con ello `pip` rechaza el entorno
incompatible antes de descargar nada, en lugar de fallar a mitad de la
compilación con un error que no apunta a su causa.

Por el mismo motivo, la integración continua que el documento describe sobre
«Python 3.9, 3.10 y 3.11» cubre hoy **3.10 a 3.14 en Linux, más una corrida en
Windows y otra en macOS**, y verifica además que la instalación en 3.9 se
rechace con un mensaje claro.

### Instalación recomendada (CPU)

El paquete corre en CPU; no necesita GPU ni el CUDA Toolkit. Conviene instalar
PyTorch desde el índice de ruedas de CPU **antes** que el paquete, para que no
se descargue el runtime de CUDA:

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate

pip install -r https://raw.githubusercontent.com/ACP-Classifier-test/acp-classifier/main/requirements-cpu.txt
pip install acp-classifier
```

Desde una copia del repositorio:

```bash
git clone https://github.com/ACP-Classifier-test/acp-classifier.git
cd acp-classifier

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements-cpu.txt
pip install -e .
```

### Con Conda

```bash
conda env create -f environment.yml
conda activate acp-classifier
```

El archivo fija `python=3.11` y usa `pytorch-cpu`, no `pytorch`: el segundo
resuelve en Linux hacia la variante de GPU.

### Instalación rápida

```bash
pip install acp-classifier
```

Funciona en Windows y macOS, donde la rueda de PyPI no declara dependencias de
CUDA. **En Linux arrastra unos 2 GB de CUDA** que el paquete no usa, por el
motivo que se explica a continuación.

La primera ejecución descarga el modelo ESM-2 8M (~31 MB) desde HuggingFace y lo
deja en caché local.

### Plataformas verificadas

Hay ruedas precompiladas de `torch` en el índice de CPU para las cinco versiones
de Python admitidas, tanto en Linux (`manylinux_2_28_x86_64`) como en Windows
(`win_amd64`), y ninguna de ellas declara dependencias de CUDA. La integración
continua instala y ejecuta la suite completa en esta matriz:

| Sistema | Python | Vía de instalación |
|---|---|---|
| Linux x86_64 | 3.10, 3.11, 3.12, 3.13, 3.14 | `requirements-cpu.txt` |
| Windows AMD64 | 3.11 | `requirements-cpu.txt` |
| macOS arm64 | 3.11 | `requirements-cpu.txt` |

Cada corrida comprueba además que en el entorno no haya quedado `triton` ni
ningún paquete `nvidia-*`, y que en Linux la rueda de `torch` sea la `+cpu`.

## Problemas de instalación

### Errores de CUDA, `triton` o `cuda.h`

Los tres síntomas más frecuentes son:

- `fatal error: cuda.h: No such file or directory`
- `ModuleNotFoundError: No module named 'triton'`
- la instalación descarga varios paquetes `nvidia-*` de cientos de megabytes

Tienen el mismo origen. En **Linux**, la rueda de `torch` publicada en PyPI
declara como dependencias **obligatorias** `triton` y cuatro paquetes del
runtime de CUDA. Así las declara, por ejemplo, `torch` 2.14.1:

```
nvidia-cudnn-cu13      ; platform_system == "Linux"
nvidia-cusparselt-cu13 ; platform_system == "Linux"
nvidia-nccl-cu13       ; platform_system == "Linux"
nvidia-nvshmem-cu13    ; platform_system == "Linux"
triton~=3.8.0          ; platform_system == "Linux" and python_version < "3.15"
```

No son opcionales: un `pip install torch` las instala siempre, aunque la máquina
no tenga GPU. Si falta el CUDA Toolkit, `triton` falla al compilar sus kernels y
aparece el error de `cuda.h`. Si además la versión de Python no tiene rueda
precompilada, `pip` intenta construir PyTorch desde el código fuente y el mismo
error aparece durante la instalación.

Los cinco marcadores dependen de `platform_system == "Linux"`, así que **el
problema es exclusivo de Linux**: en Windows y macOS la rueda de PyPI no declara
ninguna dependencia de CUDA. Eso explica por qué la misma instalación funciona en
un computador y falla en otro.

**Solución.** Instalar PyTorch desde el índice de CPU, cuyas ruedas `+cpu` no
declaran ninguna de esas cinco dependencias:

```bash
pip uninstall -y torch triton
pip install -r requirements-cpu.txt
```

O, en un solo comando:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Para comprobar que el entorno quedó limpio:

```bash
python -c "import torch; print(torch.__version__)"   # debe terminar en '+cpu' en Linux
pip list | grep -Ei "triton|nvidia"                  # no debe devolver nada
```

### `ERROR: Package 'acp-classifier' requires a different Python`

La versión de Python del entorno está fuera del rango admitido. Crear un entorno
con 3.11:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
```

### El modelo tarda en la primera ejecución

Es la descarga de los pesos de ESM-2 8M (~31 MB) desde HuggingFace. Quedan en
caché (`~/.cache/huggingface`) y las ejecuciones siguientes no vuelven a bajarlos.

## Uso desde consola

```bash
# Resultados por pantalla
acp-classifier -i peptidos.fasta

# Resultados a un archivo CSV
acp-classifier -i peptidos.fasta -o resultados.csv

# Umbral de decisión más exigente
acp-classifier -i peptidos.fasta -o resultados.csv --umbral 0.8
```

Opciones:

| Opción | Descripción | Por defecto |
|---|---|---|
| `-i`, `--input` | Archivo FASTA de entrada (obligatorio) | — |
| `-o`, `--output` | Archivo CSV de salida | pantalla |
| `-u`, `--umbral` | Probabilidad mínima para clasificar como ACP | `0.5` |
| `-b`, `--batch-size` | Secuencias procesadas por lote | `32` |
| `-m`, `--modelo` | Artefacto `.joblib` alternativo | el incluido |
| `-q`, `--quiet` | Suprime los mensajes de progreso | — |

### Entrada

Un archivo FASTA estándar, con una o varias secuencias. Se admiten registros repartidos en
varias líneas. Los residuos no estándar (`X`, `B`, `Z`, `U`, `O`) se descartan, igual que
durante el entrenamiento.

```fasta
>peptido_1
GLFSVVTGVLKAVGKNVAKNVGGSLLEQLKCKKISGGC
>peptido_2
NNEQIRRFSQVLLDAGIVTTIRKTRGDDIDA
```

### Salida

```csv
identificador,secuencia,longitud,probabilidad_acp,prediccion
peptido_1,GLFSVVTGVLKAVGKNVAKNVGGSLLEQLKCKKISGGC,38,0.7350,ACP
peptido_2,NNEQIRRFSQVLLDAGIVTTIRKTRGDDIDA,31,0.0643,no-ACP
```

## Uso desde Python

```python
from acp_classifier import AcpClassifier

clf = AcpClassifier()

# Desde secuencias
for p in clf.predecir_secuencias(["GIGKFLHSAKKFGKAFVGEIMNS"]):
    print(p.identificador, p.probabilidad_acp, p.prediccion)

# Desde un archivo FASTA
predicciones = clf.predecir_fasta("peptidos.fasta", umbral=0.7)

# Solo las probabilidades, como arreglo de NumPy
probas = clf.probabilidades(["GIGKFLHSAKKFGKAFVGEIMNS", "AAAAAAAAAA"])
```

## Cómo funciona

```
FASTA → limpieza → ESM-2 8M → mean pooling → StandardScaler → Regresión Logística → CSV
        (20 aa)    (320 dim)                                    (C=0.01, saga)
```

1. **Lectura y limpieza.** Se conservan únicamente los 20 aminoácidos estándar.
2. **Embeddings.** `esm2_t6_8M_UR50D` genera un vector de 320 dimensiones por secuencia,
   promediando los estados ocultos de la última capa.
3. **Normalización.** `StandardScaler` ajustado sobre el conjunto de entrenamiento.
4. **Clasificación.** Regresión Logística (`C=0.01`, solver `saga`), entrenada sobre las
   2.352 secuencias de `Training.fasta` de mACPpred 2.0.

### Por qué el modelo más pequeño

Durante el trabajo se evaluaron ESM-2 8M, ESM-2 150M y ProtBERT. **ESM-2 8M obtuvo el mejor
desempeño**, pese a ser el de menor capacidad. Los péptidos del conjunto de evaluación tienen
una longitud mediana de 9 residuos: sobre secuencias tan cortas, las representaciones de mayor
dimensionalidad no aportan poder discriminante y favorecen el sobreajuste.

Esto tiene una consecuencia práctica: el modelo con mejor desempeño es también el de menores
requisitos (31 MB frente a los 1,7 GB de ProtBERT), lo que permite ejecutarlo sin GPU.

## Rendimiento

Medido sobre un procesador AMD Ryzen 7 de ocho núcleos, sin GPU:

| Operación | Tiempo |
|---|---|
| Carga del modelo | 1,22 s |
| Clasificación de 100 péptidos, con el modelo ya cargado | 0,52 s |
| Proceso completo desde consola, incluido el arranque | 9,7 – 10,1 s |
| Rendimiento sostenido | ~193 péptidos/s |

La diferencia entre los 0,52 s de clasificación y los cerca de diez del proceso completo
corresponde al tiempo de importación de las bibliotecas de aprendizaje profundo, no al
cálculo. En lotes grandes ese coste se paga una sola vez.

## Reproducir el entrenamiento

```bash
python entrenar_modelo.py
```

Regenera el artefacto desde los FASTA de mACPpred 2.0, compara las dos estrategias de
agregación implementadas e imprime las métricas sobre el conjunto de prueba independiente.
Toma alrededor de un minuto sobre CPU.

## Pruebas

```bash
pip install -r requirements-cpu.txt
pip install -e ".[dev]"
pytest -q
```

## Limitaciones

- La validación es **exclusivamente computacional** (TRL 4). Una predicción positiva es una
  hipótesis a verificar experimentalmente, no una afirmación de actividad biológica.
- El modelo fue entrenado sobre péptidos de entre 6 y 50 residuos. Su comportamiento fuera de
  ese rango no ha sido caracterizado.
- Existe un desplazamiento de distribución conocido entre los conjuntos de entrenamiento y
  prueba del benchmark: los ACPs de entrenamiento tienen una longitud mediana de 20 residuos
  frente a 9 en el conjunto de prueba.
- El desempeño está caracterizado sobre un único benchmark (mACPpred 2.0).

## Datos

Conjunto de referencia de **mACPpred 2.0** (Sangaraju et al., 2024): 1.176 ACPs y 1.176
no-ACPs para entrenamiento; 610 ACPs y 2.760 no-ACPs para prueba independiente.

## Citación

```bibtex
@software{acp_classifier_2026,
  title  = {acp-classifier: clasificación de péptidos anticancerígenos
            mediante modelos de lenguaje proteico},
  author = {Navarro Sinuco, Brayan Arturo and Mendoza Oñate, Ivan Andres},
  year   = {2026},
  school = {Universidad Industrial de Santander},
  url    = {https://github.com/ACP-Classifier-test/acp-classifier}
}
```

### Referencias

- Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic level protein structure with
  a language model. *Science*, 379(6637), 1123–1130.
- Sangaraju, V. K. et al. (2024). mACPpred 2.0: Stacked deep learning for anticancer peptide
  prediction. *Journal of Molecular Biology*, 436(17), 168687.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
- Wolf, T. et al. (2020). Transformers: State-of-the-art natural language processing. *EMNLP:
  System Demonstrations*, 38–45.

## Licencia

MIT. Véase [LICENSE](LICENSE).
