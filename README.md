# acp-classifier

Clasificación de péptidos anticancerígenos (ACPs) a partir de secuencias en formato FASTA,
mediante representaciones contextuales generadas por el modelo de lenguaje proteico
**ESM-2** y un clasificador de **Regresión Logística**.

Desarrollado como trabajo de grado en la Escuela de Ingeniería de Sistemas e Informática de
la **Universidad Industrial de Santander**.

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

Como referencia, mACPpred 2.0 reporta un AUC-ROC de 0.817 sobre el mismo conjunto.

> El conjunto de evaluación está desbalanceado (18,1 % de positivos). Por ese motivo se
> reporta AUC-ROC como métrica principal, junto con F1 y MCC, y no la exactitud: un
> clasificador que asignara siempre la clase mayoritaria obtendría un 81,9 % de exactitud
> sin haber aprendido nada.

## Instalación

```bash
pip install acp-classifier
```

Desde el código fuente:

```bash
git clone https://github.com/uis-eisi/acp-classifier.git
cd acp-classifier
pip install -e .
```

Con Conda:

```bash
conda env create -f environment.yml
conda activate acp-classifier
```

Requiere Python 3.9 o superior. La primera ejecución descarga el modelo ESM-2 8M
(~31 MB) desde HuggingFace y lo deja en caché local.

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

## Reproducir el entrenamiento

```bash
python entrenar_modelo.py
```

Regenera el artefacto desde los FASTA de mACPpred 2.0, compara las dos estrategias de
agregación implementadas e imprime las métricas sobre el conjunto de prueba independiente.
Toma alrededor de un minuto sobre CPU.

## Pruebas

```bash
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
  url    = {https://github.com/uis-eisi/acp-classifier}
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
