"""Pruebas del paquete acp-classifier.

    pytest -q
"""

from pathlib import Path

import pytest

from acp_classifier import AcpClassifier, ErrorFasta, __version__, leer_fasta
from acp_classifier.cli import main
from acp_classifier.fasta import limpiar

EJEMPLO = Path(__file__).resolve().parent.parent / "examples" / "ejemplo.fasta"


# ── FASTA ────────────────────────────────────────────────────────────────────
def test_limpiar_descarta_residuos_no_estandar():
    assert limpiar("acdX efZ") == "ACDEF"


def test_lee_el_ejemplo():
    peptidos = leer_fasta(EJEMPLO)
    assert len(peptidos) == 8
    assert peptidos[0].identificador.startswith("Positive")
    assert peptidos[0].longitud == len(peptidos[0].secuencia)


def test_admite_secuencias_multilinea(tmp_path):
    ruta = tmp_path / "multi.fasta"
    ruta.write_text(">p1\nKWKLF\nKKIEK\n>p2\nFLPIIAKLLGGLL\n")
    peptidos = leer_fasta(ruta)
    assert len(peptidos) == 2
    assert peptidos[0].secuencia == "KWKLFKKIEK"


def test_archivo_inexistente():
    with pytest.raises(ErrorFasta, match="No existe"):
        leer_fasta("no_existe.fasta")


def test_archivo_sin_cabeceras(tmp_path):
    ruta = tmp_path / "malo.fasta"
    ruta.write_text("KWKLFKKIEK\n")
    with pytest.raises(ErrorFasta, match="cabecera"):
        leer_fasta(ruta)


def test_archivo_sin_secuencias_validas(tmp_path):
    ruta = tmp_path / "vacio.fasta"
    ruta.write_text(">p1\nXXXX\n")
    with pytest.raises(ErrorFasta, match="secuencia válida"):
        leer_fasta(ruta)


# ── Modelo ───────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def clf():
    return AcpClassifier()


def test_metadatos_del_modelo(clf):
    m = clf.metadatos
    assert m["dimensiones"] == 320
    assert m["clasificador"] == "LogisticRegression"
    assert m["metricas"]["auc_roc"] == 0.8745


def test_predice_un_acp_conocido(clf):
    # Magainina 2, péptido catiónico anfipático bien caracterizado
    [p] = clf.predecir_secuencias(["GIGKFLHSAKKFGKAFVGEIMNS"])
    assert 0.0 <= p.probabilidad_acp <= 1.0
    assert p.prediccion in ("ACP", "no-ACP")
    assert p.longitud == 23


def test_probabilidades_en_rango_y_orden(clf):
    peptidos = leer_fasta(EJEMPLO)
    preds = clf.predecir_secuencias(peptidos)
    assert len(preds) == len(peptidos)
    assert all(0.0 <= p.probabilidad_acp <= 1.0 for p in preds)
    assert [p.identificador for p in preds] == [p.identificador for p in peptidos]


def test_el_umbral_cambia_la_etiqueta(clf):
    altos = clf.predecir_fasta(EJEMPLO, umbral=0.01)
    bajos = clf.predecir_fasta(EJEMPLO, umbral=0.99)
    assert sum(p.prediccion == "ACP" for p in altos) >= sum(
        p.prediccion == "ACP" for p in bajos
    )


def test_reproduce_las_metricas_del_trabajo(clf):
    """Los positivos del ejemplo deben obtener probabilidad mayor que los negativos."""
    preds = clf.predecir_fasta(EJEMPLO)
    pos = [p.probabilidad_acp for p in preds if p.identificador.startswith("Positive")]
    neg = [p.probabilidad_acp for p in preds if p.identificador.startswith("Negative")]
    assert sum(pos) / len(pos) > sum(neg) / len(neg)


# ── CLI ──────────────────────────────────────────────────────────────────────
def test_cli_genera_csv(tmp_path, capsys):
    salida = tmp_path / "res.csv"
    assert main(["-i", str(EJEMPLO), "-o", str(salida), "-q"]) == 0
    lineas = salida.read_text(encoding="utf-8").strip().splitlines()
    assert lineas[0] == "identificador,secuencia,longitud,probabilidad_acp,prediccion"
    assert len(lineas) == 9


def test_cli_rechaza_umbral_invalido(capsys):
    assert main(["-i", str(EJEMPLO), "-u", "1.5"]) == 2
    assert "umbral" in capsys.readouterr().err


def test_cli_error_de_archivo(capsys):
    assert main(["-i", "no_existe.fasta", "-q"]) == 1
    assert "Error" in capsys.readouterr().err


def test_version():
    assert __version__ == "1.0.2"
