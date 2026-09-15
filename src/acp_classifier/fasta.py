"""Lectura y validación de archivos FASTA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

AMINOACIDOS = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class Peptido:
    """Una secuencia peptídica leída de un archivo FASTA."""

    identificador: str
    secuencia: str

    @property
    def longitud(self) -> int:
        return len(self.secuencia)


class ErrorFasta(ValueError):
    """El archivo FASTA no pudo interpretarse."""


def limpiar(secuencia: str) -> str:
    """Deja únicamente los 20 aminoácidos estándar, en mayúsculas.

    Los residuos no estándar (X, B, Z, U, O y cualquier otro carácter) se descartan.
    Es el mismo preprocesamiento aplicado durante el entrenamiento del modelo.
    """
    return "".join(aa for aa in secuencia.upper() if aa in AMINOACIDOS)


def leer_fasta(ruta: str | Path, longitud_minima: int = 2) -> list[Peptido]:
    """Lee un archivo FASTA y devuelve sus secuencias.

    Admite registros repartidos en varias líneas. Las secuencias que quedan por
    debajo de `longitud_minima` tras la limpieza se descartan con un aviso.

    Raises
    ------
    ErrorFasta
        Si el archivo no existe, está vacío o no contiene ninguna cabecera '>'.
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        raise ErrorFasta(f"No existe el archivo: {ruta}")

    peptidos: list[Peptido] = []
    descartados: list[str] = []
    identificador: str | None = None
    partes: list[str] = []

    def cerrar() -> None:
        if identificador is None:
            return
        limpia = limpiar("".join(partes))
        if len(limpia) >= longitud_minima:
            peptidos.append(Peptido(identificador, limpia))
        else:
            descartados.append(identificador)

    with ruta.open(encoding="utf-8", errors="replace") as fh:
        for linea in fh:
            linea = linea.strip()
            if not linea:
                continue
            if linea.startswith(">"):
                cerrar()
                identificador = linea[1:].strip() or f"secuencia_{len(peptidos) + 1}"
                partes = []
            else:
                partes.append(linea)
    cerrar()

    if identificador is None:
        raise ErrorFasta(
            f"{ruta} no contiene ninguna cabecera FASTA ('>'). "
            "Verifique que el archivo esté en formato FASTA."
        )
    if not peptidos:
        raise ErrorFasta(
            f"{ruta} no contiene ninguna secuencia válida de al menos "
            f"{longitud_minima} aminoácidos estándar."
        )
    if descartados:
        print(
            f"  Aviso: {len(descartados)} secuencia(s) descartada(s) por quedar "
            f"con menos de {longitud_minima} aminoácidos estándar: "
            + ", ".join(descartados[:5])
            + (" ..." if len(descartados) > 5 else "")
        )

    return peptidos
