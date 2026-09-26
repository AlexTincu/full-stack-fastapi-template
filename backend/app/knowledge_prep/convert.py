"""Conversia unui original în corp markdown (fără front-matter), prin convertorul extensiei lui.

Stadiile, căile și protecțiile (ce se poate suprascrie) stau în `registry.py`;
CLI-ul e `python -m app.knowledge_prep`.
"""

from collections.abc import Mapping
from pathlib import Path

from app.knowledge_prep.converters import CONVERTERS, Converter

CONVERTED_DIR = "converted"


class ConversionError(Exception):
    pass


def convert(source: Path, *, converters: Mapping[str, Converter] = CONVERTERS) -> str:
    converter = converters.get(source.suffix.lower())
    if converter is None:
        raise ConversionError(
            f"Niciun convertor pentru '{source.suffix}' ({source.name})."
        )
    return converter(source)
