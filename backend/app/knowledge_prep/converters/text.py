"""`.txt` / `.md` → markdown: doar curățare mecanică, conținutul rămâne cum e.

Corectura (diacritice lipsă, caractere stricate) și structurarea (titluri, liste) le face `restructure.py`, la `prepare`.
"""

import re
import unicodedata
from pathlib import Path

from app.knowledge_prep.converters.diacritics import normalize

TEXT_EXTENSIONS = frozenset({".txt", ".md"})

_FRONT_MATTER = re.compile(r"---\n.*?\n---\n", re.S)


def convert_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8-sig")  # -sig: scoate BOM-ul, dacă există
    except UnicodeDecodeError as e:
        raise ValueError(f"{path.name} nu e UTF-8 ({e.reason}).") from e

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if _FRONT_MATTER.match(text):
        raise ValueError(
            f"{path.name} are deja front-matter; pare un fișier deja convertit."
        )

    text = normalize(unicodedata.normalize("NFC", text))
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")
    if not text.strip():
        raise ValueError(f"{path.name} e gol.")
    return text + "\n"
