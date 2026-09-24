"""Registrul convertoarelor: extensie → funcție care primește fișierul din `files/` și întoarce markdown (fără front-matter).

Fiecare convertor (pdf.py, audio.py, text.py, whatsapp.py) se înregistrează aici când e implementat.
"""

from collections.abc import Callable
from pathlib import Path

Converter = Callable[[Path], str]

from app.knowledge_prep.converters.audio import (  # noqa: E402
    AUDIO_EXTENSIONS,
    transcribe_audio,
)
from app.knowledge_prep.converters.pdf import PDF_EXTENSIONS, convert_pdf  # noqa: E402

CONVERTERS: dict[str, Converter] = {
    **dict.fromkeys(AUDIO_EXTENSIONS, transcribe_audio),
    **dict.fromkeys(PDF_EXTENSIONS, convert_pdf),
}
