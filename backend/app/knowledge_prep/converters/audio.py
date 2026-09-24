"""Audio → markdown: transcriere verbatim cu Gemini (spike-ul din docs/taskuri.md, taskul 1).

Înregistrările au maximum ~20 min, deci fișierul se trimite întreg, fără segmentare.
"""

import logging
from pathlib import Path

from google import genai

from app.knowledge_prep.converters.gemini import client_and_model, generate

logger = logging.getLogger(__name__)

PROMPT = (
    "Transcrie exact acest fisier audio in limba romana, "
    "cuvant cu cuvant, fara sa rezumi sau sa corectezi stilul vorbirii."
)

# Gemini deduce tipul din extensie și nu recunoaște .opus; Opus e oricum în container Ogg.
MIME_TYPES = {
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".wma": "audio/x-ms-wma",
}
AUDIO_EXTENSIONS = frozenset(MIME_TYPES)


def transcribe_audio(
    path: Path, client: genai.Client | None = None, model: str | None = None
) -> str:
    client, model = client_and_model(client, model)
    logger.info("transcriu %s cu %s ...", path.name, model)
    # Fișier deschis, nu Path: dat o cale, SDK-ul pune numele brut într-un
    # header HTTP (X-Goog-Upload-File-Name), care trebuie să fie ASCII —
    # pică pe nume cu diacritice (docs/taskuri.md).
    with open(path, "rb") as f:
        uploaded = client.files.upload(
            file=f, config={"mime_type": MIME_TYPES[path.suffix.lower()]}
        )
    try:
        text = generate(client, model, [PROMPT, uploaded])
    finally:
        if uploaded.name:
            client.files.delete(name=uploaded.name)

    if not text or not text.strip():
        raise ValueError(f"Gemini a întors un răspuns gol pentru {path.name}.")
    return text.strip() + "\n"
