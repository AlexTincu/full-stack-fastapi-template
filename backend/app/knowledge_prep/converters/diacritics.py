"""Normalizarea diacriticelor românești, comună convertoarelor."""

# Sedilă → virgulă dedesubt; `Ġ` e cum iese `ț` din PDF-urile cu fonturi cu encoding nestandard (ex. Treben).
_NORMALIZE = str.maketrans("şţŞŢĠġ", "șțȘȚțț")


def normalize(text: str) -> str:
    return text.translate(_NORMALIZE)
