"""PDF (cu strat de text) → markdown cu Gemini, pe bucăți de 8 pagini (specificația din docs/taskuri.md, taskul 1).

Textul extras cu pypdf nu ajunge în `.md`, dar are două roluri:
- unește cuvintele despărțite la capăt de rând, pe care Gemini fără thinking le copiază (`pen-tru`);
- verifică fiecare bucată: dacă din output lipsește text, bucata se reîncearcă o dată, apoi se marchează.

Bucățile gata se păstrează în `converted/.cache/<nume>/` (lângă PDF), ca o rulare întreruptă să nu plătească din nou.
"""

import io
import logging
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

from google import genai
from google.genai import types
from pypdf import PdfReader, PdfWriter

from app.knowledge_prep.converters.diacritics import normalize
from app.knowledge_prep.converters.gemini import client_and_model, generate

logger = logging.getLogger(__name__)

PROMPT = (
    "Convertește acest PDF în markdown, fidel textului original în limba română. "
    "Păstrează toate diacriticele (ă, â, î, ș, ț). "
    "Reconstituie titlurile ca headere markdown (#, ##, ###) și tabelele ca tabele markdown. "
    "Paragrafele rupte între rânduri sau pagini se unesc. "
    "Omite antetele/subsolurile repetitive și numerele de pagină. "
    "Nu rezuma, nu corecta stilul, nu adăuga nimic. Returnează doar markdown-ul."
)

PDF_EXTENSIONS = frozenset({".pdf"})
PAGES_PER_CHUNK = 8
MIN_PAGE_CHARS = 50
# În spike, output-urile fidele pierdeau ~0.3-0.4% (numere de pagină, de note), cele cu fraze omise 1-2%.
MAX_MISSING = 0.01
_SHINGLE = 8

# Numere de pagină rămase pe rând propriu: promptul le cere omise, dar uneori scapă.
_PAGE_NUMBER_LINE = re.compile(r"^\d{1,4}\n+", re.M)
_WORD = r"[^\W\d_]+"  # doar litere: intervalele `3-4` nu sunt despărțiri
_LINE_END_BREAK = re.compile(rf"({_WORD})-[ \t]*\n\s*({_WORD})")
_INLINE_HYPHEN = re.compile(rf"(?<!\w)({_WORD})-({_WORD})(?!\w)")
# Cratime reale care pot cădea la capăt de rând: `s-a`, `într-o`, `dă-mi`, `du-te`.
_CLITIC_BEFORE = {"s", "n", "m", "l", "v", "c", "într", "dintr", "printr"}
_CLITIC_AFTER = {
    "o", "a", "i", "l", "un", "am", "ai", "au", "ați", "ar", "aș",
    "mi", "ți", "și", "se", "te", "le", "ne", "vă", "lor",
}  # fmt: skip


class ScannedPdfError(ValueError):
    pass


def _key(before: str, after: str) -> tuple[str, str]:
    return normalize(before).lower(), normalize(after).lower()


def hyphen_breaks(text: str, context: str | None = None) -> set[tuple[str, str]]:
    """Perechile `xxx-⏎yyy` din textul pypdf care sunt despărțiri în silabe, nu cratime reale.

    `context` (implicit `text`) e tot textul cărții, folosit ca dovadă: dacă `xxxyyy` apare întreg,
    e despărțire; dacă `xxx-yyy` apare în interiorul unui rând, e cratimă reală.
    """
    context = normalize(text if context is None else context)
    words = {w.lower() for w in re.findall(_WORD, context)}
    real_hyphens = {_key(a, b) for a, b in _INLINE_HYPHEN.findall(context)}

    breaks = set()
    for before, after in _LINE_END_BREAK.findall(normalize(text)):
        key = _key(before, after)
        if key in real_hyphens or key[0] in _CLITIC_BEFORE:
            continue
        if key[0] + key[1] in words or key[1] not in _CLITIC_AFTER:
            breaks.add(key)
    return breaks


def join_hyphenated(markdown: str, breaks: set[tuple[str, str]]) -> str:
    return _INLINE_HYPHEN.sub(
        lambda m: m[1] + m[2] if _key(m[1], m[2]) in breaks else m[0], markdown
    )


def _letters(text: str) -> str:
    """Doar literele și cifrele, fără diacritice: spațiile false din pypdf (`natur ă`) și markdown-ul nu contează."""
    text = re.sub(r"<!--.*?-->", "", normalize(text), flags=re.S)
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(
        c for c in decomposed if c.isalnum() and not unicodedata.combining(c)
    )


def shingles(text: str) -> Counter[str]:
    """Multisetul n-gramelor de litere din `text`."""
    letters = _letters(text)
    return Counter(
        letters[i : i + _SHINGLE] for i in range(len(letters) - _SHINGLE + 1)
    )


def missing_fraction(reference: str, text: str) -> float:
    """Cât din `reference` (pypdf) lipsește din `text` (Gemini), ca multiset de n-grame de litere, independent de ordine."""
    expected = shingles(reference)
    if not expected:
        return 0.0
    return (expected - shingles(text)).total() / expected.total()


def _chunk_pdf(reader: PdfReader, start: int, end: int) -> bytes:
    writer = PdfWriter()
    for i in range(start, end):
        writer.add_page(reader.pages[i])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _convert_chunk(
    client: genai.Client,
    model: str,
    pdf: bytes,
    reference: str,
    context: str,
    pages: str,
) -> str:
    breaks = hyphen_breaks(reference, context)
    part = types.Part.from_bytes(data=pdf, mime_type="application/pdf")
    best, best_missing = None, 1.0
    for attempt in (1, 2):
        text = generate(client, model, [PROMPT, part])
        if not text or not text.strip():
            logger.warning("  paginile %s: răspuns gol (încercarea %d)", pages, attempt)
            continue
        text = _PAGE_NUMBER_LINE.sub("", normalize(text.strip()) + "\n").strip()
        text = join_hyphenated(text, breaks)
        missing = missing_fraction(reference, text)
        if missing <= MAX_MISSING:
            return text + "\n"
        logger.warning(
            "  paginile %s: lipsește %.1f%% din text (încercarea %d)",
            pages,
            missing * 100,
            attempt,
        )
        if missing < best_missing:
            best, best_missing = text, missing

    if best is None:
        return (
            f"<!-- VERIFICĂ: paginile {pages}, Gemini nu a întors nimic; mai jos e text brut din PDF -->\n\n"
            f"{reference.strip()}\n"
        )
    words = round(best_missing * len(re.findall(_WORD, reference)))
    return f"<!-- VERIFICĂ: paginile {pages}, lipsesc ~{words} cuvinte ({best_missing:.1%}) -->\n\n{best}\n"


def convert_pdf(
    path: Path,
    client: genai.Client | None = None,
    model: str | None = None,
    cache_dir: Path | None = None,
) -> str:
    reader = PdfReader(path)
    texts = [normalize(page.extract_text() or "") for page in reader.pages]
    empty = sum(len(t.strip()) < MIN_PAGE_CHARS for t in texts)
    if empty > len(texts) / 2:
        raise ScannedPdfError(
            f"{path.name}: {empty}/{len(texts)} pagini fără text. "
            "PDF fără strat de text; OCR nu e suportat."
        )

    client, model = client_and_model(client, model)
    cache_dir = cache_dir or path.parent / "converted" / ".cache" / path.stem
    context = "\n".join(texts)
    parts = []
    for start in range(0, len(texts), PAGES_PER_CHUNK):
        end = min(start + PAGES_PER_CHUNK, len(texts))
        cached = cache_dir / f"p{start + 1:04d}-{end:04d}.md"
        if cached.exists():
            parts.append(cached.read_text(encoding="utf-8"))
            continue
        logger.info(
            "%s: paginile %d-%d/%d cu %s ...",
            path.name,
            start + 1,
            end,
            len(texts),
            model,
        )
        md = _convert_chunk(
            client,
            model,
            _chunk_pdf(reader, start, end),
            reference="\n".join(texts[start:end]),
            context=context,
            pages=f"{start + 1}-{end}",
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_text(md, encoding="utf-8")
        parts.append(md)

    shutil.rmtree(cache_dir, ignore_errors=True)
    if cache_dir.parent.is_dir() and not any(cache_dir.parent.iterdir()):
        cache_dir.parent.rmdir()  # .cache/ gol
    return "\n\n".join(p.strip() for p in parts) + "\n"
