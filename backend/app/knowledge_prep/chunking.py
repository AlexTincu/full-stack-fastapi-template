"""Împărțirea markdown-ului revizuit în chunk-uri (spec 2026-09-25, „Indexarea", pasul 2).

`split=false`: tot corpul e un singur chunk, cu `heading_path = [titlul documentului]`.
`split=true`: o secțiune la fiecare titlu `#`-`####` (headerele rămân în text); secțiunile mici
se lipesc de următoarea din același părinte, cele mari se taie cu `RecursiveCharacterTextSplitter`.

Împărțirea pe titluri e proprie, nu `MarkdownHeaderTextSplitter`: acela scoate rândurile goale
și lipește liniile cu "  \\n", deci textul din DB n-ar mai fi cel din markdown, iar secțiunile
mari nu s-ar mai putea tăia la granițe de paragraf.
"""

import re
from dataclasses import dataclass
from functools import cache

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

MIN_CHUNK_TOKENS = 80
MAX_CHUNK_TOKENS = 1000
CHUNK_OVERLAP_TOKENS = 150
UNSPLIT_WARN_TOKENS = 8000  # limita lui text-embedding-3-large e 8191
ENCODING = "cl100k_base"  # tokenizer-ul lui text-embedding-3-large

_HEADER = re.compile(r"^(#{1,4})[ \t]+(.+?)[ \t#]*$")
_FENCE = re.compile(r"^[ \t]*(```|~~~)")
_VERIFY = re.compile(r"[ \t]*<!--\s*VERIFICĂ.*?-->[ \t]*", re.S)


@dataclass
class ChunkDraft:
    heading_path: list[str]
    text: str


@cache
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def embedding_text(heading_path: list[str], text: str) -> str:
    """Textul embedded pentru vectorul `content`: calea de titluri, rând gol, chunk-ul."""
    return f"{' > '.join(heading_path)}\n\n{text}"


def _strip_verify_markers(body: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", _VERIFY.sub("", body))


def _split_sections(body: str) -> list[ChunkDraft]:
    """O secțiune la fiecare titlu `#`-`####` din afara blocurilor de cod; textul rămâne neatins."""
    sections: list[ChunkDraft] = []
    path: list[tuple[int, str]] = []
    lines: list[str] = []
    in_fence = False

    def flush() -> None:
        text = "\n".join(lines).strip()
        if text:
            sections.append(ChunkDraft([h for _, h in path], text))
        lines.clear()

    for line in body.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        header = None if in_fence else _HEADER.match(line)
        if header:
            flush()
            level = len(header[1])
            path = [(lvl, h) for lvl, h in path if lvl < level] + [
                (level, header[2].strip())
            ]
        lines.append(line)
    flush()
    return sections


def _headers_only(text: str) -> bool:
    return all(_HEADER.match(line) for line in text.splitlines() if line.strip())


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    prefix: list[str] = []
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        prefix.append(x)
    return prefix


def _merge_small(sections: list[ChunkDraft]) -> list[ChunkDraft]:
    """Sub MIN_CHUNK_TOKENS, o secțiune se lipește de următoarea dacă aceea e soră sau descendentă."""
    merged: list[ChunkDraft] = []
    pending: ChunkDraft | None = None
    for section in sections:
        if pending is not None:
            parent = pending.heading_path[:-1]
            if section.heading_path[: len(parent)] == parent:
                path = (
                    section.heading_path
                    if _headers_only(pending.text)
                    else _common_prefix(pending.heading_path, section.heading_path)
                )
                section = ChunkDraft(path, f"{pending.text}\n\n{section.text}")
            else:
                merged.append(pending)
            pending = None
        if count_tokens(section.text) < MIN_CHUNK_TOKENS:
            pending = section
        else:
            merged.append(section)
    if pending is not None:
        merged.append(pending)
    return merged


def _split_large(sections: list[ChunkDraft]) -> list[ChunkDraft]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING,
        chunk_size=MAX_CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )
    out: list[ChunkDraft] = []
    for section in sections:
        if count_tokens(section.text) <= MAX_CHUNK_TOKENS:
            out.append(section)
        else:
            out.extend(
                ChunkDraft(list(section.heading_path), piece)
                for piece in splitter.split_text(section.text)
            )
    return out


def chunk_markdown(
    body: str, title: str, split: bool
) -> tuple[list[ChunkDraft], list[str]]:
    """Corpul markdown (fără front-matter) → (chunk-uri în ordine, avertismente)."""
    text = _strip_verify_markers(body).strip()
    if not text:
        raise ValueError("markdown gol: nimic de indexat")
    if not split:
        warnings = []
        tokens = count_tokens(text)
        if tokens > UNSPLIT_WARN_TOKENS:
            warnings.append(
                f"documentul întreg are {tokens} tokeni (> {UNSPLIT_WARN_TOKENS}); "
                "vectorul `content` va fi o medie pe bucăți (imprecis), întrebările nu sunt afectate"
            )
        return [ChunkDraft([title], text)], warnings
    chunks = _split_large(_merge_small(_split_sections(text)))
    for chunk in chunks:
        if not chunk.heading_path:
            chunk.heading_path = [title]
    return chunks, []
