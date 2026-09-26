"""Front-matter-ul documentelor din `converted/`: doar pentru lizibilitate; stadiul stă în registru (spec 2026-09-25, decizia 2)."""

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

import yaml

_DELIMITER = "---\n"


@dataclass
class FrontMatter:
    title: str
    slug: str
    type: str
    author: str | None
    # numele originalului, din folderul care conține `converted/`, ex. "carte.pdf"
    source_file: str | None = None


def slugify(text: str) -> str:
    """ASCII, lowercase, cuvinte legate cu '-'. NFKD separă diacriticele (ș/ş, ț/ţ, ă, â, î) de literă."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


def title_from_filename(stem: str) -> str:
    return re.sub(r"(?<=\S)[_-](?=\S)|_", " ", stem).strip()


def render(fm: FrontMatter, body: str) -> str:
    meta = yaml.safe_dump(asdict(fm), allow_unicode=True, sort_keys=False)
    return f"{_DELIMITER}{meta}{_DELIMITER}\n{body}"


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """(metadate, corp). Fără front-matter: ({}, text). Cheile necunoscute (ex. vechiul `status`) rămân în dict."""
    if not text.startswith(_DELIMITER):
        return {}, text
    meta, sep, body = text[len(_DELIMITER) :].partition(_DELIMITER)
    if not sep:
        raise ValueError("Front-matter neînchis (lipsește al doilea '---').")
    data = yaml.safe_load(meta) or {}
    if not isinstance(data, dict):
        raise ValueError("Front-matter-ul nu e un mapping YAML.")
    return data, body.removeprefix("\n")


def strip_frontmatter(text: str) -> str:
    return split_frontmatter(text)[1]
