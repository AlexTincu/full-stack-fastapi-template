"""Front-matter-ul minimal al documentelor din `converted/` (decizia 2 din docs/taskuri.md)."""

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Literal

import yaml

DocType = Literal["carte", "regim", "nota", "insight", "transcriere", "conversatie"]
Status = Literal["draft", "reviewed"]

_DELIMITER = "---\n"


@dataclass
class FrontMatter:
    title: str
    slug: str
    type: DocType
    author: str | None
    source_file: (
        str  # numele sursei, din folderul care conține `converted/`, ex. "carte.pdf"
    )
    status: Status = "draft"


def slugify(text: str) -> str:
    """ASCII, lowercase, cuvinte legate cu '-'. NFKD separă diacriticele (ș/ş, ț/ţ, ă, â, î) de literă."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


def title_from_filename(stem: str) -> str:
    return re.sub(r"(?<=\S)[_-](?=\S)|_", " ", stem).strip()


def render(fm: FrontMatter, body: str) -> str:
    meta = yaml.safe_dump(asdict(fm), allow_unicode=True, sort_keys=False)
    return f"{_DELIMITER}{meta}{_DELIMITER}\n{body}"


def parse(text: str) -> tuple[FrontMatter, str]:
    if not text.startswith(_DELIMITER):
        raise ValueError("Fișierul nu începe cu front-matter ('---').")
    meta, sep, body = text[len(_DELIMITER) :].partition(_DELIMITER)
    if not sep:
        raise ValueError("Front-matter neînchis (lipsește al doilea '---').")
    return FrontMatter(**yaml.safe_load(meta)), body.removeprefix("\n")
