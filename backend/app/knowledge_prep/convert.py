"""CLI + funcția `convert(...)`: `<folder>/<sursă>` → `<folder>/converted/<slug>.md` cu `status: draft`.

Rulare din backend/:
    uv run python -m app.knowledge_prep.convert "<fișier sau folder>" [--type carte] [--title ...] [--author ...] [--force]

Reguli (docs/taskuri.md, deciziile 2, 8, 10):
- Sursa nu se copiază și nu se modifică; `source_file` din front-matter e numele ei, din folderul care conține `converted/`.
- Un `.md` existent nu se suprascrie fără `--force`; unul `reviewed` nu se suprascrie niciodată.
"""

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from app.knowledge_prep.converters import CONVERTERS, Converter
from app.knowledge_prep.converters.audio import AUDIO_EXTENSIONS
from app.knowledge_prep.frontmatter import (
    DocType,
    FrontMatter,
    parse,
    render,
    slugify,
    title_from_filename,
)

logger = logging.getLogger(__name__)

DOC_TYPES: tuple[DocType, ...] = (
    "carte",
    "regim",
    "nota",
    "insight",
    "transcriere",
    "conversatie",
)


CONVERTED_DIR = "converted"


class ConversionError(Exception):
    pass


def convert(
    source: Path,
    *,
    converters: Mapping[str, Converter] = CONVERTERS,
    doc_type: DocType | None = None,
    title: str | None = None,
    author: str | None = None,
    force: bool = False,
) -> Path | None:
    """Convertește o sursă. Întoarce calea `.md` scrisă sau None dacă există deja (draft, fără `force`)."""
    converter = converters.get(source.suffix.lower())
    if converter is None:
        raise ConversionError(
            f"Niciun convertor pentru '{source.suffix}' ({source.name})."
        )

    title = title or title_from_filename(source.stem)
    slug = slugify(title)
    source_file = source.name
    dest = source.parent / CONVERTED_DIR / f"{slug}.md"

    if dest.exists():
        existing, _ = parse(dest.read_text(encoding="utf-8"))
        if existing.source_file != source_file:
            raise ConversionError(
                f"slug '{slug}' e deja folosit de {existing.source_file}; alege alt --title."
            )
        if existing.status == "reviewed":
            raise ConversionError(
                f"{dest.name} e 'reviewed'; nu se suprascrie. Trece-l pe 'draft' sau șterge-l manual."
            )
        if not force:
            return None

    body = converter(source)
    fm = FrontMatter(
        title=title,
        slug=slug,
        type=doc_type
        or ("transcriere" if source.suffix.lower() in AUDIO_EXTENSIONS else "carte"),
        author=author,
        source_file=source_file,
    )
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(render(fm, body), encoding="utf-8")
    return dest


def main(
    argv: Sequence[str] | None = None, converters: Mapping[str, Converter] = CONVERTERS
) -> int:
    p = argparse.ArgumentParser(
        prog="python -m app.knowledge_prep.convert", description=__doc__.split("\n")[0]
    )
    p.add_argument("path", type=Path, help="fișier sau folder (parcurs recursiv)")
    p.add_argument(
        "--type",
        dest="doc_type",
        choices=DOC_TYPES,
        help="implicit: transcriere pentru audio, altfel carte",
    )
    p.add_argument(
        "--title", help="doar pentru un singur fișier; implicit din numele fișierului"
    )
    p.add_argument("--author")
    p.add_argument(
        "--force", action="store_true", help="suprascrie .md-urile draft existente"
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "google_genai"):  # log-uri per request HTTP
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not args.path.exists():
        p.error(f"nu există: {args.path}")
    if args.path.is_dir() and args.title:
        p.error("--title se poate folosi doar cu un singur fișier")

    if args.path.is_file():
        sources = [args.path]
    else:
        sources = []
        for f in sorted(x for x in args.path.rglob("*") if x.is_file()):
            if CONVERTED_DIR in f.relative_to(args.path).parts[:-1]:
                continue  # rezultatele (și cache-ul) conversiilor anterioare
            if f.suffix.lower() in converters:
                sources.append(f)
            else:
                logger.info("sărit (format nesuportat): %s", f)

    failed = 0
    for src in sources:
        try:
            out = convert(
                src,
                converters=converters,
                doc_type=args.doc_type,
                title=args.title,
                author=args.author,
                force=args.force,
            )
        except Exception as e:  # un fișier stricat nu oprește restul folderului
            failed += 1
            logger.error("EROARE %s: %s", src.name, e)
            continue
        if out:
            logger.info("scris: %s", out)
        else:
            logger.info("sărit (există, folosește --force): %s", src.name)

    logger.info("%d/%d procesate fără eroare.", len(sources) - failed, len(sources))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
