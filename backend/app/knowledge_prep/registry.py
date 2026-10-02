"""Registrul documentelor: tranzițiile de stadiu și regulile lor.

Spec: docs/superpowers/specs/2026-09-25-knowledge-pipeline-design.md („Stadii”).
Funcțiile primesc un `Session`; CLI-ul (`__main__.py`) și, mai târziu, REST-ul le cheamă la fel.
Regulile încălcate ridică `RegistryError` fără să atingă registrul; doar execuția unui pas
(conversie, hash etc.) scrie `last_error` când eșuează (`record_step`).
"""

import hashlib
import logging
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from sqlmodel import Session, col, or_, select

from app.knowledge_prep.convert import CONVERTED_DIR, convert
from app.knowledge_prep.converters import CONVERTERS, Converter
from app.knowledge_prep.converters.audio import AUDIO_EXTENSIONS
from app.knowledge_prep.converters.pdf import PDF_EXTENSIONS
from app.knowledge_prep.frontmatter import (
    FrontMatter,
    render,
    slugify,
    split_frontmatter,
    strip_frontmatter,
    title_from_filename,
)
from app.knowledge_prep.models import Document, DocumentType, Stage
from app.knowledge_prep.restructure import Restructurer
from app.knowledge_prep.restructure import restructure as restructure_markdown
from app.models import get_datetime_utc

logger = logging.getLogger(__name__)

UNSPLIT_TYPES = {DocumentType.regim, DocumentType.nota, DocumentType.insight}


class RegistryError(Exception):
    pass


class AlreadyRegistered(RegistryError):
    pass


def body_hash(path: Path) -> str:
    """sha256 pe corpul markdown-ului, fără front-matter."""
    try:
        body = strip_frontmatter(path.read_text(encoding="utf-8"))
    except ValueError as e:  # front-matter stricat la editare în IDE
        raise RegistryError(f"{path.name}: {e}")
    return hashlib.sha256(body.encode()).hexdigest()


def default_type(path: Path) -> DocumentType:
    if path.suffix.lower() in AUDIO_EXTENSIONS:
        return DocumentType.transcriere
    return DocumentType.carte


def default_split(doc_type: DocumentType) -> bool:
    return doc_type not in UNSPLIT_TYPES


def _same_file(session: Session, doc: Document) -> Document | None:
    """Rândul aceluiași fișier, oricum ar fi fost înregistrat (original sau markdown structurat)."""
    if doc.stage == Stage.uploaded:
        paths = [doc.original_path]  # markdown-ul e doar planificat, nu există încă
    else:
        paths = [p for p in (doc.original_path, doc.markdown_path) if p]
    found = session.exec(
        select(Document).where(
            or_(
                col(Document.original_path).in_(paths),
                col(Document.markdown_path).in_(paths),
            )
        )
    ).all()
    if len(found) > 1:
        raise RegistryError(
            f"{doc.original_path or doc.markdown_path} apare în mai multe documente "
            f"({', '.join(d.slug for d in found)}); șterge-le pe cele greșite."
        )
    return found[0] if found else None


def _add(session: Session, doc: Document) -> Document:
    """Inserează documentul sau, dacă fișierul e deja înregistrat, îl actualizează.

    Reînregistrarea cu alte opțiuni lasă rândul exact ca o înregistrare nouă cu aceleași opțiuni
    (păstrând id-ul). Cu aceleași opțiuni → `AlreadyRegistered` (se sare).
    Stadiul: dacă se schimbă markdown-ul (original ↔ structurat, alt slug) se ia de la capăt;
    altfel (tip, split, titlu, autor) scade cel mult la `reviewed`, pentru reindexare.
    """
    existing = _same_file(session, doc)
    taken = session.exec(select(Document).where(Document.slug == doc.slug)).first()
    if taken and taken is not existing:
        raise RegistryError(
            f"slug '{doc.slug}' e deja folosit de "
            f"{taken.original_path or taken.markdown_path}; alege alt --title."
        )
    if existing is None:
        session.add(doc)
    else:
        fields = (
            "slug",
            "title",
            "author",
            "type",
            "split",
            "original_path",
            "markdown_path",
        )
        changed = [f for f in fields if getattr(existing, f) != getattr(doc, f)]
        if not changed:
            raise AlreadyRegistered(
                f"deja înregistrat ca '{existing.slug}' ({existing.stage}), cu aceleași opțiuni"
            )
        logger.info(
            "reînregistrat %s: %s",
            existing.slug,
            ", ".join(
                f"{f} {getattr(existing, f)} → {getattr(doc, f)}" for f in changed
            ),
        )
        for f in changed:
            setattr(existing, f, getattr(doc, f))
        if (
            "markdown_path" in changed
        ):  # alt markdown: stadiul și hash-urile vechi nu mai contează
            existing.stage = doc.stage
            existing.reviewed_hash = None
            existing.indexed_hash = None
        elif existing.stage.rank > Stage.reviewed.rank:
            existing.stage = Stage.reviewed  # chunk-urile vechi rămân până la `index`
        existing.last_error = None
        existing.updated_at = get_datetime_utc()
        doc = existing
    session.commit()
    session.refresh(doc)
    return doc


def register(
    session: Session,
    original: Path,
    *,
    doc_type: DocumentType | None = None,
    title: str | None = None,
    author: str | None = None,
    split: bool | None = None,
    converters: Mapping[str, Converter] = CONVERTERS,
) -> Document:
    """Înregistrează un original → `uploaded`; markdown-ul va fi `<folderul lui>/converted/<slug>.md`."""
    original = original.resolve()
    if original.suffix.lower() not in converters:
        raise RegistryError(
            f"Niciun convertor pentru '{original.suffix}' ({original.name})."
        )
    title = title or title_from_filename(original.stem)
    doc_type = doc_type or default_type(original)
    slug = slugify(title)
    return _add(
        session,
        Document(
            slug=slug,
            title=title,
            author=author,
            type=doc_type,
            split=default_split(doc_type) if split is None else split,
            stage=Stage.uploaded,
            original_path=str(original),
            markdown_path=str(original.parent / CONVERTED_DIR / f"{slug}.md"),
        ),
    )


def register_structured(
    session: Session,
    markdown: Path,
    *,
    doc_type: DocumentType | None = None,
    title: str | None = None,
    author: str | None = None,
    split: bool | None = None,
) -> Document:
    """Înregistrează un markdown deja structurat → `structured`.

    Front-matter-ul (dacă există) are prioritate față de opțiuni; vechiul `status` e ignorat.
    Un fișier din `converted/` își leagă originalul din `source_file`, dacă acesta există.
    """
    markdown = markdown.resolve()
    meta, _ = split_frontmatter(markdown.read_text(encoding="utf-8"))
    title = meta.get("title") or title or title_from_filename(markdown.stem)
    raw_type = meta.get("type") or doc_type or DocumentType.carte
    try:
        final_type = DocumentType(raw_type)
    except ValueError:
        raise RegistryError(f"tip necunoscut '{raw_type}' în {markdown.name}")
    original = None
    source_file = meta.get("source_file")
    if source_file and markdown.parent.name == CONVERTED_DIR:
        candidate = markdown.parent.parent / source_file
        original = str(candidate) if candidate.is_file() else None
    return _add(
        session,
        Document(
            slug=slugify(title),
            title=title,
            author=meta.get("author") or author,
            type=final_type,
            split=default_split(final_type) if split is None else split,
            stage=Stage.structured,
            original_path=original,
            markdown_path=str(markdown),
        ),
    )


def resolve(session: Session, target: str) -> Document:
    """Ținta = id, cale (a originalului ori a markdown-ului, relativă sau absolută) sau slug."""
    doc = None
    try:
        doc = session.get(Document, uuid.UUID(target))
    except ValueError:
        pass
    if doc is None:
        path = str(Path(target).resolve())
        doc = session.exec(
            select(Document).where(
                or_(
                    col(Document.original_path) == path,
                    col(Document.markdown_path) == path,
                )
            )
        ).first()
    if doc is None:
        doc = session.exec(select(Document).where(Document.slug == target)).first()
    if doc is None:
        raise RegistryError(f"document negăsit: {target}")
    return doc


@contextmanager
def record_step(session: Session, doc: Document) -> Iterator[None]:
    """Un pas eșuat scrie `last_error` și nu schimbă stadiul; unul reușit golește `last_error`."""
    try:
        yield
    except (Exception, KeyboardInterrupt) as e:  # și oprirea manuală din notebook
        session.rollback()  # expiră obiectele: modificările făcute în pas se pierd
        doc.last_error = f"{type(e).__name__}: {str(e) or 'întrerupt'}"
        doc.updated_at = get_datetime_utc()
        session.add(doc)
        session.commit()
        session.refresh(doc)
        raise
    doc.last_error = None
    doc.updated_at = get_datetime_utc()
    session.add(doc)
    session.commit()
    session.refresh(doc)


def prepare(
    session: Session,
    doc: Document,
    *,
    force: bool = False,
    restructure: bool | None = None,
    converters: Mapping[str, Converter] = CONVERTERS,
    restructurer: Restructurer = restructure_markdown,
) -> Path:
    """Original → conversie → restructurare → `<folderul originalului>/converted/<slug>.md` → `structured`.

    Restructurarea (`restructure.py`) rulează implicit pentru tot în afară de PDF, al cărui
    prompt structurează deja; `restructure` o forțează sau o oprește.
    Un markdown existent (editat poate de om) sau un document `reviewed`+ se suprascrie
    doar cu `force`; chunk-urile deja indexate rămân în DB până la următorul `index`.
    """
    if doc.original_path is None:
        raise RegistryError(
            f"{doc.slug} nu are original (a intrat direct ca markdown); nu are ce pregăti."
        )
    md = Path(doc.markdown_path)
    if not force and (doc.stage.rank >= Stage.reviewed.rank or md.exists()):
        raise RegistryError(
            f"{doc.slug} e '{doc.stage}' și are deja {md.name}; "
            "folosește --force ca să-l suprascrii."
        )
    original = Path(doc.original_path)
    if restructure is None:
        restructure = original.suffix.lower() not in PDF_EXTENSIONS
    with record_step(session, doc):
        body = convert(original, converters=converters)
        if restructure:
            body = restructurer(body, doc.title)
        fm = FrontMatter(
            title=doc.title,
            slug=doc.slug,
            type=doc.type.value,
            author=doc.author,
            source_file=original.name,
        )
        md.parent.mkdir(exist_ok=True)
        md.write_text(render(fm, body), encoding="utf-8")
        doc.stage = Stage.structured
        doc.reviewed_hash = None
    return md


def review(session: Session, doc: Document) -> Document:
    """→ `reviewed`, cu hash-ul corpului. Fără modificări față de ultimul review, nu schimbă nimic."""
    if doc.stage == Stage.uploaded:
        raise RegistryError(f"{doc.slug} nu are încă markdown; rulează întâi prepare.")
    md = Path(doc.markdown_path)
    if not md.is_file():
        raise RegistryError(f"lipsește markdown-ul: {md}")
    current = body_hash(md)
    if doc.stage.rank >= Stage.reviewed.rank and current == doc.reviewed_hash:
        return doc  # un document indexat rămâne indexat
    with record_step(session, doc):
        doc.reviewed_hash = current
        doc.stage = Stage.reviewed
    return doc


def validate(session: Session, doc: Document) -> Document:
    if doc.stage == Stage.validated:
        return doc
    if doc.stage != Stage.indexed:
        raise RegistryError(f"{doc.slug} e '{doc.stage}'; validate cere 'indexed'.")
    with record_step(session, doc):
        doc.stage = Stage.validated
    return doc


def hash_warnings(doc: Document) -> list[str]:
    if doc.stage == Stage.uploaded:
        return []
    md = Path(doc.markdown_path)
    if not md.is_file():
        return ["markdown lipsă"]
    try:
        current = body_hash(md)
    except RegistryError as e:
        return [str(e)]
    warnings = []
    if doc.reviewed_hash and current != doc.reviewed_hash:
        warnings.append("modificat după review")
    if doc.indexed_hash and current != doc.indexed_hash:
        warnings.append("modificat după indexare")
    return warnings
