"""CLI-ul pipeline-ului de cunoștințe: register → prepare → review → index → validate, plus status și search.

Rulare din backend/:
    uv run python -m app.knowledge_prep <comandă> <țintă> [opțiuni]

<țintă> = id, slug sau cale (a originalului ori a markdown-ului).
Spec: docs/superpowers/specs/2026-09-25-knowledge-pipeline-design.md
"""

import argparse
import logging
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from langchain_core.vectorstores import VectorStore
from sqlmodel import Session, col, select

from app.knowledge_prep import registry
from app.knowledge_prep.convert import CONVERTED_DIR
from app.knowledge_prep.converters import CONVERTERS, Converter
from app.knowledge_prep.index import check_indexable, index_document
from app.knowledge_prep.models import Document, DocumentType, Stage
from app.knowledge_prep.questions import QuestionGenerator, generate_questions
from app.knowledge_prep.registry import AlreadyRegistered, RegistryError
from app.knowledge_prep.restructure import Restructurer, restructure
from app.knowledge_prep.retrieval import DEFAULT_K, build_retriever
from app.knowledge_prep.vectors import PgVectorIndex, VectorIndex

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 200


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"trebuie să fie cel puțin 1 (primit {n})")
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.knowledge_prep", description=__doc__.split("\n")[0]
    )
    sub = p.add_subparsers(dest="command", required=True)

    reg = sub.add_parser(
        "register", help="înregistrează un original (sau un markdown structurat)"
    )
    reg.add_argument("path", type=Path, help="fișier sau folder (parcurs recursiv)")
    reg.add_argument(
        "--structured",
        action="store_true",
        help="markdown deja structurat → 'structured' (fără original de convertit)",
    )
    reg.add_argument(
        "--type",
        dest="doc_type",
        type=DocumentType,
        choices=list(DocumentType),
        help="implicit: transcriere pentru audio, altfel carte",
    )
    reg.add_argument(
        "--title", help="doar pentru un singur fișier; implicit din numele fișierului"
    )
    reg.add_argument("--author")
    reg.add_argument(
        "--split",
        action=argparse.BooleanOptionalAction,
        help="implicit: fără split pentru regim, nota, insight",
    )

    prep = sub.add_parser(
        "prepare", help="conversia (+ restructurarea) originalului → 'structured'"
    )
    prep.add_argument("target", help="id, slug, cale sau folder")
    prep.add_argument(
        "--force",
        action="store_true",
        help="suprascrie markdown-ul existent; readuce un document 'reviewed'+ în 'structured'",
    )
    prep.add_argument(
        "--restructure",
        action=argparse.BooleanOptionalAction,
        help="corectură + structurare cu Gemini; implicit da, în afară de PDF",
    )

    sub.add_parser("review", help="→ 'reviewed', salvează hash-ul").add_argument(
        "target"
    )

    idx = sub.add_parser("index", help="chunking + întrebări + embeddings → 'indexed'")
    idx.add_argument("target", help="id, slug sau cale")
    idx.add_argument(
        "--questions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="--no-questions: doar vectorii 'content' (baseline)",
    )
    idx.add_argument(
        "--force",
        action="store_true",
        help="reindexează un document 'validated' (întrebările editate manual se pierd)",
    )

    sub.add_parser("validate", help="→ 'validated'").add_argument("target")
    sub.add_parser("status", help="tabel cu documentele").add_argument(
        "target", nargs="?"
    )

    srch = sub.add_parser(
        "search", help="caută chunk-uri pentru o întrebare (verificare manuală)"
    )
    srch.add_argument("query", help="întrebarea, între ghilimele")
    srch.add_argument(
        "-k", type=_positive_int, default=DEFAULT_K, help="câte chunk-uri (implicit 5)"
    )
    srch.add_argument(
        "--questions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="--no-questions: doar vectorii 'content' (baseline)",
    )
    srch.add_argument(
        "--hybrid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="--no-hybrid: fără BM25",
    )
    return p


def _run_each[T](
    session: Session, items: Iterable[T], action: Callable[[T], str], name: str
) -> int:
    """Rulează `action` pe fiecare element; un eșec nu oprește restul. Întoarce nr. de eșecuri."""
    failed = 0
    for item in items:
        try:
            logger.info("%s", action(item))
        except AlreadyRegistered as e:
            logger.info("sărit %s: %s", name.format(item), e)
        except Exception as e:
            session.rollback()
            failed += 1
            logger.error("EROARE %s: %s", name.format(item), e)
    return failed


def _files_under(folder: Path, structured: bool, converters: Mapping[str, Converter]):
    for f in sorted(x for x in folder.rglob("*") if x.is_file()):
        if structured:
            if f.suffix.lower() == ".md":
                yield f
        elif CONVERTED_DIR in f.relative_to(folder).parts[:-1]:
            continue  # rezultatele conversiilor anterioare
        elif f.suffix.lower() in converters:
            yield f
        else:
            logger.info("sărit (format nesuportat): %s", f)


def _register(
    args: argparse.Namespace,
    session: Session,
    converters: Mapping[str, Converter],
    parser: argparse.ArgumentParser,
) -> int:
    if not args.path.exists():
        parser.error(f"nu există: {args.path}")
    if args.path.is_dir() and args.title:
        parser.error("--title se poate folosi doar cu un singur fișier")
    files = (
        [args.path]
        if args.path.is_file()
        else list(_files_under(args.path, args.structured, converters))
    )
    options = {
        "doc_type": args.doc_type,
        "title": args.title,
        "author": args.author,
        "split": args.split,
    }

    def register_one(f: Path) -> str:
        if args.structured:
            doc = registry.register_structured(session, f, **options)
        else:
            doc = registry.register(session, f, converters=converters, **options)
        return f"înregistrat: {doc.slug} ({doc.stage})"

    failed = _run_each(session, files, register_one, "{}")
    logger.info("%d/%d fără eroare.", len(files) - failed, len(files))
    return 1 if failed else 0


def _prepare(
    args: argparse.Namespace,
    session: Session,
    converters: Mapping[str, Converter],
    restructurer: Restructurer,
) -> int:
    folder = Path(args.target)
    if folder.is_dir():
        prefix = str(folder.resolve()) + os.sep
        docs = list(
            session.exec(
                select(Document)
                .where(col(Document.original_path).startswith(prefix))
                .order_by(col(Document.original_path))
            )
        )
        if not args.force:
            for d in docs:
                if d.stage != Stage.uploaded:
                    logger.info("sărit (%s; folosește --force): %s", d.stage, d.slug)
            docs = [d for d in docs if d.stage == Stage.uploaded]
    else:
        try:
            docs = [registry.resolve(session, args.target)]
        except RegistryError as e:
            logger.error("EROARE: %s", e)
            return 1

    def prepare_one(doc: Document) -> str:
        out = registry.prepare(
            session,
            doc,
            force=args.force,
            restructure=args.restructure,
            converters=converters,
            restructurer=restructurer,
        )
        return f"scris: {out}"

    failed = _run_each(session, docs, prepare_one, "{0.slug}")
    return 1 if failed else 0


def _single(
    step: Callable[[Session, Document], Document],
    args: argparse.Namespace,
    session: Session,
) -> int:
    try:
        doc = step(session, registry.resolve(session, args.target))
    except RegistryError as e:
        logger.error("EROARE: %s", e)
        return 1
    logger.info("%s: %s", doc.slug, doc.stage)
    return 0


def _index(
    args: argparse.Namespace,
    session: Session,
    vectors: VectorIndex | None,
    questioner: QuestionGenerator,
) -> int:
    try:
        doc = registry.resolve(session, args.target)
        check_indexable(
            doc, force=args.force
        )  # refuză înainte de a construi vectorii reali
        report = index_document(
            session,
            doc,
            vectors=vectors if vectors is not None else PgVectorIndex.from_settings(),
            questioner=questioner if args.questions else None,
            force=args.force,
        )
    except RegistryError as e:  # regulă încălcată (mesaj clar, fără traceback)
        logger.error("EROARE %s: %s", args.target, e)
        return 1
    except Exception as e:  # pas neașteptat (last_error e deja scris); traceback util
        logger.exception("EROARE %s: %s", args.target, e)
        return 1
    logger.info(
        "%s: %s (%d chunk-uri, %d întrebări, %d avertismente)",
        doc.slug,
        doc.stage,
        report.chunks,
        report.questions,
        len(report.warnings),
    )
    return 0


def _search(
    args: argparse.Namespace, session: Session, vectorstore: VectorStore | None
) -> int:
    query = args.query.strip()
    if not query:
        logger.error("EROARE: întrebarea e goală")
        return 1
    try:
        retriever = build_retriever(
            vectorstore
            if vectorstore is not None
            else PgVectorIndex.from_settings().store,
            session.get_bind(),
            k=args.k,
            questions=args.questions,
            hybrid=args.hybrid,
        )
        docs = retriever.invoke(query)
    except Exception as e:
        logger.exception("EROARE: %s", e)
        return 1
    if not docs:
        sys.stdout.write("niciun rezultat\n")
        return 0
    for i, d in enumerate(docs, 1):
        m = d.metadata
        snippet = " ".join(d.page_content.split())[:SNIPPET_CHARS]
        sys.stdout.write(
            f"{i}. {m['slug']} — {' > '.join(m['heading_path'])}\n   {snippet}\n"
        )
    return 0


def _status(args: argparse.Namespace, session: Session) -> int:
    if args.target:
        try:
            docs = [registry.resolve(session, args.target)]
        except RegistryError as e:
            logger.error("EROARE: %s", e)
            return 1
    else:
        docs = list(session.exec(select(Document).order_by(col(Document.slug))))
    rows = [("slug", "tip", "stadiu", "avertismente", "eroare")] + [
        (
            d.slug,
            d.type.value,
            d.stage.value,
            ", ".join(registry.hash_warnings(d)),
            d.last_error or "",
        )
        for d in docs
    ]
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        line = "  ".join(cell.ljust(w) for cell, w in zip(r, widths, strict=True))
        sys.stdout.write(line.rstrip() + "\n")
    return 0


def _dispatch(
    args: argparse.Namespace,
    session: Session,
    converters: Mapping[str, Converter],
    restructurer: Restructurer,
    vectors: VectorIndex | None,
    questioner: QuestionGenerator,
    vectorstore: VectorStore | None,
    parser: argparse.ArgumentParser,
) -> int:
    match args.command:
        case "register":
            return _register(args, session, converters, parser)
        case "prepare":
            return _prepare(args, session, converters, restructurer)
        case "review":
            return _single(registry.review, args, session)
        case "index":
            return _index(args, session, vectors, questioner)
        case "validate":
            return _single(registry.validate, args, session)
        case "search":
            return _search(args, session, vectorstore)
        case _:
            return _status(args, session)


def main(
    argv: Sequence[str] | None = None,
    *,
    session: Session | None = None,
    converters: Mapping[str, Converter] = CONVERTERS,
    restructurer: Restructurer = restructure,
    vectors: VectorIndex | None = None,
    questioner: QuestionGenerator = generate_questions,
    vectorstore: VectorStore | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "google_genai", "openai"):  # log-uri per request HTTP
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if session is not None:
        return _dispatch(
            args,
            session,
            converters,
            restructurer,
            vectors,
            questioner,
            vectorstore,
            parser,
        )
    from app.core.db import engine  # doar la rulare reală: cere .env + Postgres

    with Session(engine) as s:
        return _dispatch(
            args, s, converters, restructurer, vectors, questioner, vectorstore, parser
        )


if __name__ == "__main__":
    sys.exit(main())
