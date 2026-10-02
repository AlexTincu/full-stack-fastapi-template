"""Comparația metodelor de retrieval în Gradio: aceeași întrebare, două panouri (A | B).

Rulare din backend/:
    uv run python -m app.knowledge_prep.explorer

Metodele: `rag_simple` (rescriere + semantic + BM25 + rerank, peste `rag_simple_chunks`) și cele
4 moduri din `evaluation.MODES` (peste `knowledge_vectors`). Rezultatele găsite și de celălalt
panou sunt marcate: același chunk (doar între modurile `knowledge_prep`) sau același document
(cheia comună e numele fișierului markdown). „Acoperire” arată ce documente are indexate
fiecare metodă: o diferență între metode contează doar pe documentele pe care le au amândouă.
"""

import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from sqlalchemy import Engine, func
from sqlmodel import Session, col, select

from app.knowledge_prep import evaluation, models
from app.knowledge_prep.docstore import Bind
from app.knowledge_prep.models import Chunk, Stage
from app.knowledge_prep.retrieval import DEFAULT_K, build_retriever

RAG_SIMPLE = "rag_simple"
METHODS = [RAG_SIMPLE, *evaluation.MODES]
MAX_K = 20
MAX_TEXT_CHARS = 3000  # un părinte poate fi un document întreg
SAME_CHUNK = "★ același chunk"
SAME_DOCUMENT = "◆ același document"
ONLY_HERE = "○ doar aici"


@dataclass(frozen=True)
class Hit:
    """Un rezultat, în aceeași formă pentru toate metodele."""

    source: str  # numele fișierului markdown: cheia comună între rag_simple și knowledge_prep
    location: str  # titlurile (secțiunea) din care vine textul
    text: str
    chunk_ids: frozenset[str] = frozenset()  # gol pentru rag_simple (alt chunking)
    detail: str = ""


@dataclass
class PanelResult:
    method: str
    hits: list[Hit] = field(default_factory=list)
    note: str = ""
    seconds: float = 0.0
    error: str | None = None


def rag_simple_hits(docs: Iterable[Document]) -> list[Hit]:
    hits = []
    for d in docs:
        m = d.metadata
        headings = [m[h] for h in ("h3", "h4", "h5") if m.get(h)]
        hits.append(Hit(m.get("source", "?"), " > ".join(headings), d.page_content))
    return hits


def knowledge_hits(docs: Iterable[Document], files: Mapping[str, str]) -> list[Hit]:
    """`files`: document_id → numele fișierului markdown (`document_files`)."""
    hits = []
    for d in docs:
        m = d.metadata
        if "scope" in m:  # părinte (expand=True)
            ids = frozenset(m["matched_chunk_ids"])
            detail = f"{m['scope']}, {len(ids)}/{len(m['chunk_ids'])} chunk-uri găsite"
        else:
            ids = frozenset([m["chunk_id"]])
            detail = ""
        hits.append(
            Hit(
                files.get(m["document_id"], m["slug"]),
                " > ".join(m["heading_path"]),
                d.page_content,
                ids,
                detail,
            )
        )
    return hits


def overlap_marks(hits: Sequence[Hit], other: Sequence[Hit] | None) -> list[str]:
    """Pentru fiecare rezultat: dacă îl are și celălalt panou (chunk sau document)."""
    if other is None:
        return [""] * len(hits)
    other_chunks = frozenset().union(*(h.chunk_ids for h in other))
    other_sources = {h.source for h in other}
    marks = []
    for h in hits:
        if h.chunk_ids & other_chunks:
            marks.append(SAME_CHUNK)
        elif h.source in other_sources:
            marks.append(SAME_DOCUMENT)
        else:
            marks.append(ONLY_HERE)
    return marks


def _shorten(text: str) -> str:
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return f"{text[:MAX_TEXT_CHARS]}\n\n*… (+{len(text) - MAX_TEXT_CHARS} caractere)*"


def render(result: PanelResult, other: PanelResult | None) -> str:
    head = f"### {result.method} · {result.seconds:.1f} s"
    if result.error:
        return f"{head}\n\n**EROARE:** {result.error}"
    if not result.hits:
        return f"{head}\n\n*niciun rezultat*"
    marks = overlap_marks(
        result.hits, other.hits if other and not other.error else None
    )
    parts = [head]
    if result.note:
        parts.append(f"*{result.note}*")
    for i, (h, mark) in enumerate(zip(result.hits, marks, strict=True), 1):
        meta = " · ".join(x for x in (h.location, h.detail, mark) if x)
        parts.append(f"---\n**{i}. {h.source}**  \n{meta}\n\n{_shorten(h.text)}")
    return "\n\n".join(parts)


def document_files(bind: Bind) -> dict[str, str]:
    with Session(bind) as s:
        rows = s.exec(select(models.Document.id, models.Document.markdown_path))
        return {str(doc_id): Path(path).name for doc_id, path in rows}


def knowledge_coverage(bind: Bind) -> dict[str, tuple[str, int]]:
    """Fișier → (stadiu, nr. chunk-uri) pentru documentele din registru."""
    with Session(bind) as s:
        rows = s.exec(
            select(
                models.Document.markdown_path,
                models.Document.stage,
                func.count(col(Chunk.id)),
            )
            .join(Chunk, isouter=True)
            .group_by(col(models.Document.id))
        )
        return {Path(path).name: (Stage(stage).value, n) for path, stage, n in rows}


def coverage_rows(
    rag_simple: Mapping[str, int], knowledge: Mapping[str, tuple[str, int]]
) -> list[list[Any]]:
    """[fișier, chunk-uri rag_simple, stadiu knowledge_prep, chunk-uri knowledge_prep]; întâi cele comune."""
    rows = []
    for name in set(rag_simple) | set(knowledge):
        stage, n = knowledge.get(name, ("—", 0))
        rows.append([name, rag_simple.get(name, 0), stage, n])
    return sorted(rows, key=lambda r: (not (r[1] and r[3]), r[0].lower()))


# --- rularea reală (Postgres, OpenAI) ---


_warm: set[str] = set()  # metodele care au rulat deja o dată fără eroare


@lru_cache
def _engine() -> Engine:
    from app.core.db import engine

    return engine


@lru_cache
def _store():
    from app.knowledge_prep.vectors import PgVectorIndex

    return PgVectorIndex.from_settings().store


def _search(method: str, query: str, k: int, expand: bool) -> PanelResult:
    result = PanelResult(
        method if method == RAG_SIMPLE or not expand else f"{method} (părinți)"
    )
    start = time.perf_counter()
    try:
        if method == RAG_SIMPLE:
            # import leneș: încarcă modelele și BM25-ul din rag_simple (câteva secunde)
            from app.rag_simple.retrieval.pipeline import retrieve_with_trace

            trace = retrieve_with_trace(query)
            result.hits = rag_simple_hits(trace.final[:k])
            result.note = f"întrebarea rescrisă: {trace.rewritten_question}"
        else:
            questions, hybrid = evaluation.MODES[method]
            retriever = build_retriever(
                _store(),
                _engine(),
                k=k,
                questions=questions,
                hybrid=hybrid,
                expand=expand,
            )
            result.hits = knowledge_hits(
                retriever.invoke(query), document_files(_engine())
            )
    except Exception as e:  # afișat în panou; celălalt panou merge mai departe
        result.error = f"{type(e).__name__}: {e}"
    result.seconds = time.perf_counter() - start
    return result


def compare(
    query: str, k: int, method_a: str, expand_a: bool, method_b: str, expand_b: bool
) -> tuple[str, str]:
    query = query.strip()
    if not query:
        return "*Scrie o întrebare.*", ""
    if {method_a, method_b} <= _warm:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(_search, method_a, query, int(k), expand_a)
            fb = pool.submit(_search, method_b, query, int(k), expand_b)
            a, b = fa.result(), fb.result()
    else:
        # prima rulare a unei metode importă leneș module (ex. openai.resources.*): în paralel,
        # două fire care importă același modul pot da _DeadlockError, deci întâi secvențial
        a = _search(method_a, query, int(k), expand_a)
        b = _search(method_b, query, int(k), expand_b)
    _warm.update(r.method.split(" ")[0] for r in (a, b) if r.error is None)
    return render(a, b), render(b, a)


def coverage() -> list[list[Any]]:
    from app.rag_simple.ingestion.tracking import chunk_counts

    return coverage_rows(chunk_counts(_engine()), knowledge_coverage(_engine()))


def main() -> None:
    import gradio as gr

    def panel(label: str, default: str):
        with gr.Column(scale=1):
            method = gr.Dropdown(METHODS, value=default, label=f"Metoda {label}")
            expand = gr.Checkbox(
                False,
                label="părinți (document/secțiune) în loc de chunk-uri",
                interactive=default != RAG_SIMPLE,
            )
            out = gr.Markdown(container=True, height=700)
        method.change(
            lambda m: gr.update(interactive=m != RAG_SIMPLE, value=False),
            method,
            expand,
        )
        return method, expand, out

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])
    with gr.Blocks(title="Comparație retrieval", theme=theme) as ui:
        gr.Markdown(
            "# 🔎 Comparație retrieval\n"
            f"{SAME_CHUNK} · {SAME_DOCUMENT} · {ONLY_HERE} — față de celălalt panou"
        )
        with gr.Row():
            query = gr.Textbox(
                label="Întrebarea", placeholder="ex. ce mănânc dimineața?", scale=4
            )
            k = gr.Slider(1, MAX_K, value=DEFAULT_K, step=1, label="k", scale=1)
            button = gr.Button("Caută", variant="primary", scale=0)
        with gr.Row():
            method_a, expand_a, out_a = panel("A", RAG_SIMPLE)
            method_b, expand_b, out_b = panel("B", "multi-vector+bm25")
        with gr.Accordion(
            "Acoperire: ce documente are indexate fiecare metodă", open=False
        ):
            table = gr.Dataframe(
                headers=[
                    "fișier",
                    "rag_simple: chunk-uri",
                    "knowledge_prep: stadiu",
                    "knowledge_prep: chunk-uri",
                ],
                interactive=False,
            )
            gr.Button("Reîncarcă").click(coverage, outputs=table)

        inputs = [query, k, method_a, expand_a, method_b, expand_b]
        query.submit(compare, inputs, [out_a, out_b])
        button.click(compare, inputs, [out_a, out_b])
        ui.load(coverage, outputs=table)

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()
