"""BM25 peste `chunk.text` (spec 2026-09-25, „Retrieval” → „Hibrid”).

Index în memorie cu `rank_bm25`, construit o dată din toate chunk-urile din DB. Întoarce doar
id-uri: documentele se citesc prin `ChunkDocStore`, deci un chunk șters între timp nu apare.
De ce nu `hybrid_search_config` din PGVectorStore: vezi planul fazei 4, decizia 2.
"""

import re
import unicodedata

from rank_bm25 import BM25Okapi
from sqlmodel import Session, col, select

from app.knowledge_prep.docstore import Bind
from app.knowledge_prep.models import Chunk

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Cuvinte cu litere mici, fără diacritice: `Grâul` = `graul`, `ș`/`ş` = `s`, `ț`/`ţ` = `t`.

    Utilizatorii scriu des fără diacritice, iar unele texte au sedilă în loc de virgulă.
    Fără stemming: `grâu` ≠ `grâul` (partea semantică acoperă formele flexionate).
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _WORD.findall(plain)


class ChunkBM25:
    def __init__(self, chunk_ids: list[str], texts: list[str]):
        self.chunk_ids = chunk_ids
        # BM25Okapi împarte la lungimea medie: pe corpus gol ar da eroare
        self._bm25 = BM25Okapi([tokenize(t) for t in texts]) if texts else None

    @classmethod
    def from_db(cls, bind: Bind) -> ChunkBM25:
        with Session(bind) as s:
            rows = s.exec(
                select(Chunk.id, Chunk.text).order_by(
                    col(Chunk.document_id), col(Chunk.position)
                )
            ).all()
        return cls([str(i) for i, _ in rows], [t for _, t in rows])

    def search(self, query: str, k: int) -> list[str]:
        """Cele mai bune `k` chunk-uri cu scor pozitiv.

        Listă goală doar când query-ul nu are niciun token comun cu corpusul (ex. cuvinte
        absente din toate chunk-urile). Pe un corpus real, IDF-ul negativ e limitat la
        `epsilon * average_idf` (pozitiv), deci cuvintele comune tot primesc scor > 0 — nu
        „niciun cuvânt comun (semnificativ)”, ci literalmente niciun token în comun.
        """
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunk_ids[i] for i in ranked[:k] if scores[i] > 0]
