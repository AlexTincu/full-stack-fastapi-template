"""BM25 peste titluri + `chunk.text` (spec 2026-09-25, „Retrieval” → „Hibrid”).

Index în memorie cu `rank_bm25`, construit o dată din toate chunk-urile din DB. Întoarce doar
id-uri: documentele se citesc prin `ChunkDocStore`, deci un chunk șters între timp nu apare.
De ce nu `hybrid_search_config` din PGVectorStore: vezi planul fazei 4, decizia 2.
"""

import re
import unicodedata
from functools import cache

import snowballstemmer
from rank_bm25 import BM25Okapi
from sqlmodel import Session, col, select

from app.knowledge_prep.chunking import embedding_text
from app.knowledge_prep.docstore import Bind
from app.knowledge_prep.models import Chunk

_WORD = re.compile(r"\w+")

# Cuvinte de legătură, fără diacritice. Fără ele, „ce”, „cine”, „pentru” din întrebare
# primesc IDF mare pe un corpus mic și aduc chunk-uri care doar le conțin (măsurat 2026-09-29).
STOPWORDS = frozenset(
    """
    a acea aceasta aceea acei acel acela acele acest acesta aceste acestea acestei acestui
    ai al ale am ar are as asa asta au azi ca care cat cata cate cati ce cea cei cel cele
    celor cine cu cum cand da daca dar de deci deja din dintr dintre doar dupa e ea ei el
    ele era este eu fara fi fie fiecare fost foarte i iar il ii imi in inca intr intre isi
    la le li lor lui m ma mai mea mei meu mi mie mult multe multi ne nici niste noi nu o
    ori pe pentru poate pot prea prin s sa sau se si sunt spre ta tale tau te ti tu un una
    unde unei unor unui va vom vor voi
    """.split()
)


@cache
def _stemmer():
    return snowballstemmer.stemmer("romanian")


def tokenize(text: str) -> list[str]:
    """Rădăcini de cuvinte, fără diacritice și fără cuvinte de legătură.

    Diacriticele se scot înainte de stemming (`ș`/`ş` = `s`, `ț`/`ţ` = `t`): utilizatorii scriu
    des fără ele, iar stemmer-ul dă așa aceeași rădăcină pentru „constipație” și „constipatia”.
    Limită: articolul după vocală rămâne (`grâul` ≠ `grâu`).
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    words = [w for w in _WORD.findall(plain) if w not in STOPWORDS]
    return _stemmer().stemWords(words)


class ChunkBM25:
    def __init__(self, chunk_ids: list[str], texts: list[str]):
        self.chunk_ids = chunk_ids
        # BM25Okapi împarte la lungimea medie: pe corpus gol ar da eroare
        self._bm25 = BM25Okapi([tokenize(t) for t in texts]) if texts else None

    @classmethod
    def from_db(cls, bind: Bind) -> ChunkBM25:
        with Session(bind) as s:
            rows = s.exec(
                select(Chunk.id, Chunk.heading_path, Chunk.text).order_by(
                    col(Chunk.document_id), col(Chunk.position)
                )
            ).all()
        # titlurile contează: „Germenii de grâu combat constipația” e în heading_path
        return cls(
            [str(i) for i, _, _ in rows],
            [embedding_text(list(path), text) for _, path, text in rows],
        )

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
