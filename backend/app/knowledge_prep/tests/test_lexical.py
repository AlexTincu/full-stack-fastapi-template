from sqlmodel import Session

from app.knowledge_prep.lexical import ChunkBM25, tokenize
from app.knowledge_prep.tests.retrieval_fakes import add_document

TEXTS = {
    "a": "Grâul încolțit dă putere organismului.",
    "b": "Ceaiul de mușețel calmează stomacul.",
    "c": "Sucul de morcovi şi ţelină pentru ficat.",  # sedilă, ca în textele vechi
    "d": "Postul negru durează o zi.",
}


def bm25() -> ChunkBM25:
    return ChunkBM25(list(TEXTS), list(TEXTS.values()))


def test_tokenize_lowercases_strips_diacritics_stopwords_and_stems():
    assert tokenize("Grâul ÎNCOLȚIT, şi ţelina!") == ["graul", "incolt", "telin"]
    assert tokenize("?! — ...") == []
    assert tokenize("ce să mănânc pentru cine") == ["mananc"]


def test_tokenize_matches_inflections_with_and_without_diacritics():
    forms = ["constipație", "constipația", "constipatie", "constipatia"]
    assert len({tuple(tokenize(f)) for f in forms}) == 1
    assert tokenize("nervoasă") == tokenize("nervoasa")


def test_search_ignores_diacritics_both_ways():
    index = bm25()
    assert index.search("graul incoltit", 3) == ["a"]
    assert index.search("țelină", 3) == ["c"]  # virgulă în query, sedilă în text
    assert index.search("MUSETEL", 3) == ["b"]


def test_search_ranks_and_limits_to_k():
    index = bm25()
    ranked = index.search("sucul postul ceaiul", 4)
    assert set(ranked) == {"b", "c", "d"}
    assert len(index.search("sucul postul ceaiul", 2)) == 2


def test_query_without_words_and_empty_corpus_return_nothing():
    assert bm25().search("?!", 3) == []
    assert bm25().search("necunoscut", 3) == []
    assert ChunkBM25([], []).search("grâu", 3) == []


def test_from_db_indexes_all_chunks(session: Session):
    chunks = add_document(
        session,
        "grau",
        [(["G", str(i)], text) for i, text in enumerate(TEXTS.values())],
    )
    index = ChunkBM25.from_db(session.get_bind())
    assert index.search("morcovi", 3) == [str(chunks[2].id)]


def test_from_db_indexes_heading_paths_too(session: Session):
    a, *_ = add_document(
        session,
        "grau",
        [
            (["Grâul", "Combate constipația"], "Se ia cu apă."),
            (["Grâul", "B"], "Altceva."),
            (["Grâul", "C"], "Încă ceva."),  # cu 2 chunk-uri, IDF-ul ar fi 0
        ],
    )
    index = ChunkBM25.from_db(session.get_bind())
    assert index.search("constipatie", 3) == [str(a.id)]
