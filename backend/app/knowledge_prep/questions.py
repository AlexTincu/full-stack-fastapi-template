"""Întrebările la care răspunde un chunk, generate cu Gemini (spec 2026-09-25, „Indexarea”, pasul 3).

Fiecare întrebare devine un vector-copil (`kind=question`) al chunk-ului: o întrebare a
utilizatorului formulată altfel decât textul găsește totuși chunk-ul (multi-vector retrieval).
"""

import logging
from collections.abc import Callable

from google import genai
from pydantic import BaseModel, Field

from app.knowledge_prep.converters.gemini import client_and_model, generate_structured

logger = logging.getLogger(__name__)

# (titlul documentului, heading_path, textul chunk-ului, e documentul întreg?) → întrebări.
# `index` îl primește injectat, ca testele să nu cheme Gemini.
QuestionGenerator = Callable[[str, list[str], str, bool], list[str]]

MAX_QUESTIONS = 5

PROMPT = """Primești un fragment dintr-un text în limba română din corpusul „Calea de Autovindecare” (cărți, regimuri alimentare, note și transcrieri despre vindecarea naturală).

Scrie între 1 și 5 întrebări la care fragmentul răspunde complet. Reguli:
1. Întrebările sunt în limba română, formulate cum ar întreba un om care caută informația, nu copiate din text.
2. Fiecare întrebare se înțelege singură, fără context: numește explicit subiectul (ex. „grâul încolțit”, „regimul pentru anemie”), niciodată „acesta”, „textul”, „autorul” sau „fragmentul”.
3. Răspunsul complet la fiecare întrebare se găsește în fragment. Nu pune întrebări la care fragmentul răspunde doar parțial.
4. Întrebările sunt diferite între ele și acoperă ideile principale. Mai bine mai puține și bune decât multe și slabe.
5. Nu adăuga informații care nu sunt în fragment.
"""


class GeneratedQuestions(BaseModel):
    questions: list[str] = Field(
        description="1-5 întrebări în limba română, la care fragmentul răspunde complet"
    )


def _message(
    title: str, heading_path: list[str], text: str, whole_document: bool
) -> str:
    scope = (
        "Fragmentul e documentul întreg: întrebările trebuie să acopere tot documentul, "
        "nu doar începutul.\n"
        if whole_document
        else ""
    )
    return (
        f"Documentul: „{title}”\nSecțiunea: {' > '.join(heading_path)}\n{scope}"
        f"\n<text>\n{text}\n</text>"
    )


def _clean(questions: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in questions:
        q = " ".join(q.split())
        if q and q.casefold() not in seen:
            seen.add(q.casefold())
            out.append(q)
    return out[:MAX_QUESTIONS]


def generate_questions(
    title: str,
    heading_path: list[str],
    text: str,
    whole_document: bool,
    client: genai.Client | None = None,
    model: str | None = None,
) -> list[str]:
    client, model = client_and_model(client, model)
    reply = generate_structured(
        client,
        model,
        [PROMPT, _message(title, heading_path, text, whole_document)],
        GeneratedQuestions,
    )
    if reply is None:
        logger.warning(
            "  Gemini nu a întors întrebări valide pentru „%s”",
            " > ".join(heading_path),
        )
        return []
    return _clean(reply.questions)
