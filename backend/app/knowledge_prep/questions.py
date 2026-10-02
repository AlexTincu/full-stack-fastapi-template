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

Scrie între 1 și 5 întrebări pe care un om le-ar pune unui asistent și pentru care fragmentul e o sursă bună de răspuns. Omul de obicei nu știe ce conține corpusul: pornește de la problema lui sau de la un subiect despre care a auzit vag. Scrie, în această ordine:
1. Întrebări pornind de la problemă (1-2), dacă fragmentul spune la ce ajută ceva: formulate de cineva care NU cunoaște remediul, fără să-l numească (ex. „Ce ajută la probleme ale sistemului nervos?”, „Ce pot face pentru colesterolul mare?”).
2. O întrebare de ansamblu despre subiectul fragmentului, cu subiectul numit (ex. „Care sunt beneficiile grâului încolțit?”, „Cât grâu încolțit se consumă zilnic?”).
3. Întrebări de detaliu (0-2), doar pentru idei cu adevărat diferite de cele de mai sus. Nu împărți aceeași idee în variante (pe vârste, doze, etape, afecțiuni înrudite).

Reguli:
- Limba română, cuvinte simple, cum vorbește un om, nu copiate din text.
- Fiecare întrebare se înțelege singură: niciodată „acesta”, „textul”, „autorul” sau „fragmentul”.
- Fragmentul trebuie să conțină un răspuns relevant la întrebare, chiar dacă nu singurul posibil.
- Nu adăuga informații care nu sunt în fragment. Mai bine mai puține și bune decât multe și slabe.
- O singura intrebare poate fi de ajuns, dacă fragmentul nu are mai multe idei diferite.
"""


class GeneratedQuestions(BaseModel):
    questions: list[str] = Field(
        description="1-5 întrebări în limba română, pentru care fragmentul e o sursă bună de răspuns"
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
