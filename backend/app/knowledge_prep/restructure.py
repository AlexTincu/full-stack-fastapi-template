"""Corectarea și structurarea markdown-ului convertit cu Gemini (spec 2026-09-25, „Restructurarea”).

Corpul se taie în ferestre de ~8k tokeni, la granițe de paragraf. Fiecare fereastră primește
calea de titluri de la finalul celei anterioare, ca ierarhia să continue în tot documentul.
Rezultatul se verifică pe n-grame de litere față de fereastră (`pdf.shingles`), în ambele sensuri:
peste prag se reîncearcă o dată, apoi se păstrează încercarea mai bună, marcată `VERIFICĂ`.
"""

import logging
import re
from collections import Counter
from collections.abc import Callable

from google import genai

from app.knowledge_prep.converters.diacritics import normalize
from app.knowledge_prep.converters.gemini import client_and_model, generate
from app.knowledge_prep.converters.pdf import shingles

logger = logging.getLogger(__name__)

# (corp, titlu) → corp restructurat; `prepare` îl primește injectat, ca testele să nu cheme Gemini.
Restructurer = Callable[[str, str], str]

# Promptul din spike-ul din 2026-09-25 (calea_de_autovindecare/.claude/skills/structurare-carte/prompt.md),
# fără nota specifică unei singure cărți.
PROMPT = """\
Primești o bucată dintr-un text în limba română (carte, articol sau transcriere), obținut prin conversie automată (PDF, OCR, transcriere audio sau text copiat). Returnează ACELAȘI conținut, corectat și structurat ca markdown.

CORECTEAZĂ:
1. Diacriticele: restabilește ă, â, î, ș, ț (cu virgulă dedesubt, nu ş/ţ) peste tot unde lipsesc.
2. Caracterele stricate: simboluri de font (, , (cid:NNN), Ġ etc.), cuvinte lipite sau despărțite greșit, cratime de la capăt de rând.
3. Greșelile evidente de tipar (ex. „vitamia” → „vitamina”, „reagleaza” → „reglează”).

NU MODIFICA:
- sensul, stilul, regionalismele sau terminologia autorului; arhaismele le poți înlocui cu cuvinte cunoscute de publicul larg;
- numerele, dozele, orele, zilele, cantitățile, numele proprii, denumirile de plante și boli;
- Nu rezuma, nu adăuga explicații, nu completa informații, nu elimina nimic. Dacă e nevoie să schimbi ordinea ideilor, asigură-te că nu pierzi nimic.
Dacă nu ești sigur că e o greșeală, lasă textul original.

STRUCTUREAZĂ:
4. Un singur `#` (titlul documentului); `##` capitole/părți mari; `###` secțiuni; `####` subsecțiuni. Nu sări niveluri.
5. Titlurile existente (inclusiv cele lipite în text, ex. „Indicatii Graul incoltit este indicat in:”)
   devin headere pe rând propriu, cu textul lor original (corectat).
6. Unde nu există titlu dar subiectul se schimbă clar, poți adăuga un header scurt, descriptiv, luat din cuvintele textului. Titlurile trebuie să fie înțelese și singure: „Contraindicații ale curei cu grâu încolțit”, nu „Contraindicații”.
7. O secțiune tratează un singur subiect. Nu rupe o listă sau o rețetă care formează o unitate.
8. Listele devin liste markdown (`- ` sau `1. `). Paragrafele rupte se unesc.

SEMNALEAZĂ, NU REZOLVA:
9. Text care pare mutat, trunchiat, început la jumătatea unei fraze, duplicat sau dintr-o altă sursă:
   lasă-l pe loc și pune înainte `<!-- VERIFICĂ: <motivul> -->`.
10. Un pasaj ilizibil pe care nu-l poți reconstitui sigur: păstrează-l cu `<!-- VERIFICĂ: ilizibil -->`.

BUCĂȚI DIN TEXTE MARI:
Textul lung e tăiat mecanic în bucăți consecutive, iar mesajul îți spune a câta bucată primești.
- Bucata 1 conține începutul documentului: ea primește singurul `#`.
- La bucățile următoare primești calea de titluri de până acum. Continuă ierarhia de acolo și nu
  mai folosi `#`. Dacă bucata continuă secțiunea curentă, nu repeta headerul ei.
- Tăietura dintre bucăți e artificială. Nu pune `VERIFICĂ` doar pentru că bucata începe sau se
  termină la jumătatea unei fraze sau a unui paragraf. Nu completa și nu uni textul de dincolo de
  tăietură; lasă începutul și sfârșitul exact cum sunt.

IEȘIRE: doar markdown-ul, fără front-matter, fără ```, fără comentarii în afara marcajelor VERIFICĂ.
"""

WINDOW_CHARS = 24_000  # ~8k tokeni de text românesc
MAX_DIFF = 0.01  # lipsă sau adăugat, per fereastră

_HEADER = re.compile(r"^(#{1,4}) \S.*$", re.M)
_FENCE = re.compile(r"\A```[\w-]*\n(.*?)\n?```\Z", re.S)


def split_windows(text: str, max_chars: int | None = None) -> list[str]:
    """Paragrafe consecutive până la `max_chars`; un paragraf mai lung rămâne singur."""
    max_chars = max_chars or WINDOW_CHARS
    windows: list[str] = []
    current = ""
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        if current and len(current) + 2 + len(para) > max_chars:
            windows.append(current)
            current = ""
        current = f"{current}\n\n{para}" if current else para
    if current:
        windows.append(current)
    return windows


def heading_path(markdown: str) -> str:
    """Ultimul titlu de pe fiecare nivel, ex. `# A > ## B > ### C`."""
    path: dict[int, str] = {}
    for m in _HEADER.finditer(markdown):
        level = len(m[1])
        path = {k: v for k, v in path.items() if k < level}
        path[level] = m[0].strip()
    return " > ".join(path[k] for k in sorted(path))


def _message(window: str, i: int, total: int, title: str, done: str) -> str:
    if i == 1:
        where = (
            f"Aceasta e bucata 1 din {total}, începutul documentului „{title}”. "
            "Pune singurul `#` cu titlul documentului."
        )
    else:
        where = (
            f"Aceasta e bucata {i} din {total} a documentului „{title}”. "
            f"Calea de titluri de până acum: {heading_path(done) or '(niciuna)'}. "
            "Continuă ierarhia de acolo."
        )
    return f"{where}\n\n<text>\n{window}\n</text>"


def _clean(text: str) -> str:
    """Scoate gardurile ``` puse uneori în jurul răspunsului și normalizează ş/ţ."""
    text = text.strip()
    if fenced := _FENCE.match(text):
        text = fenced[1].strip()
    return normalize(text)


def _paragraph_shingles(text: str) -> Counter[str]:
    # pe paragrafe: altfel, un titlu scos din text ar lipi literele vecinilor în n-grame noi
    return sum((shingles(p) for p in re.split(r"\n\s*\n", text)), Counter())


def _fraction(part: Counter[str], whole: Counter[str]) -> float:
    return part.total() / whole.total() if whole else 0.0


def _diff(window: str, text: str) -> tuple[float, float]:
    """(lipsă, adăugat) față de fereastră, ca în `pdf.missing_fraction`.

    Titlurile noi sunt cerute de prompt (regula 6), deci nu contează ca text adăugat;
    la „lipsă” rămân, pentru că titlurile scoase din text conțin text original.
    """
    original = _paragraph_shingles(window)
    added = _paragraph_shingles(_HEADER.sub("", text))
    missing = original - _paragraph_shingles(text)
    return _fraction(missing, original), _fraction(added - original, added)


def _restructure_window(
    client: genai.Client, model: str, window: str, message: str, n: int
) -> str:
    best, best_diff = None, (1.0, 1.0)
    for attempt in (1, 2):
        reply = generate(client, model, [PROMPT, message])
        if not reply or not reply.strip():
            logger.warning("  fereastra %d: răspuns gol (încercarea %d)", n, attempt)
            continue
        text = _clean(reply)
        diff = _diff(window, text)
        if max(diff) <= MAX_DIFF:
            return text
        logger.warning(
            "  fereastra %d: lipsă %.1f%%, adăugat %.1f%% (încercarea %d)",
            n,
            diff[0] * 100,
            diff[1] * 100,
            attempt,
        )
        if max(diff) < max(best_diff):
            best, best_diff = text, diff

    if best is None:
        return (
            f"<!-- VERIFICĂ: fereastra {n}, Gemini nu a întors nimic; "
            f"mai jos e textul nerestructurat -->\n\n{window}"
        )
    return (
        f"<!-- VERIFICĂ: fereastra {n}, lipsă {best_diff[0]:.1%}, "
        f"adăugat {best_diff[1]:.1%} -->\n\n{best}"
    )


def restructure(
    body: str,
    title: str,
    client: genai.Client | None = None,
    model: str | None = None,
) -> str:
    client, model = client_and_model(client, model)
    windows = split_windows(body)
    parts: list[str] = []
    for i, window in enumerate(windows, 1):
        logger.info(
            "restructurez „%s”: fereastra %d/%d cu %s ...",
            title,
            i,
            len(windows),
            model,
        )
        message = _message(window, i, len(windows), title, "\n\n".join(parts))
        parts.append(_restructure_window(client, model, window, message, i))
    return "\n\n".join(parts) + "\n"
