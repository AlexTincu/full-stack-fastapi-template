from types import SimpleNamespace
from typing import cast

import pytest
from google import genai

from app.knowledge_prep.restructure import heading_path, restructure, split_windows

PARA = "Grâul încolțit curăță sângele și dă putere organismului slăbit"
SECOND = "Al doilea paragraf important despre germeni si vitamine"


class FakeClient:
    """Întoarce pe rând textele din `replies`; `None` = răspuns gol."""

    def __init__(self, replies: list[str | None]):
        self.messages: list[str] = []
        self._replies = list(replies)
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, *, model, contents, config):
        self.messages.append(contents[1])
        return SimpleNamespace(text=self._replies.pop(0))


def run(body: str, client: FakeClient, title: str = "Grâul încolțit") -> str:
    return restructure(body, title, cast(genai.Client, client), "gemini-2.5-flash")


def test_split_windows_cuts_at_paragraphs_and_keeps_long_paragraph_alone():
    paras = ["a" * 40, "b" * 40, "c" * 150, "d" * 40]
    windows = split_windows("\n\n".join(paras) + "\n", max_chars=100)
    assert windows == [f"{'a' * 40}\n\n{'b' * 40}", "c" * 150, "d" * 40]


def test_heading_path_keeps_last_header_per_level():
    md = "# A\n\n## B\n\n### C\n\ntext\n\n## D\n\ntext #nu\n"
    assert heading_path(md) == "# A > ## D"
    assert heading_path("fără titluri\n") == ""


def test_restructure_passes_heading_path_to_next_window(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("app.knowledge_prep.restructure.WINDOW_CHARS", 80)
    first, second = f"{PARA} unu", f"{PARA} doi"
    client = FakeClient([f"# Grâul încolțit\n\n## Beneficii\n\n{first}", second])

    out = run(f"{first}\n\n{second}\n", client)

    assert out == f"# Grâul încolțit\n\n## Beneficii\n\n{first}\n\n{second}\n"
    assert "bucata 1 din 2" in client.messages[0]
    assert "„Grâul încolțit”" in client.messages[0]
    assert "# Grâul încolțit > ## Beneficii" in client.messages[1]
    assert f"<text>\n{second}\n</text>" in client.messages[1]


def test_restructure_strips_code_fences_and_normalizes_cedilla():
    client = FakeClient([f"```markdown\n# Titlu\n\n{PARA} şi ţara\n```"])
    assert run(f"{PARA} si tara\n", client) == f"# Titlu\n\n{PARA} și țara\n"


def test_restructure_new_headers_do_not_count_as_added_text():
    reply = f"# Grâul încolțit\n\n## Beneficiile curei cu grâu încolțit\n\n{PARA}"
    client = FakeClient([reply])
    assert run(f"{PARA}\n", client) == reply + "\n"
    assert len(client.messages) == 1


def test_restructure_retries_window_with_missing_text_then_accepts():
    body = f"{PARA}\n\n{SECOND}\n"
    client = FakeClient([PARA, body])
    assert run(body, client) == body
    assert len(client.messages) == 2


def test_restructure_marks_window_still_off_after_retry_keeping_better_attempt():
    body = f"{PARA}\n\n{SECOND}\n"
    added = f"{body}\nO frază inventată de model, care nu era în original deloc.\n"
    client = FakeClient([PARA, added])

    out = run(body, client)

    assert out.startswith("<!-- VERIFICĂ: fereastra 1, lipsă 0.0%, adăugat ")
    assert "O frază inventată" in out


def test_restructure_keeps_original_window_when_gemini_returns_nothing():
    client = FakeClient([None, "  "])
    assert run(f"{PARA}\n", client) == (
        "<!-- VERIFICĂ: fereastra 1, Gemini nu a întors nimic; "
        f"mai jos e textul nerestructurat -->\n\n{PARA}\n"
    )
