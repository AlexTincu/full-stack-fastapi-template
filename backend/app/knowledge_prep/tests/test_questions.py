from types import SimpleNamespace
from typing import cast

from google import genai

from app.knowledge_prep.questions import GeneratedQuestions, generate_questions


class FakeClient:
    """Întoarce pe rând `parsed` din `replies`; `None` = JSON care nu s-a validat."""

    def __init__(self, replies: list[GeneratedQuestions | None]):
        self.requests: list[dict] = []
        self._replies = list(replies)
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(parsed=self._replies.pop(0))


def ask(client: FakeClient, *, whole: bool = False) -> list[str]:
    return generate_questions(
        "Grâul încolțit",
        ["Grâul încolțit", "Contraindicații"],
        "Text chunk.",
        whole,
        client=cast(genai.Client, client),
        model="gemini-2.5-flash",
    )


def test_sends_title_heading_path_text_and_json_schema():
    client = FakeClient(
        [GeneratedQuestions(questions=["Cine nu are voie să consume grâu încolțit?"])]
    )
    assert ask(client) == ["Cine nu are voie să consume grâu încolțit?"]
    request = client.requests[0]
    message = request["contents"][1]
    assert "„Grâul încolțit”" in message
    assert "Grâul încolțit > Contraindicații" in message
    assert "<text>\nText chunk.\n</text>" in message
    assert "documentul întreg" not in message
    config = request["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema is GeneratedQuestions
    assert config.thinking_config.thinking_budget == 0


def test_whole_document_asks_to_cover_everything():
    client = FakeClient([GeneratedQuestions(questions=["Ce conține regimul?"])])
    ask(client, whole=True)
    assert "documentul întreg" in client.requests[0]["contents"][1]


def test_cleans_empty_duplicate_and_extra_questions():
    raw = [
        "  Ce este grâul încolțit? ",
        "",
        "ce este grâul  încolțit?",
        "Q2?",
        "Q3?",
        "Q4?",
        "Q5?",
        "Q6?",
    ]
    client = FakeClient([GeneratedQuestions(questions=raw)])
    assert ask(client) == ["Ce este grâul încolțit?", "Q2?", "Q3?", "Q4?", "Q5?"]


def test_unparseable_reply_returns_no_questions():
    assert ask(FakeClient([None])) == []
