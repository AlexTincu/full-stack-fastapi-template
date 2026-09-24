import io
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from google import genai

from app.knowledge_prep.converters import CONVERTERS
from app.knowledge_prep.converters.audio import AUDIO_EXTENSIONS, transcribe_audio
from app.knowledge_prep.converters.gemini import thinking_config


class FakeClient:
    def __init__(self, text: str | None = "Bună seara.", fail: Exception | None = None):
        self.uploads: list[dict] = []
        self.deleted: list[str] = []
        self.requests: list[dict] = []
        self._text, self._fail = text, fail
        self.files = SimpleNamespace(upload=self._upload, delete=self._delete)
        self.models = SimpleNamespace(generate_content=self._generate)

    def _upload(self, *, file, config):
        self.uploads.append({"file": file, **config})
        return SimpleNamespace(name="files/abc")

    def _delete(self, *, name):
        self.deleted.append(name)

    def _generate(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        if self._fail:
            raise self._fail
        return SimpleNamespace(text=self._text)


def transcribe(audio: Path, client: FakeClient) -> str:
    return transcribe_audio(
        audio, client=cast(genai.Client, client), model="gemini-2.5-flash"
    )


def test_transcribe_audio_returns_text_and_cleans_up_upload(tmp_path: Path):
    audio = tmp_path / "rostul suferintei.opus"
    audio.write_bytes(b"x")
    client = FakeClient("  Bună seara. Haideți să vă explic.  ")

    text = transcribe(audio, client)

    assert text == "Bună seara. Haideți să vă explic.\n"
    assert client.uploads[0]["mime_type"] == "audio/ogg"
    assert "cuvant cu cuvant" in client.requests[0]["contents"][0]
    assert client.deleted == ["files/abc"]


def test_transcribe_audio_uploads_an_open_file_not_a_bare_path(tmp_path: Path):
    """google-genai pune numele fișierului într-un header HTTP ASCII-only
    (`X-Goog-Upload-File-Name`) doar când primește o cale; un fișier deschis
    ocolește acel cod și evită crash-ul pe nume cu diacritice (docs/taskuri.md)."""
    audio = tmp_path / "Înțelepciunea aplecarii.ogg"
    audio.write_bytes(b"x")
    client = FakeClient()

    transcribe(audio, client)

    assert isinstance(client.uploads[0]["file"], io.IOBase)


def test_transcribe_audio_rejects_empty_response(tmp_path: Path):
    audio = tmp_path / "a.ogg"
    audio.write_bytes(b"x")

    with pytest.raises(ValueError, match="gol"):
        transcribe(audio, FakeClient(None))


def test_transcribe_audio_deletes_upload_when_generation_fails(tmp_path: Path):
    audio = tmp_path / "a.ogg"
    audio.write_bytes(b"x")
    client = FakeClient(fail=RuntimeError("boom"))

    with pytest.raises(RuntimeError):
        transcribe(audio, client)
    assert client.deleted == ["files/abc"]


def test_thinking_is_disabled_for_25_and_low_for_newer_models():
    assert thinking_config("gemini-2.5-flash").thinking_budget == 0
    assert thinking_config("gemini-3.8-flash").thinking_level == "LOW"


def test_every_audio_extension_is_registered():
    assert all(CONVERTERS[ext] is transcribe_audio for ext in AUDIO_EXTENSIONS)
