"""Ce au comun convertoarele care folosesc Gemini (audio, pdf, restructurare, întrebări): clientul, thinking-ul, retry-ul."""

import logging

from google import genai
from google.genai import errors, types
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def client_and_model(
    client: genai.Client | None, model: str | None
) -> tuple[genai.Client, str]:
    """Completează ce lipsește din `.env` (GEMINI_API_KEY, GEMINI_MODEL); testele le dau pe ambele."""
    if client is None or model is None:
        from app.core.config import settings

        client = client or genai.Client(api_key=settings.GEMINI_API_KEY)
        model = model or settings.GEMINI_MODEL
    return client, model


def thinking_config(model: str) -> types.ThinkingConfig:
    """Conversia nu are nevoie de raționament: pe 2.5 thinking-ul doar costă (și, la PDF, a dus la omisiuni)."""
    if model.startswith("gemini-2.5"):
        return types.ThinkingConfig(thinking_budget=0)
    return types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)


def log_retry(retry_state):
    logger.warning(
        "[retry] %s attempt %d failed: %r — retrying...",
        retry_state.fn.__name__,
        retry_state.attempt_number,
        retry_state.outcome.exception(),
    )


@retry(
    retry=retry_if_exception_type(errors.APIError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=log_retry,
    reraise=True,
)
def generate(client: genai.Client, model: str, contents: list) -> str | None:
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(thinking_config=thinking_config(model)),
    )
    return response.text


@retry(
    retry=retry_if_exception_type(errors.APIError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=log_retry,
    reraise=True,
)
def generate_structured[M: BaseModel](
    client: genai.Client, model: str, contents: list, schema: type[M]
) -> M | None:
    """Răspuns JSON validat pe `schema` (`response.parsed`); `None` dacă nu se validează."""
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            thinking_config=thinking_config(model),
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    parsed = response.parsed
    return parsed if isinstance(parsed, schema) else None
