from langchain.chat_models import init_chat_model
from tenacity import retry, wait_exponential

from app.rag_simple.config import CHAT_MODEL

wait = wait_exponential(multiplier=1, min=10, max=240)

model = init_chat_model(CHAT_MODEL, temperature=0)

REWRITE_PROMPT = """You are in a conversation with a user.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of the conversation so far:
{history}

And this is the user's current question:
{question}

Since the conversation is contextual, understand the meaning of the user's question and add \
details based on the history. Condense everything into a single, contextually-rich, VERY short \
and specific query, most likely to surface relevant content in the Knowledge Base.

If there is no history, or the question is already self-contained, just return the question as is.

IMPORTANT: Respond ONLY with the precise knowledge-base query, nothing else."""


@retry(wait=wait)
def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    """Condense the question + conversation history into a standalone search query."""
    history = history or []
    prompt = REWRITE_PROMPT.format(history=history, question=question)
    response = model.invoke(prompt)
    return response.content.strip()
