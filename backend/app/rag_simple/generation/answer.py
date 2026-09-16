from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from tenacity import retry, wait_exponential

from app.rag_simple.config import CHAT_MODEL
from app.rag_simple.retrieval.pipeline import retrieve

wait = wait_exponential(multiplier=1, min=10, max=240)

model = init_chat_model(CHAT_MODEL, temperature=0)

SYSTEM_PROMPT = """Ești un asistent cunoscător, care răspunde la întrebări despre macrobiotică, \
sănătate naturistă și spiritualitate, pe baza unei baze de cunoștințe curatoriate (seminarii \
Michio Kushi și alte materiale despre macrobiotică).

Răspunsul tău va fi evaluat pentru acuratețe, relevanță și completitudine, deci răspunde doar la \
întrebare și răspunde-i complet. Dacă informația nu se regăsește în contextul de mai jos, spune \
clar că nu știi — nu inventa.

Context relevant din baza de cunoștințe:
{context}

Pe baza acestui context, răspunde la întrebarea userului. Fii precis, relevant și complet."""


def format_context(chunks: list[Document]) -> str:
    return "\n\n".join(
        f"Extras din {chunk.metadata.get('source', '?')}:\n{chunk.page_content}" for chunk in chunks
    )


def make_messages(question: str, history: list[dict], chunks: list[Document]) -> list[dict]:
    system_prompt = SYSTEM_PROMPT.format(context=format_context(chunks))
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


@retry(wait=wait)
def answer_question(question: str, history: list[dict] | None = None) -> tuple[str, list[Document]]:
    """Answer a question using RAG. Returns (answer, retrieved chunks)."""
    history = history or []
    chunks = retrieve(question, history)
    messages = make_messages(question, history, chunks)
    response = model.invoke(messages)
    return response.content, chunks
