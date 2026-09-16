from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential

from app.rag_simple.config import CHAT_MODEL

wait = wait_exponential(multiplier=1, min=10, max=240)

model = init_chat_model(CHAT_MODEL, temperature=0)

RERANK_SYSTEM_PROMPT = """You are a document re-ranker.
You are given a question and a list of chunks of text retrieved from a knowledge base.
The chunks are provided in no particular order of relevance.
Rank all of the chunks by relevance to the question, most relevant first.
Reply only with the ranked list of chunk ids, nothing else. Include every chunk id you were given, exactly once."""


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="Chunk ids ordered from most to least relevant to the question"
    )


@retry(wait=wait)
def rerank(question: str, chunks: list[Document]) -> list[Document]:
    if not chunks:
        return []

    user_prompt = f"Question:\n{question}\n\nChunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the ranked list of chunk ids, most relevant first."

    structured_model = model.with_structured_output(RankOrder)
    result = structured_model.invoke(
        [
            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    # Defensive against the LLM skipping or repeating ids: keep valid, unique
    # ids in the order given, then append anything missing at the end.
    seen: set[int] = set()
    ranked: list[Document] = []
    for chunk_id in result.order:
        if 1 <= chunk_id <= len(chunks) and chunk_id not in seen:
            seen.add(chunk_id)
            ranked.append(chunks[chunk_id - 1])
    for chunk_id in range(1, len(chunks) + 1):
        if chunk_id not in seen:
            ranked.append(chunks[chunk_id - 1])
    return ranked
