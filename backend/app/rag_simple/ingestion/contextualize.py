from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from tenacity import retry, wait_exponential
from tqdm import tqdm

from app.rag_simple.config import CHAT_MODEL
from app.rag_simple.ingestion.load import SourceDocument

wait = wait_exponential(multiplier=1, min=10, max=240)

CONTEXT_PROMPT = """
Here is the full text of a document:
<document>
{document}
</document>

Here is a chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Give a short, succinct context (1-2 sentences) to situate this chunk within the overall document, \
for the purposes of improving search retrieval of the chunk. Answer only with the succinct context \
and nothing else."""

model = init_chat_model(CHAT_MODEL, temperature=0)


@retry(wait=wait)
def _generate_context(document_text: str, chunk_text: str) -> str:
    prompt = CONTEXT_PROMPT.format(document=document_text, chunk=chunk_text)
    response = model.invoke(prompt)
    return response.content.strip()


def add_context(chunks: list[Document], documents: list[SourceDocument]) -> list[Document]:
    documents_by_source = {document.source: document.text for document in documents}
    contextualized = []
    for chunk in tqdm(chunks, desc="Generating chunk context"):
        document_text = documents_by_source[chunk.metadata["source"]]
        context = _generate_context(document_text, chunk.page_content)
        contextualized.append(
            Document(
                page_content=f"{context}\n\n{chunk.page_content}",
                metadata={**chunk.metadata, "original_text": chunk.page_content, "context": context},
            )
        )
    return contextualized
