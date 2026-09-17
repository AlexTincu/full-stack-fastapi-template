import logging

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.rag_simple.ingestion.load import SourceDocument

logger = logging.getLogger(__name__)

HEADERS_TO_SPLIT_ON = [
    ("###", "h3"),
    ("####", "h4"),
    ("#####", "h5"),
]

CHUNK_SIZE_TOKENS = 1000
CHUNK_OVERLAP_TOKENS = 175

header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False)
size_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE_TOKENS, chunk_overlap=CHUNK_OVERLAP_TOKENS
)


def split_documents(documents: list[SourceDocument]) -> list[Document]:
    chunks = []
    for document in documents:
        sections = header_splitter.split_text(document.text)
        for section in sections:
            section.metadata["source"] = document.source
        chunks.extend(size_splitter.split_documents(sections))
    logger.info("Split into %d chunks", len(chunks))
    return chunks
