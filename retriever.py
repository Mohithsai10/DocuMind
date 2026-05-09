"""
retriever.py — ChromaDB retrieval layer for DocuMind.

Loads the persisted vector store produced by ingest.py and exposes two
retrieval interfaces: a standard LangChain retriever and a metadata-rich
helper that returns chunk text alongside citation fields.
"""

import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.retrievers import BaseRetriever

load_dotenv()

CHROMA_DIR = ".chroma"
DEFAULT_K = 5


class ChunkResult(TypedDict):
    """A single retrieval result with text and citation metadata."""

    text: str
    filename: str
    page_number: int
    source: str


def _load_vectorstore() -> Chroma:
    """Load the persisted ChromaDB vector store from disk.

    Uses the same OpenAI embedding model (text-embedding-ada-002) that was
    used during ingestion so that query vectors are in the same space.

    Returns:
        A Chroma instance backed by the on-disk collection in CHROMA_DIR.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is missing or empty.
        FileNotFoundError: If the CHROMA_DIR directory does not exist,
            meaning no documents have been ingested yet.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )

    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists():
        raise FileNotFoundError(
            f"Vector store directory '{CHROMA_DIR}' not found. "
            "Run ingest_pdfs() first to index your documents."
        )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=api_key,
    )

    return Chroma(
        persist_directory=str(chroma_path),
        embedding_function=embeddings,
    )


def get_retriever(k: int = DEFAULT_K) -> BaseRetriever:
    """Return a LangChain retriever over the persisted ChromaDB store.

    The retriever uses cosine-similarity search and returns the top-k most
    relevant chunks. It can be passed directly into a LangChain chain or
    used with .invoke() / .get_relevant_documents().

    Args:
        k: Number of chunks to return per query. Defaults to 5.

    Returns:
        A configured VectorStoreRetriever instance.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not configured.
        FileNotFoundError: If the vector store has not been created yet.
    """
    vectorstore = _load_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def retrieve_with_metadata(query: str, k: int = DEFAULT_K) -> list[ChunkResult]:
    """Retrieve the top-k chunks most relevant to query, with citation metadata.

    Returns structured results that include both the chunk text and the
    metadata fields stored during ingestion (filename, page_number, source),
    ready for use in prompt assembly and citation rendering.

    Args:
        query: The natural-language question or search string.
        k: Number of chunks to return. Defaults to 5.

    Returns:
        A list of ChunkResult dicts ordered by relevance (most relevant first),
        each containing:
            - text:        The raw chunk text.
            - filename:    Original PDF filename (e.g. "report.pdf").
            - page_number: 1-based page number within that PDF.
            - source:      Absolute path to the source PDF.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not configured.
        FileNotFoundError: If the vector store has not been created yet.
        ValueError: If query is empty.
    """
    if not query.strip():
        raise ValueError("Query must not be empty.")

    vectorstore = _load_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)

    results: list[ChunkResult] = []
    for doc in docs:
        meta = doc.metadata
        results.append(
            ChunkResult(
                text=doc.page_content,
                filename=meta.get("filename", "unknown"),
                page_number=meta.get("page_number", 0),
                source=meta.get("source", ""),
            )
        )

    return results
