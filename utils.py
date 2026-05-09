"""
utils.py — Shared helper utilities for DocuMind.

Covers source formatting, PDF validation, database management, and
inspection of what is currently indexed in the ChromaDB store.
"""

import os
import shutil
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = ".chroma"


def format_sources(sources_list: list[dict[str, Any]]) -> str:
    """Format a list of source dicts into a human-readable citation block.

    Accepts the 'sources' list returned by chain.answer_question(), which
    contains dicts with 'filename' and 'page_number' keys.

    Args:
        sources_list: List of {"filename": str, "page_number": int} dicts.

    Returns:
        A formatted string starting with '📄 Sources:' followed by one
        bullet per unique source.  Returns an empty string when the list
        is empty so callers can do a simple truthiness check.

    Examples:
        >>> format_sources([{"filename": "report.pdf", "page_number": 12}])
        '📄 Sources:\\n- Page 12 from report.pdf'
    """
    if not sources_list:
        return ""

    lines = ["📄 Sources:"]
    for src in sources_list:
        filename = src.get("filename", "unknown")
        page = src.get("page_number", "?")
        lines.append(f"- Page {page} from {filename}")

    return "\n".join(lines)


def validate_pdf(file_path: str) -> tuple[bool, str]:
    """Check whether a file exists and is a readable, non-empty PDF.

    Opens the file with PyMuPDF to confirm it is a valid PDF structure,
    not just a file with a .pdf extension.

    Args:
        file_path: Path to the file to validate.

    Returns:
        A (is_valid, message) tuple.  is_valid is True only when all checks
        pass; message describes the failure reason on False, or confirms
        success with a page count on True.
    """
    path = Path(file_path)

    if not path.exists():
        return False, f"File not found: {file_path}"

    if not path.is_file():
        return False, f"Path is not a file: {file_path}"

    if path.suffix.lower() != ".pdf":
        return False, f"File does not have a .pdf extension: {path.name}"

    if path.stat().st_size == 0:
        return False, f"File is empty: {path.name}"

    try:
        pdf = fitz.open(str(path))
    except Exception as exc:
        return False, f"Cannot open as PDF: {exc}"

    with pdf:
        page_count = len(pdf)
        if page_count == 0:
            return False, f"PDF has no pages: {path.name}"

    return True, f"Valid PDF with {page_count} page(s): {path.name}"


def clear_database() -> tuple[bool, str]:
    """Delete the persisted ChromaDB directory to allow a full re-index.

    Removes CHROMA_DIR and all its contents.  After calling this function
    the application will behave as if no documents have ever been ingested.

    Returns:
        A (success, message) tuple.  success is False if the directory does
        not exist (nothing to clear) or if deletion fails.
    """
    chroma_path = Path(CHROMA_DIR)

    if not chroma_path.exists():
        return False, f"No database found at '{CHROMA_DIR}' — nothing to clear."

    try:
        shutil.rmtree(chroma_path)
    except OSError as exc:
        return False, f"Failed to delete '{CHROMA_DIR}': {exc}"

    return True, f"Database cleared. '{CHROMA_DIR}' has been removed."


def get_indexed_files() -> list[str]:
    """Return the unique filenames of all documents currently in ChromaDB.

    Queries the ChromaDB collection's stored metadata directly — no
    embedding or LLM call is made.

    Returns:
        A sorted list of unique filename strings (e.g. ["manual.pdf",
        "report.pdf"]).  Returns an empty list when the database does not
        exist or contains no documents.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set (required to
            initialise the embedding function for the Chroma client).
    """
    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists():
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )

    # Import here to avoid a hard dependency cycle at module load time.
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=api_key,
    )
    vectorstore = Chroma(
        persist_directory=str(chroma_path),
        embedding_function=embeddings,
    )

    # get() with no arguments fetches all stored entries.
    collection = vectorstore.get()
    metadatas: list[dict] = collection.get("metadatas") or []

    filenames = {
        meta.get("filename")
        for meta in metadatas
        if meta.get("filename")
    }
    return sorted(filenames)
