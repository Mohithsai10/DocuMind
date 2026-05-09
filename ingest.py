"""
ingest.py — PDF ingestion pipeline for DocuMind.

Extracts text from PDFs page-by-page, splits into chunks, embeds with
OpenAI text-embedding-ada-002, and persists to a local ChromaDB store.
"""

import os
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

CHROMA_DIR = ".chroma"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


def _extract_page_text(page: fitz.Page) -> str:
    """Extract all text from a single page using the block-level dict API.

    get_text("dict") returns every text block with its bounding box, lines,
    and character spans. Table cells stored as individual positioned fragments
    (which get_text() can conflate or silently drop) are captured here as
    separate blocks and then assembled in explicit top-to-bottom,
    left-to-right order based on their bounding-box coordinates.

    Args:
        page: An open PyMuPDF Page object.

    Returns:
        A single string with all text content joined by newlines, preserving
        the spatial reading order of both prose and table cells.
    """
    page_dict = page.get_text("dict")

    # type=0 → text block; type=1 → image block (skip)
    text_blocks = [b for b in page_dict.get("blocks", []) if b.get("type") == 0]

    # Sort top-to-bottom (y0), then left-to-right (x0) so table rows read
    # in natural order even when cells are stored out of sequence in the PDF.
    text_blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

    lines: list[str] = []
    for block in text_blocks:
        for line in block.get("lines", []):
            span_text = " ".join(
                s["text"] for s in line.get("spans", []) if s.get("text", "").strip()
            )
            if span_text.strip():
                lines.append(span_text)

    return "\n".join(lines)


def extract_pages(pdf_path: str) -> list[Document]:
    """Extract text from every page of a PDF as LangChain Documents.

    Each Document carries metadata with the original filename and 1-based
    page number so citations can be generated downstream.

    Args:
        pdf_path: Absolute or relative path to a PDF file.

    Returns:
        A list of Documents, one per non-empty page.

    Raises:
        FileNotFoundError: If the path does not point to an existing file.
        ValueError: If the file cannot be opened as a PDF.
    """
    path = Path(pdf_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path}")

    try:
        pdf = fitz.open(str(path))
    except Exception as exc:
        raise ValueError(f"Could not open PDF '{pdf_path}': {exc}") from exc

    docs: list[Document] = []
    with pdf:
        for page_index in range(len(pdf)):
            text = _extract_page_text(pdf[page_index])
            if not text.strip():
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "filename": path.name,
                        "page_number": page_index + 1,
                        "source": str(path),
                    },
                )
            )

    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Split Documents into smaller, overlapping chunks for embedding.

    Metadata (filename, page_number) is preserved on every child chunk so
    citation information is never lost after splitting.

    Args:
        docs: Documents to split, typically the output of extract_pages().

    Returns:
        A flat list of chunk Documents ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """Embed chunks with OpenAI and persist them to ChromaDB.

    Reads OPENAI_API_KEY from the environment (loaded via python-dotenv).
    ChromaDB is stored in CHROMA_DIR and is automatically persistent when
    persist_directory is supplied (chromadb >= 0.4.0).

    Args:
        chunks: Chunk Documents to embed and index.

    Returns:
        The populated Chroma vector store instance.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is missing or empty.
        RuntimeError: If the Chroma indexing step fails.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=api_key,
    )

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )
    except Exception as exc:
        raise RuntimeError(f"ChromaDB indexing failed: {exc}") from exc

    # persist() is a no-op in chromadb >= 0.4 (auto-persisted), but kept
    # for compatibility with older langchain-community builds.
    try:
        vectorstore.persist()
    except AttributeError:
        pass

    return vectorstore


def ingest_pdfs(pdf_paths: list[str]) -> int:
    """Ingest a list of PDF files into the ChromaDB vector store.

    Orchestrates the full pipeline: extraction → splitting → embedding →
    persistence. Progress is printed to stdout at each stage.

    Args:
        pdf_paths: Paths to the PDF files to ingest.

    Returns:
        Total number of chunks indexed across all files.

    Raises:
        FileNotFoundError: If any PDF path does not exist.
        ValueError: If any file is not a valid PDF.
        EnvironmentError: If OPENAI_API_KEY is not configured.
        RuntimeError: If ChromaDB indexing fails.
    """
    all_docs: list[Document] = []

    for pdf_path in pdf_paths:
        print(f"[ingest] Extracting: {pdf_path}")
        pages = extract_pages(pdf_path)
        print(f"         {len(pages)} page(s) extracted")
        all_docs.extend(pages)

    if not all_docs:
        print("[ingest] No extractable text found in the provided PDFs.")
        return 0

    print(f"[ingest] Splitting {len(all_docs)} page(s) into chunks...")
    chunks = split_documents(all_docs)
    print(f"         {len(chunks)} chunk(s) created")

    print(f"[ingest] Embedding and storing in '{CHROMA_DIR}/'...")
    build_vectorstore(chunks)
    print(f"[ingest] Done — {len(chunks)} chunk(s) indexed.")

    return len(chunks)
