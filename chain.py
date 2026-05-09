"""
chain.py — LCEL RAG chain and question-answering logic for DocuMind.

LangChain 1.x removed RetrievalQA from the top-level package. The modern
equivalent is an LCEL chain (prompt | llm | parser). Retrieval is handled
explicitly in answer_question() so that similarity scores are accessible for
confidence labelling without a second round-trip to the vector store.
"""

import math
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from prompts import SYSTEM_PROMPT
from retriever import CHROMA_DIR

load_dotenv()

DEFAULT_MODEL = "gpt-3.5-turbo"

# Module-level cache: model name → built LCEL chain
_chain_cache: dict[str, Runnable] = {}


def _build_chat_prompt() -> ChatPromptTemplate:
    """Build a ChatPromptTemplate with the DocuMind system persona.

    SYSTEM_PROMPT no longer contains curly-brace placeholders, so no escaping
    is required. Only {context} and {question} in the human turn are template
    variables that LangChain fills at inference time.

    Returns:
        A ChatPromptTemplate with input_variables ["context", "question"].
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            ),
        ]
    )


def _format_docs_with_source(docs: list) -> str:
    """Format retrieved documents into a context string with inline source labels.

    Each block is prefixed with [Source: filename, Page N] so the LLM can
    copy those labels verbatim into its citations.

    Args:
        docs: List of LangChain Documents with filename/page_number metadata.

    Returns:
        A single string with all chunks separated by a horizontal rule.
    """
    blocks = []
    for doc in docs:
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page_number", "?")
        blocks.append(f"[Source: {filename}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def build_chain(model: str = DEFAULT_MODEL) -> Runnable:
    """Build an LCEL RAG chain: prompt | ChatOpenAI | StrOutputParser.

    The chain expects {"context": str, "question": str} and returns a plain
    string answer. Retrieval and confidence scoring are handled separately in
    answer_question() so that similarity scores stay accessible.

    Args:
        model: OpenAI chat model name. Defaults to "gpt-3.5-turbo".

    Returns:
        A LangChain Runnable ready to call with .invoke({"context": ..., "question": ...}).

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )

    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)
    return _build_chat_prompt() | llm | StrOutputParser()


def _get_chain(model: str = DEFAULT_MODEL) -> Runnable:
    """Return a cached LCEL chain, building it on first use.

    Args:
        model: OpenAI chat model name.

    Returns:
        The cached (or freshly built) Runnable chain.
    """
    if model not in _chain_cache:
        _chain_cache[model] = build_chain(model)
    return _chain_cache[model]


def _relevance_score_to_confidence(score: float) -> str:
    """Map a normalised relevance score in [0, 1] to a confidence label.

    Chroma's similarity_search_with_relevance_scores returns 1.0 for a
    perfect match, 0.0 for completely dissimilar.

    Args:
        score: Relevance score of the top retrieved chunk.

    Returns:
        "high" if score > 0.8, "medium" if > 0.6, "low" otherwise.
    """
    if score > 0.5:
        return "high"
    if score > 0.3:
        return "medium"
    return "low"


def answer_question(
    question: str,
    chat_history: list[Any] | None = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Answer a question using RAG over indexed PDF documents.

    Loads the vector store once per call to retrieve the top-5 chunks and
    their relevance scores, formats them as context, then invokes the LCEL
    chain. Sources are deduplicated from the retrieved documents' metadata.

    The chat_history parameter is accepted for API compatibility but is not
    used by the stateless LCEL chain.

    Args:
        question:     The user's natural-language question.
        chat_history: Prior conversation turns (unused).
        model:        OpenAI chat model to use. Defaults to "gpt-3.5-turbo".

    Returns:
        A dict with:
            answer      — The LLM response string.
            sources     — Deduplicated list of {"filename": str,
                          "page_number": int} dicts, ordered by relevance.
            confidence  — "high", "medium", or "low".

    Raises:
        ValueError:        If question is empty.
        EnvironmentError:  If OPENAI_API_KEY is not configured.
    """
    if not question.strip():
        raise ValueError("Question must not be empty.")

    if not Path(CHROMA_DIR).exists():
        return {
            "answer": (
                "No documents have been indexed yet. "
                "Please upload and ingest PDFs first."
            ),
            "sources": [],
            "confidence": "low",
        }

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file."
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    # similarity_search_with_score returns raw L2 distances (lower = more similar).
    # It is stable across chromadb versions; similarity_search_with_relevance_scores
    # can silently return empty results in chromadb 1.x due to normalization issues.
    docs_and_scores = vectorstore.similarity_search_with_score(question, k=8)

    if not docs_and_scores:
        return {
            "answer": (
                "I don't have enough information in the uploaded documents "
                "to answer this confidently."
            ),
            "sources": [],
            "confidence": "low",
        }

    # Convert L2 distance to a [0, 1] relevance score: 1 − dist/√2.
    # OpenAI embeddings are unit-normalised so L2 ∈ [0, 2] and relevance ∈ [-0.41, 1].
    top_dist = docs_and_scores[0][1]
    top_score = max(0.0, 1.0 - top_dist / math.sqrt(2))

    docs = [doc for doc, _ in docs_and_scores]
    # Prepend source labels so the LLM can cite them verbatim.
    context = _format_docs_with_source(docs)

    chain = _get_chain(model)
    answer: str = chain.invoke({"context": context, "question": question})

    confidence = _relevance_score_to_confidence(top_score)

    seen: set[tuple[str, int]] = set()
    sources: list[dict[str, Any]] = []
    for doc in docs:
        filename = doc.metadata.get("filename", "unknown")
        page_number = doc.metadata.get("page_number", 0)
        key = (filename, page_number)
        if key not in seen:
            seen.add(key)
            sources.append({"filename": filename, "page_number": page_number})

    return {"answer": answer, "sources": sources, "confidence": confidence}
