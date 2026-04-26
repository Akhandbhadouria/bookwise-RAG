"""
vector_store.py — FAISS Embedding & Retrieval

This module handles:
1. Converting text chunks into dense vector embeddings using
   HuggingFace's all-MiniLM-L6-v2 model (runs locally, 100% FREE).
2. Storing those vectors in a FAISS index for fast similarity search.
3. Querying the index to find chunks most relevant to a user question.

Why FAISS?
  - It's a battle-tested library from Meta for efficient similarity search.
  - Works entirely in-memory (no external server needed).
  - For a single book (a few thousand chunks), it's instant.

Why all-MiniLM-L6-v2?
  - Completely FREE — runs locally on your CPU, no API key needed.
  - Produces high-quality 384-dimensional embeddings.
  - Small model (~80 MB), fast inference, great for semantic search.
  - One of the most popular sentence-transformer models on HuggingFace.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def create_vector_store(chunks: list[Document]) -> FAISS:
    """
    Build a FAISS vector store from a list of text chunks.

    Args:
        chunks: List of LangChain Document objects (from pdf_parser.chunk_documents).

    Returns:
        A FAISS vector store instance ready for similarity search.

    How it works:
        1. The HuggingFaceEmbeddings wrapper loads the all-MiniLM-L6-v2
           model locally and converts each chunk's text into a 384-dim vector.
        2. FAISS.from_documents builds an IndexFlatL2 index from those
           vectors and keeps a mapping from vector index → Document.
    """

    # Initialize the embedding model — runs locally, no API key needed!
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},       # use CPU (works everywhere)
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity
    )

    # Build the FAISS index — this runs locally, so it's fast and free.
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store


def search_similar_chunks(
    vector_store: FAISS,
    query: str,
    k: int = 4,
) -> list[Document]:
    """
    Find the top-k chunks most semantically similar to the query.

    Args:
        vector_store: The FAISS index built by create_vector_store.
        query       : The user's natural-language question.
        k           : Number of top results to return (default 4).

    Returns:
        A list of Document objects ranked by relevance. Each Document
        contains the chunk text and its metadata (including page number).

    Under the hood:
        1. The query string is embedded into the same 384-dim space.
        2. FAISS computes L2 distances between the query vector and
           every stored vector.
        3. The k nearest vectors (= most similar chunks) are returned.
    """

    results = vector_store.similarity_search(query, k=k)
    return results
