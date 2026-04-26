"""
rag_pipeline.py — Core RAG Logic

This module ties retrieval and generation together:
1. Takes the user's question and retrieves relevant chunks from FAISS.
2. Constructs a tightly constrained prompt that includes only the
   retrieved context, preventing the LLM from hallucinating.
3. Calls Groq (FREE tier) via LangChain and returns the answer along
   with the source page numbers.

Design decisions:
  - The system prompt explicitly forbids answering from general knowledge.
  - Chat history (last 5 messages) is included so the LLM can handle
    follow-up questions like "Tell me more about that character."

Why Groq?
  - Completely FREE: 30 requests/min, 14,400 requests/day.
  - Runs Llama 3.3 70B — a powerful open-source model.
  - Extremely fast inference (runs on custom LPU hardware).
  - Free API key from https://console.groq.com (no credit card needed).
"""

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.vectorstores import FAISS

from vector_store import search_similar_chunks


# ── System prompt ──────────────────────────────────────────────
# This is the most critical piece for preventing hallucination.

SYSTEM_PROMPT = """You are a helpful book assistant. Your ONLY job is to
answer questions using the provided context from the book.

STRICT RULES:
1. Answer ONLY from the provided context below.
2. If the answer is not in the context, reply EXACTLY:
   "❌ Sorry, the answer was not found in the book."
3. Do NOT use any outside knowledge. Do NOT guess.
4. Keep answers clear, concise, and well-structured.
5. When possible, quote relevant text from the book.
6. Always mention which page(s) the information comes from.
"""


def build_prompt(
    context_chunks: list,
    question: str,
    chat_history: list[dict],
) -> list:
    """
    Assemble the full message list for the LLM call.

    Args:
        context_chunks: Retrieved Document objects (with .page_content and .metadata).
        question      : The user's current question.
        chat_history  : List of {"role": "user"/"assistant", "content": "..."} dicts.

    Returns:
        A list of LangChain message objects:
          [SystemMessage, ...history messages, HumanMessage with context + question]

    Why include chat history?
        So the user can ask follow-up questions like:
          "Who is that character?" → refers to a name in the previous answer.
        We limit history to the last 5 pairs to stay within the token limit.
    """

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # ── Inject chat history (last 5 exchanges) ────────────────
    recent_history = chat_history[-5:]  # keep only the latest 5
    for msg in recent_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # ── Build context block from retrieved chunks ─────────────
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        page = chunk.metadata.get("page", "?")
        context_parts.append(
            f"[Chunk {i} | Page {page}]\n{chunk.page_content}"
        )
    context_block = "\n\n---\n\n".join(context_parts)

    # ── Compose the final user message ────────────────────────
    user_message = (
        f"CONTEXT FROM THE BOOK:\n\n{context_block}\n\n"
        f"---\n\n"
        f"QUESTION: {question}\n\n"
        f"Remember: Answer ONLY from the context above. "
        f"Mention the page number(s) in your answer."
    )

    messages.append(HumanMessage(content=user_message))

    return messages


def get_answer(
    vector_store: FAISS,
    question: str,
    chat_history: list[dict],
) -> dict:
    """
    End-to-end RAG: retrieve → prompt → generate.

    Args:
        vector_store: The FAISS index containing book embeddings.
        question    : The user's question.
        chat_history: Previous conversation messages.

    Returns:
        A dict with two keys:
          - "answer"       : The LLM's response string.
          - "source_pages" : A sorted list of unique page numbers
                             from the retrieved chunks.
    """

    # Step 1 — Retrieve the most relevant chunks
    relevant_chunks = search_similar_chunks(vector_store, question, k=4)

    # Step 2 — Collect source page numbers for display
    source_pages = sorted(
        set(chunk.metadata.get("page", 0) for chunk in relevant_chunks)
    )

    # Step 3 — Build the constrained prompt
    messages = build_prompt(relevant_chunks, question, chat_history)

    # Step 4 — Call Groq (FREE tier — Llama 3.3 70B)
    # Uses GROQ_API_KEY from the environment automatically
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",  # free, powerful open-source model
        temperature=0.2,                        # low temp → factual answers
        max_tokens=1024,                        # enough for a detailed answer
    )

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "source_pages": source_pages,
    }
